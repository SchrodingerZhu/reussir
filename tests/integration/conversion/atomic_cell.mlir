// RUN: %reussir-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=SCF
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-acquire-drop-expansion,reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!atomic_i64 = !reussir.rc<!reussir.cell<i64 atomic> atomic>
!atomic_f32 = !reussir.rc<!reussir.cell<f32 atomic> atomic>

module {
  // Atomic Cell creation is structural initialization, not an atomic access,
  // so ConvertToSTD still materializes the RC box and payload slot.
  // SCF-LABEL: func.func @atomic_create
  // SCF: reussir.rc.create
  // SCF-NOT: reussir.cell.create
  func.func @atomic_create(%value: i64) -> !atomic_i64 {
    %cell = reussir.cell.create value(%value : i64) : !atomic_i64
    return %cell : !atomic_i64
  }

  // ROUNDTRIP-LABEL: func.func @atomic_get
  // ROUNDTRIP-SAME: !reussir.rc<!reussir.cell<i64 atomic> atomic>
  // SCF-LABEL: func.func @atomic_get
  // SCF: reussir.cell.get
  // LLVM-LABEL: define i64 @atomic_get
  // LLVM: load atomic i64, ptr %{{.+}} acquire, align 8
  func.func @atomic_get(%cell: !atomic_i64) -> i64 {
    %value = reussir.cell.get(%cell : !atomic_i64) : i64
    return %value : i64
  }

  // ROUNDTRIP-LABEL: func.func @atomic_get_monotonic
  // ROUNDTRIP: reussir.cell.get(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(monotonic) : i64
  // LLVM-LABEL: define i64 @atomic_get_monotonic
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  func.func @atomic_get_monotonic(%cell: !atomic_i64) -> i64 {
    %value = reussir.cell.get(%cell : !atomic_i64) ordering(monotonic) : i64
    return %value : i64
  }

  // SCF-LABEL: func.func @atomic_set
  // SCF: reussir.cell.set
  // LLVM-LABEL: define void @atomic_set
  // LLVM: store atomic i64 %{{.+}}, ptr %{{.+}} release, align 8
  func.func @atomic_set(%value: i64, %cell: !atomic_i64) {
    reussir.cell.set(%value : i64, %cell : !atomic_i64)
    return
  }

  // ROUNDTRIP-LABEL: func.func @atomic_set_seq_cst
  // ROUNDTRIP: reussir.cell.set(%{{.+}} : i64, %{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(seq_cst)
  // LLVM-LABEL: define void @atomic_set_seq_cst
  // LLVM: store atomic i64 %{{.+}}, ptr %{{.+}} seq_cst, align 8
  func.func @atomic_set_seq_cst(%value: i64, %cell: !atomic_i64) {
    reussir.cell.set(%value : i64, %cell : !atomic_i64) ordering(seq_cst)
    return
  }

  // ROUNDTRIP-LABEL: func.func @direct_add
  // ROUNDTRIP: reussir.cell.rmw addi(%{{.+}} : i64, %{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(acquire) -> i64
  // SCF-LABEL: func.func @direct_add
  // SCF: reussir.cell.rmw addi
  // LLVM-LABEL: define i64 @direct_add
  // LLVM: atomicrmw add ptr %{{.+}}, i64 %{{.+}} acquire, align 8
  func.func @direct_add(%delta: i64, %cell: !atomic_i64) -> i64 {
    %old = reussir.cell.rmw addi(%delta : i64, %cell : !atomic_i64) ordering(acquire) -> i64
    return %old : i64
  }

  // LLVM has no atomicrmw multiply, so ConvertToSTD expands the direct
  // multiply into an scf.while retry loop committed with the weak
  // ref.cmpxchg bridge; the LLVM lowering never touches control flow.
  // SCF-LABEL: func.func @direct_multiply
  // SCF: %[[MUL_INIT:.+]] = reussir.cell.get(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(monotonic) : i64
  // SCF: scf.while (%[[MUL_EXPECTED:.+]] = %[[MUL_INIT]]) : (i64) -> i64
  // SCF: arith.muli %[[MUL_EXPECTED]], %{{.+}} : i64
  // SCF: %[[MUL_OBSERVED:.+]], %[[MUL_OK:.+]] = reussir.ref.cmpxchg(%[[MUL_EXPECTED]] : i64, %{{.+}} : i64, %{{.+}} : !reussir.ref<i64 field atomic>) weak ordering(acq_rel) : i64, i1
  // SCF: %[[MUL_RETRY:.+]] = arith.xori %[[MUL_OK]], %{{.+}} : i1
  // SCF: scf.condition(%[[MUL_RETRY]]) %[[MUL_OBSERVED]] : i64
  // SCF-NOT: reussir.cell.rmw
  // LLVM-LABEL: define i64 @direct_multiply
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: br label %[[LOOP:.+]]
  // LLVM: [[LOOP]]:
  // LLVM: mul i64
  // LLVM: cmpxchg weak ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} acq_rel monotonic, align 8
  // LLVM: br i1 %{{.+}}, label %[[LOOP]], label %[[DONE:.+]]
  // LLVM: [[DONE]]:
  func.func @direct_multiply(%factor: i64, %cell: !atomic_i64) -> i64 {
    %old = reussir.cell.rmw muli(%factor : i64, %cell : !atomic_i64) -> i64
    return %old : i64
  }

  // The region body is inlined into the scf.while before-region: the
  // optional output is computed on each attempt but only the successful
  // attempt's value leaves the loop.
  // ROUNDTRIP-LABEL: func.func @region_rmw
  // ROUNDTRIP: reussir.cell.rmw(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(seq_cst) -> i64 {
  // SCF-LABEL: func.func @region_rmw
  // SCF: %[[INIT:.+]] = reussir.cell.get(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic> atomic>) ordering(monotonic) : i64
  // SCF: scf.while (%[[EXPECTED:.+]] = %[[INIT]]) : (i64) -> (i64, i64)
  // SCF: %[[OBSERVED:.+]], %[[OK:.+]] = reussir.ref.cmpxchg(%[[EXPECTED]] : i64, %{{.+}} : i64, %{{.+}} : !reussir.ref<i64 field atomic>) weak ordering(seq_cst) : i64, i1
  // SCF: %[[RETRY:.+]] = arith.xori %[[OK]], %{{.+}} : i1
  // SCF: scf.condition(%[[RETRY]]) %[[OBSERVED]], %{{.+}} : i64, i64
  // SCF-NOT: reussir.cell.rmw
  // LLVM-LABEL: define i64 @region_rmw
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: add i64
  // LLVM: mul i64
  // LLVM: cmpxchg weak ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} seq_cst monotonic, align 8
  // LLVM: br i1 %{{.+}}, label %{{.+}}, label %[[REGION_DONE:.+]]
  // LLVM: [[REGION_DONE]]:
  // LLVM: ret i64
  func.func @region_rmw(%cell: !atomic_i64) -> i64 {
    %result = reussir.cell.rmw(%cell : !atomic_i64) ordering(seq_cst) -> i64 {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        %two = arith.constant 2 : i64
        %output = arith.muli %next, %two : i64
        reussir.cell.yield(%next : i64) output(%output : i64)
    }
    return %result : i64
  }

  // A body may omit the auxiliary output. Captured SSA values stay visible
  // inside the scf.while region without entering the atomic op's explicit
  // operand list.
  // LLVM-LABEL: define void @region_no_output
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: add i64 %{{.+}}, %{{.+}}
  // LLVM: cmpxchg weak ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} acq_rel monotonic, align 8
  // LLVM: ret void
  func.func @region_no_output(%delta: i64, %cell: !atomic_i64) {
    reussir.cell.rmw(%cell : !atomic_i64) {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64)
    }
    return
  }

  // The SCF-level loop carries the float element; only the ref.cmpxchg
  // lowering compares integer bit patterns, avoiding an invalid
  // floating-point llvm.cmpxchg.
  // LLVM-LABEL: define float @region_float
  // ROUNDTRIP: reussir.cell.rmw(%{{.+}} : !reussir.rc<!reussir.cell<f32 atomic> atomic>) ordering(release) -> f32 {
  // SCF-LABEL: func.func @region_float
  // SCF: scf.while
  // SCF: reussir.ref.cmpxchg(%{{.+}} : f32, %{{.+}} : f32, %{{.+}} : !reussir.ref<f32 field atomic>) weak ordering(release) : f32, i1
  // LLVM: load atomic float, ptr %{{.+}} monotonic, align 4
  // LLVM: fadd float
  // LLVM: bitcast float %{{.+}} to i32
  // LLVM: bitcast float %{{.+}} to i32
  // LLVM: cmpxchg weak ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} release monotonic, align 4
  // LLVM: bitcast i32 %{{.+}} to float
  func.func @region_float(%cell: !atomic_f32) -> f32 {
    %old = reussir.cell.rmw(%cell : !atomic_f32) ordering(release) -> f32 {
      ^bb0(%current: f32):
        %one = arith.constant 1.0 : f32
        %next = arith.addf %current, %one : f32
        reussir.cell.yield(%next : f32) output(%current : f32)
    }
    return %old : f32
  }

  // The bridge op itself round-trips (in both strong and weak spellings) and
  // stays a straight-line cmpxchg.
  // ROUNDTRIP-LABEL: func.func @raw_cmpxchg
  // ROUNDTRIP: reussir.ref.cmpxchg(%{{.+}} : i64, %{{.+}} : i64, %{{.+}} : !reussir.ref<i64 field>) ordering(monotonic) : i64, i1
  // LLVM-LABEL: define i1 @raw_cmpxchg
  // LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} monotonic monotonic, align 8
  func.func @raw_cmpxchg(%expected: i64, %desired: i64,
                         %slot: !reussir.ref<i64 field>) -> i1 {
    %observed, %ok = reussir.ref.cmpxchg(%expected : i64, %desired : i64,
        %slot : !reussir.ref<i64 field>) ordering(monotonic) : i64, i1
    return %ok : i1
  }
}
