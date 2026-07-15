// RUN: %reussir-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=EXPAND
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,reussir-acquire-drop-expansion,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!atomic_i64 = !reussir.rc<!reussir.cell<i64 atomic>>
!atomic_f32 = !reussir.rc<!reussir.cell<f32 atomic>>

module {
  // ROUNDTRIP-LABEL: func.func @atomic_get
  // ROUNDTRIP-SAME: !reussir.rc<!reussir.cell<i64 atomic>>
  // EXPAND-LABEL: func.func @atomic_get
  // EXPAND: reussir.cell.get
  // LLVM-LABEL: define i64 @atomic_get
  // LLVM: load atomic i64, ptr %{{.+}} acquire, align 8
  func.func @atomic_get(%cell: !atomic_i64) -> i64 {
    %value = reussir.cell.get(%cell : !atomic_i64) : i64
    return %value : i64
  }

  // ROUNDTRIP-LABEL: func.func @atomic_get_monotonic
  // ROUNDTRIP: reussir.cell.get(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic>>) ordering(monotonic) : i64
  // LLVM-LABEL: define i64 @atomic_get_monotonic
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  func.func @atomic_get_monotonic(%cell: !atomic_i64) -> i64 {
    %value = reussir.cell.get(%cell : !atomic_i64) ordering(monotonic) : i64
    return %value : i64
  }

  // EXPAND-LABEL: func.func @atomic_set
  // EXPAND: reussir.cell.set
  // LLVM-LABEL: define void @atomic_set
  // LLVM: store atomic i64 %{{.+}}, ptr %{{.+}} release, align 8
  func.func @atomic_set(%value: i64, %cell: !atomic_i64) {
    reussir.cell.set(%value : i64, %cell : !atomic_i64)
    return
  }

  // ROUNDTRIP-LABEL: func.func @atomic_set_seq_cst
  // ROUNDTRIP: reussir.cell.set(%{{.+}} : i64, %{{.+}} : !reussir.rc<!reussir.cell<i64 atomic>>) ordering(seq_cst)
  // LLVM-LABEL: define void @atomic_set_seq_cst
  // LLVM: store atomic i64 %{{.+}}, ptr %{{.+}} seq_cst, align 8
  func.func @atomic_set_seq_cst(%value: i64, %cell: !atomic_i64) {
    reussir.cell.set(%value : i64, %cell : !atomic_i64) ordering(seq_cst)
    return
  }

  // ROUNDTRIP-LABEL: func.func @direct_add
  // ROUNDTRIP: reussir.cell.rmw addi(%{{.+}} : i64, %{{.+}} : !reussir.rc<!reussir.cell<i64 atomic>>) ordering(acquire) -> i64
  // EXPAND-LABEL: func.func @direct_add
  // EXPAND: reussir.cell.rmw addi
  // LLVM-LABEL: define i64 @direct_add
  // LLVM: atomicrmw add ptr %{{.+}}, i64 %{{.+}} acquire, align 8
  func.func @direct_add(%delta: i64, %cell: !atomic_i64) -> i64 {
    %old = reussir.cell.rmw addi(%delta : i64, %cell : !atomic_i64) ordering(acquire) -> i64
    return %old : i64
  }

  // LLVM-LABEL: define i64 @direct_multiply
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: br label %[[LOOP:.+]]
  // LLVM: [[LOOP]]:
  // LLVM: mul i64
  // LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} acq_rel monotonic, align 8
  // LLVM: br i1 %{{.+}}, label %[[DONE:.+]], label %[[LOOP]]
  // LLVM: [[DONE]]:
  func.func @direct_multiply(%factor: i64, %cell: !atomic_i64) -> i64 {
    %old = reussir.cell.rmw muli(%factor : i64, %cell : !atomic_i64) -> i64
    return %old : i64
  }

  // The optional output is computed on each attempt but only the successful
  // attempt reaches the continuation.
  // ROUNDTRIP-LABEL: func.func @region_rmw
  // ROUNDTRIP: reussir.cell.rmw(%{{.+}} : !reussir.rc<!reussir.cell<i64 atomic>>) ordering(seq_cst) -> i64 {
  // EXPAND-LABEL: func.func @region_rmw
  // EXPAND: reussir.cell.rmw
  // EXPAND: reussir.cell.yield
  // LLVM-LABEL: define i64 @region_rmw
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: add i64
  // LLVM: mul i64
  // LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} seq_cst monotonic, align 8
  // LLVM: br i1 %{{.+}}, label %[[REGION_DONE:.+]], label %{{.+}}
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

  // A body may omit the auxiliary output. Captured SSA values are cloned
  // into the retry computation while still remaining outside the atomic op's
  // explicit operand list.
  // LLVM-LABEL: define void @region_no_output
  // LLVM: load atomic i64, ptr %{{.+}} monotonic, align 8
  // LLVM: add i64 %{{.+}}, %{{.+}}
  // LLVM: cmpxchg ptr %{{.+}}, i64 %{{.+}}, i64 %{{.+}} acq_rel monotonic, align 8
  // LLVM: ret void
  func.func @region_no_output(%delta: i64, %cell: !atomic_i64) {
    reussir.cell.rmw(%cell : !atomic_i64) {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64)
    }
    return
  }

  // Float cmpxchg compares integer bit patterns, avoiding an invalid
  // floating-point llvm.cmpxchg.
  // LLVM-LABEL: define float @region_float
  // ROUNDTRIP: reussir.cell.rmw(%{{.+}} : !reussir.rc<!reussir.cell<f32 atomic>>) ordering(release) -> f32 {
  // LLVM: load atomic float, ptr %{{.+}} monotonic, align 4
  // LLVM: bitcast float %{{.+}} to i32
  // LLVM: bitcast i32 %{{.+}} to float
  // LLVM: fadd float
  // LLVM: bitcast float %{{.+}} to i32
  // LLVM: cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} release monotonic, align 4
  func.func @region_float(%cell: !atomic_f32) -> f32 {
    %old = reussir.cell.rmw(%cell : !atomic_f32) ordering(release) -> f32 {
      ^bb0(%current: f32):
        %one = arith.constant 1.0 : f32
        %next = arith.addf %current, %one : f32
        reussir.cell.yield(%next : f32) output(%current : f32)
    }
    return %old : f32
  }
}
