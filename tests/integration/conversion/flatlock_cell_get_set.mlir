// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!flatlock_i64 = !reussir.rc<!reussir.cell<i64 flatlock> atomic>
!flatlock_rc = !reussir.rc<!reussir.cell<!inner flatlock> atomic>

module {
  // The combining-lock critical section is outlined for the contended path:
  // a cold wrapper recovers the payload view and the captures from the raw
  // node pointer and replays the body — here the get's acquire-and-load into
  // the caller's out slot.
  // STD: func.func private @__sync_combining_lock_slow_{{[0-9]+}}
  // STD: "sync.combining_lock.recover"

  // A flatlock get routes the cloned value through a captured stack slot:
  // the combining-lock region has no results because its body may be served
  // by another thread. The fast path checks the runtime tail, test-and-sets
  // the status flag, runs the body inline, and releases.
  // STD-LABEL: func.func @get_i64
  // STD: %[[OUT:.+]] = memref.alloca() : memref<i64>
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.combining_lock<i64>>
  // STD: %[[PAYLOAD:.+]] = sync.combining_lock.get_payload %[[VIEW]]
  // STD: sync.combining_lock.has_tail %[[VIEW]]
  // STD: sync.combining_lock.try_acquire %[[VIEW]]
  // STD: "sync.combining_lock.capture"
  // STD: func.call @__sync_trampoline_mlir_sync_combining_lock_attach_slow_path{{.*}} {CConv = #llvm.cconv<preserve_mostcc>}
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: reussir.ref.acquire(%[[SLOT]]
  // STD: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: memref.store %[[VALUE]], %[[OUT]][] : memref<i64>
  // STD: sync.combining_lock.release %[[VIEW]]
  // STD: %[[RESULT:.+]] = memref.load %[[OUT]][] : memref<i64>
  // STD: return %[[RESULT]]
  // LLVM-LABEL: define i64 @get_i64
  // LLVM: load atomic ptr, ptr %{{.+}} monotonic
  // LLVM: atomicrmw xchg ptr %{{.+}}, i8 1 acquire
  // LLVM: call preserve_mostcc void @__sync_trampoline_mlir_sync_combining_lock_attach_slow_path
  // LLVM: ret i64
  func.func @get_i64(%cell: !flatlock_i64) -> i64 {
    %value = reussir.cell.get(%cell : !flatlock_i64) : i64
    return %value : i64
  }

  // Set needs no out slot: the body drops the protected old value and stores
  // the caller-owned replacement, which reaches the slow path as a capture.
  // STD-LABEL: func.func @set_i64
  // STD-NOT: memref.alloca
  // STD: sync.combining_lock.has_tail
  // STD: sync.combining_lock.try_acquire
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref
  // STD: reussir.ref.drop(%[[SLOT]]
  // STD: reussir.ref.store(%[[SLOT]]
  // STD: sync.combining_lock.release
  // STD: return
  // LLVM-LABEL: define void @set_i64
  // LLVM: atomicrmw xchg ptr %{{.+}}, i8 1 acquire
  // LLVM: call preserve_mostcc void @__sync_trampoline_mlir_sync_combining_lock_attach_slow_path
  // LLVM: ret void
  func.func @set_i64(%value: i64, %cell: !flatlock_i64) {
    reussir.cell.set(%value : i64, %cell : !flatlock_i64)
    return
  }

  // A managed element is retained inside the critical section before its
  // pointer leaves through the out slot, so a concurrent set can never free
  // the box between the load and the inc.
  // STD-LABEL: func.func @get_rc
  // STD: memref.alloca() : memref<!reussir.rc<i64 atomic>>
  // STD: %[[VALUE:.+]] = reussir.ref.load
  // STD-NEXT: reussir.rc.inc(%[[VALUE]]
  // STD: memref.store %[[VALUE]]
  // STD: sync.combining_lock.release
  // LLVM-LABEL: define ptr @get_rc
  // LLVM: ret ptr
  func.func @get_rc(%cell: !flatlock_rc) -> !inner {
    %value = reussir.cell.get(%cell : !flatlock_rc) : !inner
    return %value : !inner
  }

  // The replacement moves in without a retain; the old value is released
  // exactly once inside the section.
  // STD-LABEL: func.func @set_rc
  // STD-NOT: reussir.rc.inc
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref
  // STD: reussir.ref.drop(%[[SLOT]]
  // STD: reussir.ref.store(%[[SLOT]]
  // STD-NOT: reussir.rc.inc
  // LLVM-LABEL: define void @set_rc
  // LLVM: ret void
  func.func @set_rc(%value: !inner, %cell: !flatlock_rc) {
    reussir.cell.set(%value : !inner, %cell : !flatlock_rc)
    return
  }
}
