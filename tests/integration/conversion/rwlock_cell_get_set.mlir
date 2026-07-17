// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --reussir-convert-to-std --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=MANAGE
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // A get takes the lock shared: the reader-count retry loop and the read
  // slow path, never the write ones.
  // STD-LABEL: func.func @get_i64
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<i64>>
  // STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %[[VIEW]]
  // STD: scf.while
  // STD: sync.raw_rwlock.load_state %[[RAW]]
  // STD: sync.raw_rwlock.cmpxchg_state %[[RAW]]
  // STD: func.call @__sync_trampoline_mlir_sync_rwlock_read_lock_slow_path
  // STD-NOT: write_lock
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: reussir.ref.acquire(%[[SLOT]]
  // STD-NEXT: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: sync.raw_rwlock.read_unlock_fast %[[RAW]]
  // STD: func.call @__sync_trampoline_mlir_sync_rwlock_unlock_slow_path
  // STD: return %[[VALUE]]
  // LLVM-LABEL: define i64 @get_i64
  // LLVM: cmpxchg
  // LLVM: load i64
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1 release
  // LLVM: ret i64
  func.func @get_i64(%cell: !rwlock_i64) -> i64 {
    %value = reussir.cell.get(%cell : !rwlock_i64) : i64
    return %value : i64
  }

  // A set mutates the protected slot, so it takes the lock exclusively: the
  // write acquisition compare-exchanges the whole writer mask in.
  // STD-LABEL: func.func @set_i64
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<i64>>
  // STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %[[VIEW]]
  // STD: scf.while
  // STD: sync.raw_rwlock.load_state %[[RAW]]
  // STD: sync.raw_rwlock.cmpxchg_state %[[RAW]]
  // STD: func.call @__sync_trampoline_mlir_sync_rwlock_write_lock_slow_path
  // STD-NOT: read_lock
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: reussir.ref.drop(%[[SLOT]]
  // STD: reussir.ref.store(%[[SLOT]]
  // STD: sync.raw_rwlock.write_unlock_fast %[[RAW]]
  // STD: func.call @__sync_trampoline_mlir_sync_rwlock_unlock_slow_path
  // STD: return
  // LLVM-LABEL: define void @set_i64
  // LLVM: cmpxchg ptr %{{.+}}, i32 0, i32 1073741823
  // LLVM: store i64
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1073741823 release
  // LLVM: ret void
  func.func @set_i64(%value: i64, %cell: !rwlock_i64) {
    reussir.cell.set(%value : i64, %cell : !rwlock_i64)
    return
  }

  // Cloning an RC element under the read lock only bumps the element's own
  // atomic count; the protected slot is never written, so concurrent readers
  // are safe.
  // STD-LABEL: func.func @get_rc
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<!reussir.rc<i64 atomic>>>
  // STD: func.call @__sync_trampoline_mlir_sync_rwlock_read_lock_slow_path
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<!reussir.rc<i64 atomic>>) : !reussir.ref<!reussir.rc<i64 atomic> field atomic>
  // STD: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD-NEXT: reussir.rc.inc(%[[VALUE]]
  // STD: sync.raw_rwlock.read_unlock_fast
  // STD: return %[[VALUE]]
  // LLVM-LABEL: define ptr @get_rc
  // LLVM: %[[VALUE:.+]] = load ptr
  // LLVM: atomicrmw add ptr %{{.+}}, i32 1
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1 release
  // LLVM: ret ptr %[[VALUE]]
  func.func @get_rc(%cell: !rwlock_rc) -> !inner {
    %value = reussir.cell.get(%cell : !rwlock_rc) : !inner
    return %value : !inner
  }

  // The old element is dropped and the caller-owned replacement moved in,
  // all while holding the lock exclusively.
  // MANAGE-LABEL: func.func @set_rc(
  // MANAGE-SAME: %[[NEW:[a-zA-Z0-9_]+]]: !reussir.rc<i64 atomic>
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<!reussir.rc<i64 atomic>>>
  // MANAGE: func.call @__sync_trampoline_mlir_sync_rwlock_write_lock_slow_path
  // MANAGE: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // MANAGE: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<!reussir.rc<i64 atomic>>) : !reussir.ref<!reussir.rc<i64 atomic> field atomic>
  // MANAGE: %[[OLD:.+]] = reussir.ref.load(%[[SLOT]]
  // MANAGE-NEXT: %{{.+}} = reussir.rc.dec(%[[OLD]] : !reussir.rc<i64 atomic>)
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: reussir.ref.store(%[[SLOT]]{{.*}}) (%[[NEW]] : !reussir.rc<i64 atomic>)
  // MANAGE: sync.raw_rwlock.write_unlock_fast
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: return
  // LLVM-LABEL: define void @set_rc(
  // LLVM-SAME: ptr %[[NEW:.+]], ptr
  // LLVM: %[[OLD:.+]] = load ptr, ptr %[[SLOT:.+]], align 8
  // LLVM: atomicrmw sub ptr %[[OLD]], i32 1
  // LLVM: call void @__reussir_deallocate(ptr %[[OLD]], i64 8, i64 16)
  // LLVM: store ptr %[[NEW]], ptr %[[SLOT]]
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1073741823 release
  // LLVM: ret void
  func.func @set_rc(%value: !inner, %cell: !rwlock_rc) {
    reussir.cell.set(%value : !inner, %cell : !rwlock_rc)
    return
  }
}
