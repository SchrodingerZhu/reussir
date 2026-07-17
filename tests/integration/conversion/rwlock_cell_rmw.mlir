// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // The region form on an rwlock cell is a RefCell-style borrow under the
  // WRITE lock: the replacement store mutates the protected slot, so the
  // section excludes readers and writers alike. The element moves into the
  // body and the yielded replacement is stored back before the unlock.
  // STD-LABEL: func.func @rmw_addi
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<i64>>
  // STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %[[VIEW]]
  // STD: sync.raw_rwlock.load_state %[[RAW]]
  // STD: sync.raw_rwlock.cmpxchg_state %[[RAW]]
  // STD: func.call @mlir_sync_rwlock_write_lock_slow_path
  // STD-NOT: read_lock
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: %[[OLD:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: %[[NEW:.+]] = arith.addi %[[OLD]]
  // STD: reussir.ref.store(%[[SLOT]]{{.*}}) (%[[NEW]] : i64)
  // STD: sync.raw_rwlock.write_unlock_fast %[[RAW]]
  // STD-NOT: reussir.cell.rmw
  // STD: return %[[OLD]]
  // LLVM-LABEL: define i64 @rmw_addi
  // LLVM: cmpxchg ptr %{{.+}}, i32 0, i32 1073741823
  // LLVM: %[[OLD:.+]] = load i64
  // LLVM: %[[NEXT:.+]] = add i64 %[[OLD]]
  // LLVM: store i64 %[[NEXT]]
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1073741823 release
  // LLVM: ret i64 %[[OLD]]
  func.func @rmw_addi(%delta: i64, %cell: !rwlock_i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !rwlock_i64) -> i64 {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  // A managed element moves through the body without any implicit retain or
  // release: the old value's ownership transfers out as the output, the
  // caller-owned replacement transfers in, and the only atomics left are the
  // lock words themselves.
  // STD-LABEL: func.func @rmw_swap
  // STD: func.call @mlir_sync_rwlock_write_lock_slow_path
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.acquire
  // STD-NOT: reussir.ref.drop
  // STD: %[[OLD:.+]] = reussir.ref.load
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.drop
  // STD: reussir.ref.store
  // STD: sync.raw_rwlock.write_unlock_fast
  // STD: return %[[OLD]]
  // LLVM-LABEL: define ptr @rmw_swap
  // LLVM-NOT: atomicrmw add
  // LLVM-NOT: call void @__reussir_deallocate
  // LLVM: %[[OLD:.+]] = load ptr
  // LLVM: store ptr
  // LLVM: ret ptr %[[OLD]]
  func.func @rmw_swap(%new: !inner, %cell: !rwlock_rc) -> !inner {
    %old = reussir.cell.rmw(%cell : !rwlock_rc) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%new : !inner) output(%current : !inner)
    }
    return %old : !inner
  }

  // The output is optional; a resultless read-modify-write erases cleanly.
  // STD-LABEL: func.func @rmw_no_output
  // STD: func.call @mlir_sync_rwlock_write_lock_slow_path
  // STD: reussir.ref.store
  // STD: sync.raw_rwlock.write_unlock_fast
  // STD-NOT: reussir.cell.rmw
  // STD: return
  func.func @rmw_no_output(%cell: !rwlock_i64) {
    reussir.cell.rmw(%cell : !rwlock_i64) {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64)
    }
    return
  }
}
