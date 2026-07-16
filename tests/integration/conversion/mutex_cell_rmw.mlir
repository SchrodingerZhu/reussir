// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!mutex_i64 = !reussir.rc<!reussir.cell<i64 mutex> atomic>
!mutex_rc = !reussir.rc<!reussir.cell<!inner mutex> atomic>

module {
  // The region form on a mutex cell is a RefCell-style borrow under the lock:
  // the element moves into the body inside the critical section and the
  // yielded replacement is stored back before the unlock — no in-use flag is
  // read or written.
  // STD-LABEL: func.func @rmw_addi
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.mutex<i64>>
  // STD: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // STD: sync.raw_mutex.try_lock %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: %[[OLD:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: %[[NEW:.+]] = arith.addi %[[OLD]]
  // STD: reussir.ref.store(%[[SLOT]]{{.*}}) (%[[NEW]] : i64)
  // STD: sync.raw_mutex.unlock_fast %[[RAW]]
  // STD-NOT: reussir.cell.rmw
  // STD: return %[[OLD]]
  // LLVM-LABEL: define i64 @rmw_addi
  // LLVM: cmpxchg
  // LLVM: %[[OLD:.+]] = load i64
  // LLVM: %[[NEXT:.+]] = add i64 %[[OLD]]
  // LLVM: store i64 %[[NEXT]]
  // LLVM: atomicrmw xchg
  // LLVM: ret i64 %[[OLD]]
  func.func @rmw_addi(%delta: i64, %cell: !mutex_i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !mutex_i64) -> i64 {
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
  // STD: sync.raw_mutex.try_lock
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.acquire
  // STD-NOT: reussir.ref.drop
  // STD: %[[OLD:.+]] = reussir.ref.load
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.drop
  // STD: reussir.ref.store
  // STD: sync.raw_mutex.unlock_fast
  // STD: return %[[OLD]]
  // LLVM-LABEL: define ptr @rmw_swap
  // LLVM-NOT: atomicrmw add
  // LLVM-NOT: atomicrmw sub
  // LLVM-NOT: call void @__reussir_deallocate
  // LLVM: %[[OLD:.+]] = load ptr
  // LLVM: store ptr
  // LLVM: atomicrmw xchg
  // LLVM: ret ptr %[[OLD]]
  func.func @rmw_swap(%new: !inner, %cell: !mutex_rc) -> !inner {
    %old = reussir.cell.rmw(%cell : !mutex_rc) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%new : !inner) output(%current : !inner)
    }
    return %old : !inner
  }

  // The output is optional; a resultless read-modify-write erases cleanly.
  // STD-LABEL: func.func @rmw_no_output
  // STD: sync.raw_mutex.try_lock
  // STD: reussir.ref.store
  // STD: sync.raw_mutex.unlock_fast
  // STD-NOT: reussir.cell.rmw
  // STD: return
  func.func @rmw_no_output(%cell: !mutex_i64) {
    reussir.cell.rmw(%cell : !mutex_i64) {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64)
    }
    return
  }
}
