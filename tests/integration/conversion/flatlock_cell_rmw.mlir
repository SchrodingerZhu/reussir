// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!flatlock_i64 = !reussir.rc<!reussir.cell<i64 flatlock> atomic>
!flatlock_rc = !reussir.rc<!reussir.cell<!inner flatlock> atomic>

module {
  // The region form on a flatlock cell is the RefCell-style borrow inside a
  // combining-lock critical section: the element moves into the body, the
  // yielded replacement is stored back, and the optional output leaves
  // through a captured stack slot (the region has no results because the
  // body may run on the combining thread).
  // STD-LABEL: func.func @rmw_addi
  // STD: %[[OUT:.+]] = memref.alloca() : memref<i64>
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.combining_lock<i64>>
  // STD: sync.combining_lock.has_tail %[[VIEW]]
  // STD: sync.combining_lock.try_acquire %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref
  // STD: %[[OLD:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: %[[NEW:.+]] = arith.addi %[[OLD]]
  // STD: reussir.ref.store(%[[SLOT]]{{.*}}) (%[[NEW]] : i64)
  // STD: memref.store %[[OLD]], %[[OUT]][] : memref<i64>
  // STD: sync.combining_lock.release %[[VIEW]]
  // STD-NOT: reussir.cell.rmw
  // STD: %[[RESULT:.+]] = memref.load %[[OUT]][] : memref<i64>
  // STD: return %[[RESULT]]
  // LLVM-LABEL: define i64 @rmw_addi
  // LLVM: atomicrmw xchg ptr %{{.+}}, i8 1 acquire
  // LLVM: call preserve_mostcc void @mlir_sync_combining_lock_attach_slow_path
  // LLVM: ret i64
  func.func @rmw_addi(%delta: i64, %cell: !flatlock_i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !flatlock_i64) -> i64 {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  // A managed element moves through the body without any implicit retain or
  // release: ownership of the old value transfers out through the out slot,
  // the caller-owned replacement transfers in.
  // STD-LABEL: func.func @rmw_swap
  // STD: memref.alloca() : memref<!reussir.rc<i64 atomic>>
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.acquire
  // STD-NOT: reussir.ref.drop
  // STD: %[[OLD:.+]] = reussir.ref.load
  // STD-NOT: reussir.rc.inc
  // STD-NOT: reussir.ref.drop
  // STD: reussir.ref.store
  // STD: memref.store %[[OLD]]
  // STD: sync.combining_lock.release
  // LLVM-LABEL: define ptr @rmw_swap
  // LLVM-NOT: atomicrmw add
  // LLVM-NOT: atomicrmw sub
  // LLVM-NOT: call void @__reussir_deallocate
  // LLVM: ret ptr
  func.func @rmw_swap(%new: !inner, %cell: !flatlock_rc) -> !inner {
    %old = reussir.cell.rmw(%cell : !flatlock_rc) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%new : !inner) output(%current : !inner)
    }
    return %old : !inner
  }

  // Without an output there is no out slot at all.
  // STD-LABEL: func.func @rmw_no_output
  // STD-NOT: memref.alloca
  // STD: sync.combining_lock.try_acquire
  // STD: reussir.ref.store
  // STD: sync.combining_lock.release
  // STD-NOT: reussir.cell.rmw
  // STD: return
  func.func @rmw_no_output(%cell: !flatlock_i64) {
    reussir.cell.rmw(%cell : !flatlock_i64) {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64)
    }
    return
  }
}
