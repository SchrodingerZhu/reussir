// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // The whole transaction body runs inside the read critical section — after
  // the shared acquisition, before the read unlock — and only the yielded
  // result leaves the section. The element is cloned in like a get; nothing
  // is stored back.
  // STD-LABEL: func.func @read_i64
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<i64>>
  // STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %[[VIEW]]
  // STD: func.call @mlir_sync_rwlock_read_lock_slow_path
  // STD-NOT: write_lock
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: reussir.ref.acquire(%[[SLOT]]
  // STD-NEXT: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: %[[RESULT:.+]] = arith.muli %[[VALUE]]
  // STD: sync.raw_rwlock.read_unlock_fast %[[RAW]]
  // STD-NOT: reussir.ref.store
  // STD-NOT: reussir.cell.rdlock
  // STD: return %[[RESULT]]
  // LLVM-LABEL: define i64 @read_i64
  // LLVM: cmpxchg
  // LLVM: %[[VALUE:.+]] = load i64
  // LLVM: %[[RESULT:.+]] = mul i64 %[[VALUE]], 2
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1 release
  // LLVM-NOT: store i64
  // LLVM: ret i64 %[[RESULT]]
  func.func @read_i64(%cell: !rwlock_i64) -> i64 {
    %doubled = reussir.cell.rdlock(%cell : !rwlock_i64) -> i64 {
      ^bb0(%current: i64):
        %two = arith.constant 2 : i64
        %result = arith.muli %current, %two : i64
        reussir.scf.yield %result : i64
    }
    return %doubled : i64
  }

  // An RC element is acquired into the body — the inc hits the element's own
  // atomic count, never the protected slot, so concurrent read transactions
  // are safe — and the body owns and consumes the clone. The result is
  // optional; a resultless transaction erases cleanly.
  // STD-LABEL: func.func @read_rc_no_output
  // STD: func.call @mlir_sync_rwlock_read_lock_slow_path
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<!reussir.rc<i64 atomic>>) : !reussir.ref<!reussir.rc<i64 atomic> field atomic>
  // STD: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD-NEXT: reussir.rc.inc(%[[VALUE]]
  // STD: call @consume(%[[VALUE]])
  // STD: sync.raw_rwlock.read_unlock_fast
  // STD-NOT: reussir.cell.rdlock
  // STD: return
  // LLVM-LABEL: define void @read_rc_no_output
  // LLVM: %[[VALUE:.+]] = load ptr
  // LLVM: atomicrmw add ptr %{{.+}}, i32 1
  // LLVM: call void @consume(ptr %[[VALUE]])
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1 release
  // LLVM: ret void
  func.func private @consume(!inner) -> ()
  func.func @read_rc_no_output(%cell: !rwlock_rc) {
    reussir.cell.rdlock(%cell : !rwlock_rc) {
      ^bb0(%current: !inner):
        func.call @consume(%current) : (!inner) -> ()
        reussir.scf.yield
    }
    return
  }
}
