// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion)' | %FileCheck %s --check-prefix=DROP
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // Dropping the last reference to an rwlock cell releases the managed
  // payload through the same door every mutation uses: a write critical
  // section over the rwlock view, whose body loads the protected RC and
  // decrements it — never a leading-slot projection, which would address the
  // lock header.
  // DROP-LABEL: func.func @rwlock_cell_drop
  // DROP: reussir.rc.fetch_sub
  // DROP-NOT: reussir.ref.project
  // DROP: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.rwlock<!reussir.rc<i64 atomic>>>
  // DROP: sync.rwlock.write_critical_section %[[VIEW]]
  // DROP: ^bb0(%[[PAYLOAD:.+]]: memref<!reussir.rc<i64 atomic>>):
  // DROP: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]]
  // DROP: %[[INNER:.+]] = reussir.ref.load(%[[SLOT]]
  // DROP: reussir.rc.dec(%[[INNER]] : !reussir.rc<i64 atomic>)
  // DROP-NOT: reussir.ref.project

  // Fully lowered: the box count drops first; on the last reference the
  // state word of the two-word raw rwlock header is write-acquired (the
  // whole writer mask compare-exchanged in), the payload pointer is loaded
  // from rwlock field 1 — past the lock header — and its count is
  // decremented under the lock. Both frees happen after the unlock, and the
  // cell box dies through the rwlock-aware 24-byte layout.
  // LLVM-LABEL: define void @rwlock_cell_drop(ptr
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1
  // LLVM: %[[CELL:.+]] = getelementptr { i32, { { i32, i32 }, ptr } }, ptr %{{.+}}, i32 0, i32 1
  // LLVM: %[[HEADER:.+]] = getelementptr { { i32, i32 }, ptr }, ptr %[[CELL]], i32 0, i32 0
  // LLVM: %[[STATE:.+]] = getelementptr { i32, i32 }, ptr %[[HEADER]], i32 0, i32 0
  // LLVM: cmpxchg ptr %[[STATE]], i32 0, i32 1073741823
  // LLVM: %[[SLOT:.+]] = getelementptr { { i32, i32 }, ptr }, ptr %[[CELL]], i32 0, i32 1
  // LLVM: %[[INNER:.+]] = load ptr, ptr %[[SLOT]]
  // LLVM: atomicrmw sub ptr %[[INNER]], i32 1
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1073741823 release
  // LLVM: call void @__reussir_deallocate(ptr %{{.+}}, i64 8, i64 24)
  // LLVM: call void @__reussir_deallocate(ptr %{{.+}}, i64 8, i64 16)
  func.func @rwlock_cell_drop(%cell: !rwlock_rc) {
    %t = reussir.rc.dec (%cell : !rwlock_rc) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }
}
