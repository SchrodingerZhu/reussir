// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion)' | %FileCheck %s --check-prefix=DROP
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!flatlock_rc = !reussir.rc<!reussir.cell<!inner flatlock> atomic>

module {
  // Dropping the last reference to a flatlock cell releases the managed
  // payload through the same door every access uses: a combining-lock
  // critical section whose body loads the protected RC and decrements it —
  // never a leading-slot projection, which would address the lock header.
  // DROP-LABEL: func.func @flatlock_cell_drop
  // DROP: reussir.rc.fetch_sub
  // DROP-NOT: reussir.ref.project
  // DROP: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.combining_lock<!reussir.rc<i64 atomic>>>
  // DROP: sync.combining_lock.critical_section %[[VIEW]]
  // DROP: %[[SLOT:.+]] = reussir.ref.from_memref
  // DROP: %[[INNER:.+]] = reussir.ref.load(%[[SLOT]]
  // DROP: reussir.rc.dec(%[[INNER]] : !reussir.rc<i64 atomic>)
  // DROP-NOT: reussir.ref.project

  // Fully lowered, the inline path: box count drops, the status flag is
  // test-and-set (the fast-path acquire), the inner box's count drops and it
  // is freed under the lock (the combining region cannot thread the token
  // out), the status is released, and the 32-byte cell box dies last.
  // LLVM-LABEL: define void @flatlock_cell_drop(ptr
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1
  // LLVM: atomicrmw xchg ptr %{{.+}}, i8 1 acquire
  // LLVM: atomicrmw sub ptr %{{.+}}, i32 1
  // LLVM: call void @__reussir_deallocate(ptr %{{.+}}, i64 8, i64 16)
  // LLVM: store atomic i8 0, ptr %{{.+}} release
  // LLVM: call void @__reussir_deallocate(ptr %{{.+}}, i64 8, i64 32)
  func.func @flatlock_cell_drop(%cell: !flatlock_rc) {
    %t = reussir.rc.dec (%cell : !flatlock_rc) : !reussir.nullable<!reussir.token<align: 8, size: 32>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 32>>)
    return
  }
}
