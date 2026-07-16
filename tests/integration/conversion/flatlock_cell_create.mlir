// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation))' | %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!flatlock_i64 = !reussir.rc<!reussir.cell<i64 flatlock> atomic>
!flatlock_rc = !reussir.rc<!reussir.cell<!inner flatlock> atomic>

module {
  // The flatlock storage is `{{ptr tail, i8 status}, i64 payload}`. Together
  // with the atomic RC count, its box requires 32 bytes at 8-byte alignment.
  // TOKEN-LABEL: func.func @create_i64
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 32>
  // TOKEN: reussir.cell.create value(%{{.+}} : i64) token(%[[TOKEN]] : !reussir.token<align : 8, size : 32>)

  // `sync.combining_lock.init` is a straight-line bridge operation: it
  // survives convert-to-std over the storage view and lowers directly in the
  // LLVM conversion.
  // STD-LABEL: func.func @create_i64
  // STD: %[[CELL:.+]] = reussir.rc.create
  // STD: %[[REF:.+]] = reussir.rc.borrow(%[[CELL]]
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref(%[[REF]] : !reussir.ref<!reussir.cell<i64 flatlock> atomic>) : memref<!sync.combining_lock<i64>>
  // STD: sync.combining_lock.init %[[VIEW]] : memref<!sync.combining_lock<i64>>, %{{.+}} : i64
  // STD-NOT: reussir.cell.create

  // The init clears the runtime tail pointer and status flag, then stores the
  // payload past the lock header.
  // LLVM-LABEL: define ptr @create_i64
  // LLVM: %[[HDR0:.+]] = getelementptr { { ptr, i8 }, i64 }, ptr %{{.+}}, i32 0, i32 0, i32 0
  // LLVM: %[[HDR1:.+]] = getelementptr { { ptr, i8 }, i64 }, ptr %{{.+}}, i32 0, i32 0, i32 1
  // LLVM: store ptr null, ptr %[[HDR0]]
  // LLVM: store i8 0, ptr %[[HDR1]]
  // LLVM: %[[SLOT:.+]] = getelementptr { { ptr, i8 }, i64 }, ptr %{{.+}}, i32 0, i32 1
  // LLVM: store i64 %{{.+}}, ptr %[[SLOT]]
  func.func @create_i64(%value: i64) -> !flatlock_i64 {
    %cell = reussir.cell.create value(%value : i64) : !flatlock_i64
    return %cell : !flatlock_i64
  }

  // An RC-managed payload is moved into the protected slot without retaining
  // it: ownership transfers from the argument to the newly created cell.
  // TOKEN-LABEL: func.func @create_rc
  // TOKEN: reussir.token.alloc : <align : 8, size : 32>
  // STD-LABEL: func.func @create_rc
  // STD: reussir.ref.to_memref({{.*}}) : memref<!sync.combining_lock<!reussir.rc<i64 atomic>>>
  // STD: sync.combining_lock.init %{{.+}} : memref<!sync.combining_lock<!reussir.rc<i64 atomic>>>, %{{.+}} : !reussir.rc<i64 atomic>
  // STD-NOT: reussir.rc.inc
  // LLVM-LABEL: define ptr @create_rc
  // LLVM: store ptr null, ptr %{{.+}}
  // LLVM: store i8 0, ptr %{{.+}}
  // LLVM-NOT: atomicrmw add
  // LLVM: store ptr %{{.+}}, ptr %{{.+}}
  func.func @create_rc(%value: !inner) -> !flatlock_rc {
    %cell = reussir.cell.create value(%value : !inner) : !flatlock_rc
    return %cell : !flatlock_rc
  }
}
