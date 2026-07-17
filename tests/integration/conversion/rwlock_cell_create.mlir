// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation))' | %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // The rwlock storage is `{{i32, i32} header, i64 payload}` — the raw
  // header carries the two i32 words of the runtime's rwlock state machine.
  // Together with the atomic RC count, its box requires 24 bytes at 8-byte
  // alignment.
  // TOKEN-LABEL: func.func @create_i64
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 24>
  // TOKEN: reussir.cell.create value(%{{.+}} : i64) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>)
  // STD-LABEL: func.func @create_i64
  // STD: %[[CELL:.+]] = reussir.rc.create
  // STD: %[[REF:.+]] = reussir.rc.borrow(%[[CELL]]
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref(%[[REF]] : !reussir.ref<!reussir.cell<i64 rwlock> atomic>) : memref<!sync.rwlock<i64>>
  // STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %[[VIEW]]
  // STD: sync.raw_rwlock.init %[[RAW]]
  // STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %[[VIEW]]
  // STD: memref.store %{{.+}}, %[[PAYLOAD]][] : memref<i64>
  // STD-NOT: reussir.cell.create
  // LLVM-LABEL: define ptr @create_i64
  // LLVM: store i32 0, ptr %{{.+}}
  // LLVM: store i64 %{{.+}}, ptr %{{.+}}
  func.func @create_i64(%value: i64) -> !rwlock_i64 {
    %cell = reussir.cell.create value(%value : i64) : !rwlock_i64
    return %cell : !rwlock_i64
  }

  // An RC-managed payload is moved into the protected slot without retaining
  // it: ownership transfers from the argument to the newly created cell.
  // TOKEN-LABEL: func.func @create_rc
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 24>
  // TOKEN: reussir.cell.create value(%{{.+}} : !reussir.rc<i64 atomic>) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>)
  // STD-LABEL: func.func @create_rc
  // STD: reussir.ref.to_memref({{.*}}) : memref<!sync.rwlock<!reussir.rc<i64 atomic>>>
  // STD: memref.store %{{.+}}, %{{.+}}[] : memref<!reussir.rc<i64 atomic>>
  // STD-NOT: reussir.rc.inc
  // LLVM-LABEL: define ptr @create_rc
  // LLVM: store i32 0, ptr %{{.+}}
  // LLVM: store ptr %{{.+}}, ptr %{{.+}}
  func.func @create_rc(%value: !inner) -> !rwlock_rc {
    %cell = reussir.cell.create value(%value : !inner) : !rwlock_rc
    return %cell : !rwlock_rc
  }
}
