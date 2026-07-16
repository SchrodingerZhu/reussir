// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation))' | %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64>
!mutex_i64 = !reussir.rc<!reussir.cell<i64 mutex> atomic>
!mutex_f64 = !reussir.rc<!reussir.cell<f64 mutex> atomic>
!mutex_rc = !reussir.rc<!reussir.cell<!inner mutex> atomic>

module {
  // The mutex storage is `{i32 state, i64 payload}`. Together with the atomic
  // RC count, its box requires 24 bytes at 8-byte alignment.
  // TOKEN-LABEL: func.func @create_i64
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 24>
  // TOKEN: reussir.cell.create value(%{{.+}} : i64) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>)
  // STD-LABEL: func.func @create_i64
  // STD: %[[CELL:.+]] = reussir.rc.create
  // STD: %[[REF:.+]] = reussir.rc.borrow(%[[CELL]]
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref(%[[REF]] : !reussir.ref<!reussir.cell<i64 mutex> atomic>) : memref<!sync.mutex<i64>>
  // STD: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // STD: sync.raw_mutex.init %[[RAW]]
  // STD: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // STD: memref.store %{{.+}}, %[[PAYLOAD]][] : memref<i64>
  // STD-NOT: reussir.cell.create
  // LLVM-LABEL: define ptr @create_i64
  // LLVM: store i32 0, ptr %{{.+}}
  // LLVM: store i64 %{{.+}}, ptr %{{.+}}
  func.func @create_i64(%value: i64) -> !mutex_i64 {
    %cell = reussir.cell.create value(%value : i64) : !mutex_i64
    return %cell : !mutex_i64
  }

  // Exercise a second ordinary primitive payload and its typed sync view.
  // STD-LABEL: func.func @create_f64
  // STD: reussir.ref.to_memref({{.*}}) : memref<!sync.mutex<f64>>
  // STD: memref.store %{{.+}}, %{{.+}}[] : memref<f64>
  // LLVM-LABEL: define ptr @create_f64
  // LLVM: store i32 0, ptr %{{.+}}
  // LLVM: store double %{{.+}}, ptr %{{.+}}
  func.func @create_f64(%value: f64) -> !mutex_f64 {
    %cell = reussir.cell.create value(%value : f64) : !mutex_f64
    return %cell : !mutex_f64
  }

  // An RC-managed payload is moved into the protected slot without retaining
  // it: ownership transfers from the argument to the newly created cell.
  // TOKEN-LABEL: func.func @create_rc
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 24>
  // TOKEN: reussir.cell.create value(%{{.+}} : !reussir.rc<i64>) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>)
  // STD-LABEL: func.func @create_rc
  // STD: reussir.ref.to_memref({{.*}}) : memref<!sync.mutex<!reussir.rc<i64>>>
  // STD: memref.store %{{.+}}, %{{.+}}[] : memref<!reussir.rc<i64>>
  // STD-NOT: reussir.rc.inc
  // LLVM-LABEL: define ptr @create_rc
  // LLVM: store i32 0, ptr %{{.+}}
  // LLVM: store ptr %{{.+}}, ptr %{{.+}}
  func.func @create_rc(%value: !inner) -> !mutex_rc {
    %cell = reussir.cell.create value(%value : !inner) : !mutex_rc
    return %cell : !mutex_rc
  }
}
