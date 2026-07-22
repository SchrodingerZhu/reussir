// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// The frontend shape: the native side is a body-less `func.func` declaration
// (here with float types, which are nontrivial at the boundary) and a caller
// goes through it. The import materializes the declaration's definition; the
// caller's `func.call` links against it unchanged.
module attributes {llvm.target_triple = "x86_64-pc-linux-gnu"} {
  func.func private @hypot(f64, f64) -> f64

  reussir.trampoline import "C" @hypot_c = @hypot

  func.func @caller(%x: f64, %y: f64) -> f64 {
    %r = func.call @hypot(%x, %y) : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-DAG: llvm.func @hypot_c(!llvm.ptr, !llvm.ptr)
// CHECK-LABEL: llvm.func @hypot(
// CHECK-SAME: %[[X:.*]]: f64, %[[Y:.*]]: f64
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[RET:.*]] = llvm.alloca %[[ONE]] x f64
// CHECK: %[[ARGS:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(f64, f64)>
// CHECK: %[[F0:.*]] = llvm.getelementptr %[[ARGS]][0, 0]
// CHECK: llvm.store %[[X]], %[[F0]]
// CHECK: %[[F1:.*]] = llvm.getelementptr %[[ARGS]][0, 1]
// CHECK: llvm.store %[[Y]], %[[F1]]
// CHECK: llvm.call @hypot_c(%[[RET]], %[[ARGS]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK: %[[R:.*]] = llvm.load %[[RET]] : !llvm.ptr -> f64
// CHECK: llvm.return %[[R]] : f64

// CHECK-LABEL: llvm.func @caller(
// CHECK: llvm.call @hypot(
