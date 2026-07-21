// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// Four scalar arguments exceed the trivial threshold: the import packs them
// into a struct passed by pointer, while the scalar result stays direct.
module attributes {llvm.target_triple = "x86_64-pc-linux-gnu"} {
  llvm.func @pick_last(i64, i64, i64, i64) -> i64

  reussir.trampoline import "C" @pick_last_c = @pick_last
}

// CHECK-DAG: llvm.func @pick_last_c(!llvm.ptr) -> i64
// CHECK-LABEL: llvm.func @pick_last(
// CHECK-SAME: %[[A0:.*]]: i64, %[[A1:.*]]: i64, %[[A2:.*]]: i64, %[[A3:.*]]: i64
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[ARGS:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(i64, i64, i64, i64)>
// CHECK: %[[F0:.*]] = llvm.getelementptr %[[ARGS]][0, 0]
// CHECK: llvm.store %[[A0]], %[[F0]]
// CHECK: %[[F1:.*]] = llvm.getelementptr %[[ARGS]][0, 1]
// CHECK: llvm.store %[[A1]], %[[F1]]
// CHECK: %[[F2:.*]] = llvm.getelementptr %[[ARGS]][0, 2]
// CHECK: llvm.store %[[A2]], %[[F2]]
// CHECK: %[[F3:.*]] = llvm.getelementptr %[[ARGS]][0, 3]
// CHECK: llvm.store %[[A3]], %[[F3]]
// CHECK: %[[R:.*]] = llvm.call @pick_last_c(%[[ARGS]]) : (!llvm.ptr) -> i64
// CHECK: llvm.return %[[R]] : i64
