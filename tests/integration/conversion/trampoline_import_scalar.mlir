// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// Trivial import: all-scalar signature, so the materialized native function
// forwards straight to the external boundary symbol.
module attributes {llvm.target_triple = "x86_64-pc-linux-gnu"} {
  llvm.func @id_i64(i64) -> i64

  reussir.trampoline import "C" @id_i64_c = @id_i64
}

// CHECK-DAG: llvm.func @id_i64_c(i64) -> i64
// CHECK-LABEL: llvm.func @id_i64(
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: -> i64
// CHECK: %[[R:.*]] = llvm.call @id_i64_c(%[[A]]) : (i64) -> i64
// CHECK: llvm.return %[[R]] : i64
