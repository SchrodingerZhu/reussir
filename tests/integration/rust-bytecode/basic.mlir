// The reussir-bytecode writer emits MLIR bytecode directly; reussir-opt must
// decode it back to the expected textual module. Exercises builtin types, the
// func/arith dialects, operands, and results.
// RUN: %reussir-bytecode-demo --demo basic -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK:      module @demo {
// CHECK-NEXT:   func.func @add(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:     %0 = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
