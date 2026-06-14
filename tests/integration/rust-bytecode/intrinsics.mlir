// The arith/math intrinsic emitters produce structurally correct bytecode:
// integer/float arithmetic, comparison (predicate property), conversion, a math
// call (default fastmath), and a select all round-trip and verify.
// RUN: %reussir-bytecode-demo --demo intrinsics -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK:      func.func @compute(%arg0: i32, %arg1: i32, %arg2: f32) -> f32 {
// CHECK-NEXT:   %0 = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT:   %1 = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK-NEXT:   %2 = arith.sitofp %0 : i32 to f32
// CHECK-NEXT:   %3 = math.sqrt %arg2 : f32
// CHECK-NEXT:   %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:   %5 = arith.select %1, %4, %arg2 : f32
// CHECK-NEXT:   return %5 : f32
// CHECK-NEXT: }
