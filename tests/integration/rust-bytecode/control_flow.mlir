// Nested, non-isolated regions (scf.if) round-trip: the cross-region value
// references and the value numbering that spans region boundaries must survive,
// and arith.constant's property must rematerialize from the attribute dict.
// RUN: %reussir-bytecode-demo --demo control_flow -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK:      func.func @pick(%arg0: i1) -> i32 {
// CHECK-NEXT:   %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:   %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:   %0 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:     scf.yield %c10_i32 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %c20_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
