// Custom Reussir dialect types and operations survive the bytecode round-trip.
// The reference-counted type is carried through the bytecode's textual fallback
// encoding and re-parsed by reussir-opt's dialect.
// RUN: %reussir-bytecode-demo --demo reussir_types -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK:      module @rc_demo {
// CHECK-NEXT:   func.func @bump(%arg0: !reussir.rc<i32>) -> !reussir.rc<i32> {
// CHECK-NEXT:     reussir.rc.inc(%arg0 : !reussir.rc<i32>)
// CHECK-NEXT:     return %arg0 : !reussir.rc<i32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
