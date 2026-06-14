// reussir.record.dispatch with multiple regions carrying block arguments, plus
// reussir.ref.project, reussir.ref.load, and reussir.scf.yield, round-trips and
// verifies. This exercises non-isolated nested regions whose blocks take typed
// arguments referenced by the body.
// RUN: %reussir-bytecode-demo --demo dispatch -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK: func.func @unwrap_or
// CHECK:   reussir.record.dispatch(%arg0
// CHECK:     [0] -> {
// CHECK:     ^bb0(%arg2: !reussir.ref<!reussir.record<compound "Option::Some" [value] {i32}>>):
// CHECK:       %{{.*}} = reussir.ref.project(%arg2 {{.*}}) [0] : !reussir.ref<i32>
// CHECK:       %{{.*}} = reussir.ref.load(%{{.*}} : !reussir.ref<i32>) : i32
// CHECK:       reussir.scf.yield
// CHECK:     [1] -> {
// CHECK:       reussir.scf.yield %arg1 : i32
