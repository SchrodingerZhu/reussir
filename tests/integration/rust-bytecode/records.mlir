// Record types (including a recursive one), record construction, and
// reussir.rc.create with its operandSegmentSizes property all round-trip and
// verify. The recursive List type must print with name-only back-references.
// RUN: %reussir-bytecode-demo --demo records -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK: func.func @make_pair(%arg0: i32, %arg1: i64) -> !reussir.rc<!reussir.record<compound "Pair" [value] {i32, i64}>>
// CHECK:   %0 = reussir.record.compound(%arg0, %arg1 : i32, i64) : !reussir.record<compound "Pair" [value] {i32, i64}>
// CHECK:   %1 = reussir.rc.create value(%0 {{.*}}) : !reussir.rc<!reussir.record<compound "Pair" [value] {i32, i64}>>
// CHECK: func.func @id(%arg0: !reussir.rc<!reussir.record<variant "List" {!reussir.record<compound "List::Cons" [value] {i32, !reussir.record<variant "List">}>, !reussir.record<compound "List::Nil" [value] {}>}>>)
// CHECK:   reussir.rc.inc(%arg0
