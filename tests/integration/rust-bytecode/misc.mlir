// Broad coverage: string global/literal/cast/len, nullable check/create,
// scf.index_switch (default region encoded first, cases in order), and panic.
// RUN: %reussir-bytecode-demo --demo misc -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK: reussir.str.global @greeting = "hello"
// CHECK: func.func @string_len() -> index
// CHECK:   reussir.str.literal @greeting
// CHECK:   reussir.str.cast
// CHECK:   reussir.str.len
// CHECK: func.func @is_null({{.*}}!reussir.nullable<!reussir.rc<i32>>) -> i1
// CHECK:   reussir.nullable.check
// CHECK: func.func @wrap
// CHECK:   reussir.nullable.create
// CHECK: func.func @classify(%arg0: index) -> i32
// CHECK:     case 0 {
// CHECK:       %{{.*}} = arith.constant 10 : i32
// CHECK:     case 1 {
// CHECK:       %{{.*}} = arith.constant 20 : i32
// CHECK:     default {
// CHECK:       %{{.*}} = arith.constant 0 : i32
// CHECK: func.func @boom
// CHECK:   reussir.panic "unreachable"
