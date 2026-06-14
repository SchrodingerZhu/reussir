// Foreign-function-interface ops round-trip: a reussir.trampoline exporting a
// function under the C ABI, and a reussir.polyffi stub with a substitution dict.
// RUN: %reussir-bytecode-demo --demo ffi -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK: func.func @id_target(%arg0: i32) -> i32
// CHECK: reussir.trampoline export "C" @id_target_ffi = @id_target
// CHECK: reussir.polyffi texture("extern_call(%0)") substitutions({Elem = i32, Size = 4 : i64})
