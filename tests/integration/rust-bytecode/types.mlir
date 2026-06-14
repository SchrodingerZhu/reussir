// The remaining wrapped Reussir types survive the bytecode round-trip via the
// textual fallback: raw_ptr, hole, closure_box, array (with shape), and
// ffi_object (with name and cleanup symbol).
// RUN: %reussir-bytecode-demo --demo types -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc | %FileCheck %s

// CHECK: func.func @sig(%arg0: !reussir.raw_ptr<i32>, %arg1: !reussir.hole<i32>, %arg2: !reussir.closure_box<i32, i64>, %arg3: !reussir.array<2 x 3 x i32>, %arg4: !reussir.ffi_object<"Vec", @cleanup>)
