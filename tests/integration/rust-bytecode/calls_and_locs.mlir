// Symbol references (func.call) and source locations survive the bytecode
// round-trip. Locations are stored via the textual fallback and surface under
// --mlir-print-debuginfo.
// RUN: %reussir-bytecode-demo --demo calls_and_locs -o %t.mlirbc
// RUN: %reussir-opt %t.mlirbc --mlir-print-debuginfo | %FileCheck %s

// CHECK: call @callee(%arg0) : (i32) -> i32 loc(#[[LOC:.*]])
// CHECK: #[[LOC]] = loc("caller.rs":2:9)
