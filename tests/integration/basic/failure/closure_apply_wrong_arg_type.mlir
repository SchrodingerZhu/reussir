// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

// Test closure apply with wrong argument type
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_apply_wrong_arg_type() -> !reussir.closure<() -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          %add = arith.addi %v0, %one : i32 
          reussir.closure.yield %add : i32
      }
    }
    %arg = arith.constant 5 : i64  // Wrong type - should be i32
    // CHECK: argument type must match first closure input type, argument type: 'i64', expected type: 'i32'
    %applied = reussir.closure.apply (%arg : i64) to (%closure : !reussir.closure<(i32) -> i32>) : !reussir.closure<() -> i32>
    return %applied : !reussir.closure<() -> i32>
  }
}
