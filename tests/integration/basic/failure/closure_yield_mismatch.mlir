// RUN: not %reussir-opt %s 2>&1 | FileCheck %s

// Test closure yield with mismatched return type
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_yield_mismatch() -> !reussir.closure<(i32) -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i64  // Wrong type - should be i32
          // CHECK: yielded type must match closure return type, yielded type: 'i64', expected type: 'i32'
          reussir.closure.yield %one : i64
      }
    }
    return %0 : !reussir.closure<(i32) -> i32>
  }
}
