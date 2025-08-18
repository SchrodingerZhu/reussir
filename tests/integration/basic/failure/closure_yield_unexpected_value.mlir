// RUN: not %reussir-opt %s 2>&1 | FileCheck %s

// Test closure yield with return value when none expected
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_yield_unexpected_value() -> !reussir.closure<(i32)> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32)> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          // CHECK: closure has no return type but yield provides value of type 'i32'
          reussir.closure.yield %one : i32
      }
    }
    return %0 : !reussir.closure<(i32)>
  }
}
