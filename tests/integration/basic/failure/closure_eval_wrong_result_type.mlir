// RUN: %reussir-opt %s -verify-diagnostics

// Test failure: closure eval with wrong result type
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_eval_wrong_result_type() -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<() -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0:
          %result = arith.constant 42 : i32
          reussir.closure.yield %result : i32
      }
    }
    // This should fail because result type (i64) doesn't match closure output type (i32)
    // expected-error @+1 {{'reussir.closure.eval' op result type must match closure output type, result type: 'i64', closure output type: 'i32'}}
    %result = reussir.closure.eval (%closure : !reussir.closure<() -> i32>) : i64
    return %result : i64
  }
}
