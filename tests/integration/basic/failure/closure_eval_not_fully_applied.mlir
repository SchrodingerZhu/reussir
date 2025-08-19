// RUN: %reussir-opt %s -verify-diagnostics

// Test failure: closure eval with remaining input types
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_eval_not_fully_applied() -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%v0 : i32):
          %result = arith.constant 42 : i32
          reussir.closure.yield %result : i32
      }
    }
    // This should fail because the closure still has input types
    // expected-error @+1 {{'reussir.closure.eval' op cannot evaluate closure with remaining input types, closure has 1 input types remaining}}
    %result = reussir.closure.eval (%closure : !reussir.closure<(i32) -> i32>) : i32
    return %result : i32
  }
}
