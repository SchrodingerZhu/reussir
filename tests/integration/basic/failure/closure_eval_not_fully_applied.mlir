// RUN: %reussir-opt %s -verify-diagnostics

// Test failure: closure eval with remaining input types
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_eval_not_fully_applied() -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%v0 : i32):
          %result = arith.constant 42 : i32
          reussir.closure.yield %result : i32
      }
    }
    // This should fail because the closure still has input types and no
    // `with` pack supplies them
    // expected-error @+1 {{'reussir.closure.eval' op closure has 1 remaining input types but 0 eval arguments are supplied}}
    %result = reussir.closure.eval (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : i32
    return %result : i32
  }
}
