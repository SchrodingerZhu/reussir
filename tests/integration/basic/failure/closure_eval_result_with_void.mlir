// RUN: %reussir-opt %s -verify-diagnostics

// Test failure: closure eval with result closure but expecting void
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_eval_result_with_void() {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<() -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0:
          %result = arith.constant 42 : i32
          reussir.closure.yield %result : i32
      }
    }
    // This should fail because closure has output type i32 but we expect void
    // expected-error @+1 {{'reussir.closure.eval' op closure has output type 'i32' but result is empty}}
    reussir.closure.eval (%closure : !reussir.closure<() -> i32>)
    return
  }
}
