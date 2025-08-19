// RUN: %reussir-opt %s -verify-diagnostics

// Test failure: closure eval with void closure but expecting result
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_eval_void_with_result() -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<()> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0:
          reussir.closure.yield
      }
    }
    // This should fail because closure has no output type but we expect i32
    // expected-error @+1 {{'reussir.closure.eval' op closure has no output type but result is not empty}}
    %result = reussir.closure.eval (%closure : !reussir.closure<()>) : i32
    return %result : i32
  }
}
