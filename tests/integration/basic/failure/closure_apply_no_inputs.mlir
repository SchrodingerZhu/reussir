// RUN: %reussir-opt %s -verify-diagnostics

// Test closure apply with closure that has no input types
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_apply_no_inputs() -> !reussir.closure<() -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %closure = reussir.closure.create -> !reussir.closure<() -> i32> {
      token(%token : !reussir.token<align: 8, size: 24>)
      body {
        ^bb0():
          %one = arith.constant 1 : i32
          reussir.closure.yield %one : i32
      }
    }
    %arg = arith.constant 5 : i32
    // expected-error @+1 {{cannot apply to closure with no input types}}
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.closure<() -> i32>) : !reussir.closure<() -> i32>
    return %applied : !reussir.closure<() -> i32>
  }
}
