// RUN: %not %reussir-opt %s 2>&1 | %filecheck %s

// Test closure yield with no value when one expected
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @test_closure_yield_missing_value() -> !reussir.closure<(i32) -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          // CHECK: closure has return type 'i32' but yield provides no value
          reussir.closure.yield
      }
    }
    return %0 : !reussir.closure<(i32) -> i32>
  }
}
