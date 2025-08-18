// RUN: %reussir-opt %s | %reussir-opt

// Test successful closure yield operations
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Test closure yield with correct return type
  func.func private @test_closure_yield_success() -> !reussir.closure<(i32) -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          %add = arith.addi %v0, %one : i32 
          reussir.closure.yield %add : i32
      }
    }
    return %0 : !reussir.closure<(i32) -> i32>
  }

  // Test closure yield with no return value (void)
  func.func private @test_closure_yield_void() -> !reussir.closure<(i32)> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32)> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          reussir.closure.yield
      }
    }
    return %0 : !reussir.closure<(i32)>
  }
}
