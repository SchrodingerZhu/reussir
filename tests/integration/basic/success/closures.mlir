// RUN: %reussir-opt %s  | %reussir-opt

// Test closure type conversion and layout
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Define a function that returns a closure
  func.func private @test_closure() -> !reussir.closure<(i32, i64) -> i32>

  // Test closure with different input/output types
  func.func private @test_closure_complex() -> !reussir.closure<() -> i64>

  // Test closure as function parameter
  func.func private @test_closure_param(%closure : !reussir.closure<(i32) -> i32>) -> i32

  // Test closure create op
  func.func private @test_closure_create() -> !reussir.closure<(i32) -> i32> {
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

  func.func private @test_closure_create_outlined() -> !reussir.closure<(i32) -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %0 = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      vtable(@VTable)
    }
    return %0 : !reussir.closure<(i32) -> i32>
  }
}


