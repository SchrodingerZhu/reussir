// RUN: %reussir-opt %s  | %reussir-opt

// Test closure type conversion and layout
module {
  // Define a function that returns a closure
  func.func private @test_closure() -> !reussir.closure<(i32, i64) -> i32>

  // Test closure with different input/output types
  func.func private @test_closure_complex() -> !reussir.closure<() -> i64>

  // Test closure as function parameter
  func.func private @test_closure_param(%closure : !reussir.closure<(i32) -> i32>) -> i32
}


