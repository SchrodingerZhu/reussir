// RUN: %reussir-opt %s | %reussir-opt

// Test successful closure apply operations
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Test basic closure apply with single argument
  func.func private @test_closure_apply_basic() -> !reussir.closure<() -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<(i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          %add = arith.addi %v0, %one : i32 
          reussir.closure.yield %add : i32
      }
    }
    %arg = arith.constant 5 : i32
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.closure<(i32) -> i32>) : !reussir.closure<() -> i32>
    return %applied : !reussir.closure<() -> i32>
  }

  // Test closure apply with multiple arguments
  func.func private @test_closure_apply_multiple() -> !reussir.closure<(i64) -> i32> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.closure<(i32, i64) -> i32> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%v0 : i32, %v1 : i64):
          %v0_i64 = arith.extsi %v0 : i32 to i64
          %add = arith.addi %v0_i64, %v1 : i64
          %result = arith.trunci %add : i64 to i32
          reussir.closure.yield %result : i32
      }
    }
    %arg = arith.constant 10 : i32
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.closure<(i32, i64) -> i32>) : !reussir.closure<(i64) -> i32>
    return %applied : !reussir.closure<(i64) -> i32>
  }

  // Test closure apply with void return type
  func.func private @test_closure_apply_void() -> !reussir.closure<()> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.closure<(i32)> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          reussir.closure.yield
      }
    }
    %arg = arith.constant 42 : i32
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.closure<(i32)>) : !reussir.closure<()>
    return %applied : !reussir.closure<()>
  }

  // Test closure apply with complex types
  func.func private @test_closure_apply_complex() -> !reussir.closure<(i32) -> i64> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 48>
    %closure = reussir.closure.create -> !reussir.closure<(i32, i64, i32) -> i64> {
      token(%token : !reussir.token<align: 8, size: 48>)
      body {
        ^bb0(%v0 : i32, %v1 : i64, %v2 : i32):
          %v0_i64 = arith.extsi %v0 : i32 to i64
          %v2_i64 = arith.extsi %v2 : i32 to i64
          %sum1 = arith.addi %v0_i64, %v1 : i64
          %sum2 = arith.addi %sum1, %v2_i64 : i64
          reussir.closure.yield %sum2 : i64
      }
    }
    %arg1 = arith.constant 1 : i32
    %arg2 = arith.constant 2 : i64
    %applied1 = reussir.closure.apply (%arg1 : i32) to (%closure : !reussir.closure<(i32, i64, i32) -> i64>) : !reussir.closure<(i64, i32) -> i64>
    %applied2 = reussir.closure.apply (%arg2 : i64) to (%applied1 : !reussir.closure<(i64, i32) -> i64>) : !reussir.closure<(i32) -> i64>
    return %applied2 : !reussir.closure<(i32) -> i64>
  }
}
