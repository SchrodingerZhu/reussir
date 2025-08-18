// RUN: %reussir-opt %s | %reussir-opt

// Test successful closure eval operations
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Test basic closure eval with return value
  func.func private @test_closure_eval_basic() -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %closure = reussir.closure.create -> !reussir.closure<() -> i32> {
      token(%token : !reussir.token<align: 8, size: 24>)
      body {
        ^bb0:
          %result = arith.constant 42 : i32
          reussir.closure.yield %result : i32
      }
    }
    %result = reussir.closure.eval (%closure : !reussir.closure<() -> i32>) : i32
    return %result : i32
  }

  // Test closure eval with void return type
  func.func private @test_closure_eval_void() {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %closure = reussir.closure.create -> !reussir.closure<()> {
      token(%token : !reussir.token<align: 8, size: 24>)
      body {
        ^bb0:
          reussir.closure.yield
      }
    }
    reussir.closure.eval (%closure : !reussir.closure<()>)
    return
  }

  // Test closure eval with complex return type
  func.func private @test_closure_eval_complex() -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %closure = reussir.closure.create -> !reussir.closure<() -> i64> {
      token(%token : !reussir.token<align: 8, size: 24>)
      body {
        ^bb0:
          %val1 = arith.constant 10 : i64
          %val2 = arith.constant 32 : i64
          %result = arith.addi %val1, %val2 : i64
          reussir.closure.yield %result : i64
      }
    }
    %result = reussir.closure.eval (%closure : !reussir.closure<() -> i64>) : i64
    return %result : i64
  }

  // Test closure eval after apply operations
  func.func private @test_closure_eval_after_apply() -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.closure<(i32, i32) -> i32> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%v0 : i32, %v1 : i32):
          %result = arith.addi %v0, %v1 : i32
          reussir.closure.yield %result : i32
      }
    }
    %arg1 = arith.constant 5 : i32
    %arg2 = arith.constant 7 : i32
    %applied1 = reussir.closure.apply (%arg1 : i32) to (%closure : !reussir.closure<(i32, i32) -> i32>) : !reussir.closure<(i32) -> i32>
    %applied2 = reussir.closure.apply (%arg2 : i32) to (%applied1 : !reussir.closure<(i32) -> i32>) : !reussir.closure<() -> i32>
    %result = reussir.closure.eval (%applied2 : !reussir.closure<() -> i32>) : i32
    return %result : i32
  }
}
