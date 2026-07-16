// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s

// Test closure uniqify operation rewrite pattern
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @test_closure_uniqify() -> !reussir.rc<!reussir.closure<(i32) -> i32>> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %original = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          %add = arith.addi %v0, %one : i32 
          reussir.closure.yield %add : i32
      }
    }
    // Test uniqify operation - should be rewritten to SCF if-else with rc.is_unique check
    %uniqified = reussir.closure.uniqify (%original : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    return %uniqified : !reussir.rc<!reussir.closure<(i32) -> i32>>
  }
}

// CHECK: func.func @test_closure_uniqify() -> !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK: %[[TOKEN:.*]] = reussir.token.alloc : <align : 8, size : 32>
// CHECK: %[[ORIGINAL:.*]] = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK: %[[IS_UNIQUE:.*]] = reussir.rc.is_unique(%[[ORIGINAL]] : !reussir.rc<!reussir.closure<(i32) -> i32>>) : i1
// CHECK: %[[RESULT:.*]] = scf.if %[[IS_UNIQUE]] -> (!reussir.rc<!reussir.closure<(i32) -> i32>>) {
// CHECK:   scf.yield %[[ORIGINAL]] : !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK: } else {
// CHECK:   %[[CLONED:.*]] = reussir.closure.clone(%[[ORIGINAL]] : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK:   scf.yield %[[CLONED]] : !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK: }
// CHECK: return %[[RESULT]] : !reussir.rc<!reussir.closure<(i32) -> i32>>
