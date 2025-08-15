// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

module {
  func.func @test_wrong_yield_type(%nullable : !reussir.nullable<!reussir.ref<i32>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.ref<i32>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.ref<i32>):
          %wrong_value = arith.constant 42 : i64
          // CHECK: error: 'reussir.scf.yield' op yielded type must match parent operation result type, yielded type: 'i64', expected type: 'i32'
          reussir.scf.yield %wrong_value : i64
      }
      null -> {
        ^bb0:
          %default = arith.constant 0 : i32
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }
}
