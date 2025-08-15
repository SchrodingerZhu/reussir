// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

module {
  func.func @test_missing_nonnull_arg(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
      nonnull -> {
        // CHECK: error: 'reussir.nullable.dispatch' op nonnull region must have exactly one argument
        ^bb0:
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
      null -> {
        ^bb0:
          %c0 = arith.constant 0 : i32
          reussir.scf.yield %c0 : i32
      }
    }
    func.return %result : i32
  }
}
