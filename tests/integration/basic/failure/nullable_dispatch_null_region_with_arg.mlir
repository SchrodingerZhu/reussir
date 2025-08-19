// RUN: %reussir-opt %s -verify-diagnostics

module {
  func.func @test_null_region_with_arg(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
    // expected-error @+1 {{'reussir.nullable.dispatch' op null region must have no arguments}}
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i64>):
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
      null -> {
        ^bb0(%unexpected_arg : i32):
          %c0 = arith.constant 0 : i32
          reussir.scf.yield %c0 : i32
      }
    }
    func.return %result : i32
  }
}
