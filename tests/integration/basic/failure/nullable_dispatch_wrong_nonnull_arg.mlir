// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

module {
  func.func @test_wrong_nonnull_arg_type(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
      nonnull -> {
        // CHECK: error: 'reussir.nullable.dispatch' op nonnull region argument type must match nullable inner type, argument type: '!reussir.ref<i32>', expected type: '!reussir.rc<i64>'
        ^bb0(%wrong_arg : !reussir.ref<i32>):
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
