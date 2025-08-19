// RUN: %reussir-opt %s -verify-diagnostics

// Define variant record types for testing dispatch
!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

module {
  func.func @test_single_tag_missing_arg(%opt_ref : !reussir.ref<!option>) -> i32 {
    // expected-error @+1 {{'reussir.record.dispatch' op region 0 must have exactly one argument for single tag}}
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0:
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
      [1] -> {
        ^bb0(%correct_arg : !reussir.ref<!option_none>):
          %c0 = arith.constant 0 : i32
          reussir.scf.yield %c0 : i32
      }
    }
    func.return %result : i32
  }
}
