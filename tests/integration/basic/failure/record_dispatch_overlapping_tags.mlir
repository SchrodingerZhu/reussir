// RUN: %reussir-opt %s -verify-diagnostics

// Define variant record types for testing dispatch
!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

module {
  func.func @test_overlapping_tags(%opt_ref : !reussir.ref<!option>) -> i32 {
    // expected-error @+1 {{'reussir.record.dispatch' op tag 0 in tag set 1 is already covered by a previous tag set}}
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%arg : !reussir.ref<!option_some>):
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
      [0, 1] -> {
        ^bb0:
          %c0 = arith.constant 0 : i32
          reussir.scf.yield %c0 : i32
      }
    }
    func.return %result : i32
  }
}
