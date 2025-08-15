// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

// Define variant record types for testing dispatch
!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

module {
  func.func @test_incomplete_coverage(%opt_ref : !reussir.ref<!option>) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      // CHECK: error: 'reussir.record.dispatch' op tag 1 is not covered by any tag set
      [0] -> {
        ^bb0(%arg : !reussir.ref<!option_some>):
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
    }
    func.return %result : i32
  }
}
