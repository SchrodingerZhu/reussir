// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

// Define variant record types
!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

!result_ok = !reussir.record<compound "Result::Ok" {i32}>
!result_err = !reussir.record<compound "Result::Err" {i32}>
!result = !reussir.record<variant "Result" {!result_ok, !result_err}>

module {
  // Test getting tag from a variant record reference
  func.func @test_option_tag(%opt_ref : !reussir.ref<!option>) -> index {
    // CHECK: reussir.record.tag
    // CHECK: return
    %tag = reussir.record.tag(%opt_ref : !reussir.ref<!option>) : index
    return %tag : index
  }

  // Test getting tag from a result variant record
  func.func @test_result_tag(%result_ref : !reussir.ref<!result>) -> index {
    // CHECK: reussir.record.tag
    // CHECK: return
    %tag = reussir.record.tag(%result_ref : !reussir.ref<!result>) : index
    return %tag : index
  }
}
