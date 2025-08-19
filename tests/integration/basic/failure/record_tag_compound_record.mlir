// RUN: %reussir-opt %s -verify-diagnostics

// Define a compound record type
!compound_record = !reussir.record<compound "Compound" {i32, i32}>

module {
  func.func @test_compound_record(%ref : !reussir.ref<!compound_record>) -> index {
    // expected-error @+1 {{'reussir.record.tag' op can only get tag of variant records}}
    %tag = reussir.record.tag(%ref : !reussir.ref<!compound_record>) : index
    return %tag : index
  }
}
