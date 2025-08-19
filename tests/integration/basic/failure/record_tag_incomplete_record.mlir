// RUN: %reussir-opt %s -verify-diagnostics

// Define an incomplete variant record type
!incomplete_record = !reussir.record<variant "Incomplete" incomplete>

module {
  func.func @test_incomplete_record(%ref : !reussir.ref<!incomplete_record>) -> index {
    // expected-error @+1 {{'reussir.record.tag' op cannot get tag of incomplete record}}
    %tag = reussir.record.tag(%ref : !reussir.ref<!incomplete_record>) : index
    return %tag : index
  }
}
