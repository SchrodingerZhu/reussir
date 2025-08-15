// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

// Define an incomplete variant record type
!incomplete_record = !reussir.record<variant "Incomplete" incomplete>

module {
  // CHECK: error: 'reussir.record.tag' op cannot get tag of incomplete record
  func.func @test_incomplete_record(%ref : !reussir.ref<!incomplete_record>) -> index {
    %tag = reussir.record.tag(%ref : !reussir.ref<!incomplete_record>) : index
    return %tag : index
  }
}
