// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

// Define a compound record type
!compound_record = !reussir.record<compound "Compound" {i32, i32}>

module {
  // CHECK: error: 'reussir.record.tag' op can only get tag of variant records
  func.func @test_compound_record(%ref : !reussir.ref<!compound_record>) -> index {
    %tag = reussir.record.tag(%ref : !reussir.ref<!compound_record>) : index
    return %tag : index
  }
}
