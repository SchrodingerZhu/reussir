// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: 'reussir.record.tag' op operand #0 must be Reussir Reference Type, but got 'i32'
  func.func @test_invalid_type(%val : i32) -> index {
    %tag = reussir.record.tag(%val : i32) : index
    return %tag : index
  }
}
