// RUN: %reussir-opt %s -verify-diagnostics

module {
  func.func @test_invalid_type(%val : i32) -> index {
    // expected-error @+1 {{'reussir.record.tag' op operand #0 must be Reussir Reference Type, but got 'i32'}}
    %tag = reussir.record.tag(%val : i32) : index
    return %tag : index
  }
}
