// RUN: %reussir-opt %s -verify-diagnostics

// Test failure cases for record coercion operations

!variant_record = !reussir.record<variant "test_variant" {i32, f32}>
!compound_record = !reussir.record<compound "test_compound" {i32, f32}>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f32, dense<32> : vector<2xi64>>>} {
  // Test coercing non-record type
  func.func @test_record_coerce_non_record() {
    %val = arith.constant 42 : i32
    %ref = reussir.ref.spilled(%val : i32) : !reussir.ref<i32>
    // expected-error @+1 {{input must be a reference to a record type}}
    %coerced = reussir.record.coerce[0](%ref : !reussir.ref<i32>) : !reussir.ref<i32>
    func.return
  }
}
