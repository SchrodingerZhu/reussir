// RUN: %reussir-opt %s -verify-diagnostics

// Test failure cases for nullable coercion operations

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>>} {
  // Test mismatched type coercion
  func.func @test_nullable_coerce_type_mismatch() {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 8>
    %nullable = reussir.nullable.create(%token : !reussir.token<align: 8, size: 8>) : !reussir.nullable<!reussir.token<align: 8, size: 8>>
    // expected-error @+1 {{coerced type must match nullable pointer type}}
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 8>>) : !reussir.token<align: 8, size: 16>
    func.return
  }
}
