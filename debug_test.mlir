!variant_record = !reussir.record<variant "test_variant" {i32, f32}>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @test_record_coerce_mismatch() {
    %val = arith.constant 42 : i32
    %variant = reussir.record.variant[0](%val : i32) : !variant_record
    %ref = reussir.ref.spilled(%variant : !variant_record) : !reussir.ref<!variant_record>
    %coerced = reussir.record.coerce[0](%ref : !reussir.ref<!variant_record>) : !reussir.ref<f32>
    func.return
  }
}
