// RUN: %reussir-opt %s | %reussir-opt

!variant_record = !reussir.record<variant "test_variant" {i32, i64, f32}>
!compound_record = !reussir.record<compound "test_compound" {i32, i64}>

module @test {
  func.func @record_coerce_first_variant(%variant_ref : !reussir.ref<!variant_record>) {
    %coerced = reussir.record.coerce[0](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<i32>
    func.return
  }

  func.func @record_coerce_second_variant(%variant_ref : !reussir.ref<!variant_record>) {
    %coerced = reussir.record.coerce[1](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<i64>
    func.return
  }

  func.func @record_coerce_third_variant(%variant_ref : !reussir.ref<!variant_record>) {
    %coerced = reussir.record.coerce[2](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<f32>
    func.return
  }

  func.func @record_coerce_with_capabilities(%variant_ref : !reussir.ref<!variant_record shared>) {
    %coerced = reussir.record.coerce[0](%variant_ref : !reussir.ref<!variant_record shared>) : !reussir.ref<i32 shared>
    func.return
  }
}
