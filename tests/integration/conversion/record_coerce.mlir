// RUN: %reussir-opt %s --reussir-lowering-basic-ops | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!variant_record = !reussir.record<variant "test_variant" {i32, i64, f32}>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @record_coerce_first_variant(%variant_ref : !reussir.ref<!variant_record>) -> !reussir.ref<i32> {
    // CHECK: %[[GEP:.*]] = getelementptr %test_variant, ptr %0, i32 0, i32 1
    // CHECK: ret ptr %[[GEP]]
    %coerced = reussir.record.coerce[0](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<i32>
    return %coerced : !reussir.ref<i32>
  }

  func.func @record_coerce_second_variant(%variant_ref : !reussir.ref<!variant_record>) -> !reussir.ref<i64> {
    // CHECK: %[[GEP:.*]] = getelementptr %test_variant, ptr %0, i32 0, i32 1
    // CHECK: ret ptr %[[GEP]]
    %coerced = reussir.record.coerce[1](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<i64>
    return %coerced : !reussir.ref<i64>
  }

  func.func @record_coerce_third_variant(%variant_ref : !reussir.ref<!variant_record>) -> !reussir.ref<f32> {
    // CHECK: %[[GEP:.*]] = getelementptr %test_variant, ptr %0, i32 0, i32 1
    // CHECK: ret ptr %[[GEP]]
    %coerced = reussir.record.coerce[2](%variant_ref : !reussir.ref<!variant_record>) : !reussir.ref<f32>
    return %coerced : !reussir.ref<f32>
  }

  func.func @record_coerce_with_capabilities(%variant_ref : !reussir.ref<!variant_record shared>) -> !reussir.ref<i32 shared> {
    // CHECK: %[[GEP:.*]] = getelementptr %test_variant, ptr %0, i32 0, i32 1
    // CHECK: ret ptr %[[GEP]]
    %coerced = reussir.record.coerce[0](%variant_ref : !reussir.ref<!variant_record shared>) : !reussir.ref<i32 shared>
    return %coerced : !reussir.ref<i32 shared>
  }
}
