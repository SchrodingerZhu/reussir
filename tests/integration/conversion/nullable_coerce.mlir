// RUN: %reussir-opt %s --reussir-lowering-basic-ops | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @nullable_coerce_token(%nullable: !reussir.nullable<!reussir.token<align: 8, size: 64>>) -> !reussir.token<align: 8, size: 64> {
    // CHECK: ret ptr %0
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 64>>) : !reussir.token<align: 8, size: 64>
    return %coerced : !reussir.token<align: 8, size: 64>
  }

  func.func @nullable_coerce_rc(%nullable: !reussir.nullable<!reussir.rc<i64>>) -> !reussir.rc<i64> {
    // CHECK: ret ptr %0
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.rc<i64>>) : !reussir.rc<i64>
    return %coerced : !reussir.rc<i64>
  }

  func.func @nullable_coerce_ref(%nullable: !reussir.nullable<!reussir.ref<i64>>) -> !reussir.ref<i64> {
    // CHECK: ret ptr %0
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.ref<i64>>) : !reussir.ref<i64>
    return %coerced : !reussir.ref<i64>
  }
}
