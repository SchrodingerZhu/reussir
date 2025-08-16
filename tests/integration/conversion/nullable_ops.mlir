// RUN: %reussir-opt %s --reussir-lowering-basic-ops | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!nullable = !reussir.nullable<!reussir.ref<i64>>
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @nullable_check(%nullable: !nullable) -> i1 {
    // CHECK: icmp ne ptr %0, null
    %result = reussir.nullable.check (%nullable : !nullable) : i1
    return %result : i1
  }

  func.func @nullable_create_null() -> !nullable {
    // CHECK: ret ptr null
    %null = reussir.nullable.create : !nullable
    return %null : !nullable
  }

  func.func @nullable_create_ref(%ref: !reussir.ref<i64>) -> !nullable {
    // CHECK: ret ptr %0
    %not_null = reussir.nullable.create (%ref : !reussir.ref<i64>) : !nullable
    return %not_null : !nullable
  }

}
