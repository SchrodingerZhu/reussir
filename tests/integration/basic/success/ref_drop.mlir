// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s
// Test basic reference drop operations

module {
  // CHECK: func.func @drop_basic(%arg0: !reussir.ref<i32>)
  func.func @drop_basic(%ref : !reussir.ref<i32>) {
    // CHECK: reussir.ref.drop(%arg0 : !reussir.ref<i32>)
    reussir.ref.drop (%ref : !reussir.ref<i32>)
    return
  }

  // CHECK: func.func @drop_outlined(%arg0: !reussir.ref<i64>)
  func.func @drop_outlined(%ref : !reussir.ref<i64>) {
    // CHECK: reussir.ref.drop outlined(%arg0 : !reussir.ref<i64>)
    reussir.ref.drop outlined (%ref : !reussir.ref<i64>)
    return
  }
}
