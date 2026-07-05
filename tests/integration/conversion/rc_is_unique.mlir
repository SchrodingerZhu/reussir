// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

!test_struct = !reussir.record<compound "TestStruct" {i64, i64}>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  
  // CHECK-LABEL: define i1 @rc_is_unique_shared(ptr %0) {
  // CHECK: %2 = getelementptr { i32, i64 }, ptr %0, i32 0, i32 0
  // CHECK: %3 = load i32, ptr %2, align 4
  // CHECK: %4 = icmp eq i32 %3, 1
  // CHECK: ret i1 %4
  func.func @rc_is_unique_shared(%rc: !reussir.rc<i64>) -> i1 {
    %is_unique = reussir.rc.is_unique (%rc : !reussir.rc<i64>) : i1
    return %is_unique : i1
  }

  // Test with a more complex type
  // CHECK-LABEL: define i1 @rc_is_unique_struct(ptr %0) {
  // CHECK: %2 = getelementptr { i32, %TestStruct }, ptr %0, i32 0, i32 0
  // CHECK: %3 = load i32, ptr %2, align 4
  // CHECK: %4 = icmp eq i32 %3, 1
  // CHECK: ret i1 %4
  func.func @rc_is_unique_struct(%rc: !reussir.rc<!test_struct>) -> i1 {
    %is_unique = reussir.rc.is_unique (%rc : !reussir.rc<!test_struct>) : i1
    return %is_unique : i1
  }
}
