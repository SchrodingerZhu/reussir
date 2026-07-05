// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK-LABEL: define void @rc_assume_unique(ptr %0) {
  // CHECK: %2 = getelementptr { i32, i64 }, ptr %0, i32 0, i32 0
  // CHECK: %3 = load i32, ptr %2, align 4
  // CHECK: %4 = icmp eq i32 %3, 1
  // CHECK: call void @llvm.assume(i1 %4)
  func.func @rc_assume_unique(%rc: !reussir.rc<i64>) {
    reussir.rc.assume_unique(%rc : !reussir.rc<i64>)
    return
  }
}
