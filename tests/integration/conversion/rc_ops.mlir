// RUN: %reussir-opt %s --reussir-lowering-basic-ops | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!nullable = !reussir.nullable<!reussir.ref<i64>>
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // CHECK-LABEL: define void @rc_inc(ptr %0) {
  // CHECK: %2 = getelementptr { i64, i64 }, ptr %0, i32 0, i32 1
  // CHECK: %3 = load i64, ptr %2, align 8
  // CHECK: %4 = add i64 %3, 1
  // CHECK: store i64 %4, ptr %2, align 8
  // CHECK: %5 = icmp uge i64 %3, 1
  // CHECK: call void @llvm.assume(i1 %5)
  func.func @rc_inc(%rc: !reussir.rc<i64>){
    reussir.rc.inc (%rc : !reussir.rc<i64>)
    return 
  }

  // CHECK-LABEL: define void @rc_inc_atomic(ptr %0) {
  // CHECK: %2 = getelementptr { i64, i64 }, ptr %0, i32 0, i32 1
  // CHECK: %3 = atomicrmw add ptr %2, i64 1 monotonic, align 8
  // CHECK: %4 = icmp uge i64 %3, 1
  // CHECK: call void @llvm.assume(i1 %4)
  func.func @rc_inc_atomic(%rc: !reussir.rc<i64 atomic>){
    reussir.rc.inc (%rc : !reussir.rc<i64 atomic>)
    return 
  }

  func.func @rc_inc_rigid(%rc: !reussir.rc<i64 rigid>){
    // CHECK-LABEL: call void @__reussir_acquire_rigid_object(ptr %0)
    reussir.rc.inc (%rc : !reussir.rc<i64 rigid>)
    return 
  }
}
