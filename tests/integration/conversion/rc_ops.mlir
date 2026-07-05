// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

!nullable = !reussir.nullable<!reussir.ref<i64>>
!list_incomplete = !reussir.record<variant "List" incomplete>
!cons = !reussir.record<compound "List::Cons" [value] { f128, !list_incomplete }>
!nil = !reussir.record<compound "List::Nil" [value] {}>
!padding = !reussir.record<compound "List::Padding" [value] { i64, i64, i64, i64, i64, i64, i64, i64 }>
!list = !reussir.record<variant "List" {!cons, !nil, !padding}>
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK-LABEL: define void @rc_inc(ptr %0) {
  // CHECK: %2 = getelementptr { i32, i64 }, ptr %0, i32 0, i32 0
  // CHECK: %3 = load i32, ptr %2, align 4
  // CHECK: %4 = add i32 %3, 1
  // CHECK: store i32 %4, ptr %2, align 4
  // CHECK: %5 = icmp uge i32 %3, 1
  // CHECK: call void @llvm.assume(i1 %5)
  func.func @rc_inc(%rc: !reussir.rc<i64>){
    reussir.rc.inc (%rc : !reussir.rc<i64>)
    return 
  }

  // CHECK-LABEL: define void @rc_inc_atomic(ptr %0) {
  // CHECK: %2 = getelementptr { i32, i64 }, ptr %0, i32 0, i32 0
  // CHECK: %3 = atomicrmw add ptr %2, i32 1 monotonic, align 4
  // CHECK: %4 = icmp uge i32 %3, 1
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

  // CHECK-LABEL: define i64 @rc_fetch(ptr %0) {
  // CHECK: %2 = load i32, ptr %0, align 4
  // CHECK: %3 = zext i32 %2 to i64
  // CHECK: ret i64 %3
  func.func @rc_fetch(%rc: !reussir.rc<i64>) -> index {
    %count = reussir.rc.fetch (%rc : !reussir.rc<i64>) : index
    return %count : index
  }

  // CHECK-LABEL: define void @rc_set(ptr %0, i64 %1) {
  // CHECK: %3 = trunc i64 %1 to i32
  // CHECK: store i32 %3, ptr %0, align 4
  // CHECK: ret void
  func.func @rc_set(%rc: !reussir.rc<i64>, %count: index) {
    reussir.rc.set (%rc : !reussir.rc<i64>, %count : index)
    return
  }

  // CHECK-LABEL: define i1 @likely(i1 %0) {
  // CHECK: %2 = call i1 @llvm.expect.i1(i1 %0, i1 true)
  // CHECK: ret i1 %2
  func.func @likely(%cond: i1) -> i1 {
    %likely = reussir.expect (%cond : i1, 1) : i1
    return %likely : i1
  }

  // CHECK-LABEL: define ptr @rc_create(fp128 %0)
  // CHECK: %2 = call ptr @__reussir_allocate(i64 16, i64 32)
  // CHECK: %3 = getelementptr { i32, fp128 }, ptr %2, i32 0, i32 0
  // CHECK: %4 = getelementptr { i32, fp128 }, ptr %2, i32 0, i32 1
  // CHECK: store fp128 %0, ptr %4, align 16
  // CHECK: store i32 1, ptr %3, align 4
  func.func @rc_create(%value: f128) -> !reussir.rc<f128> {
    %token = reussir.token.alloc : !reussir.token<align: 16, size: 32>
    %rc = reussir.rc.create 
      value(%value : f128) 
      token(%token : !reussir.token<align: 16, size: 32>)  : !reussir.rc<f128>
    return %rc : !reussir.rc<f128>
  }

  // CHECK-LABEL: define fp128 @rc_borrow_then_load_0(ptr %0) {
  // CHECK: %2 = getelementptr { i32, fp128 }, ptr %0, i32 0, i32 1
  // CHECK: %3 = load fp128, ptr %2, align 16
  // CHECK: ret fp128 %3
  func.func @rc_borrow_then_load_0(%rc : !reussir.rc<f128>) -> f128 {
    %borrowed = reussir.rc.borrow(%rc : !reussir.rc<f128>) : !reussir.ref<f128 shared>
    %value = reussir.ref.load(%borrowed : !reussir.ref<f128 shared>) : f128
    return %value : f128
  }

  // CHECK-LABEL: define fp128 @rc_borrow_then_load_1(ptr %0) {
  // CHECK: %2 = getelementptr { ptr, ptr, ptr, fp128 }, ptr %0, i32 0, i32 3
  // CHECK: %3 = load fp128, ptr %2, align 16
  // CHECK: ret fp128 %3
  func.func @rc_borrow_then_load_1(%rc : !reussir.rc<f128 rigid>) -> f128 {
    %borrowed = reussir.rc.borrow(%rc : !reussir.rc<f128 rigid>) : !reussir.ref<f128 rigid>
    %value = reussir.ref.load(%borrowed : !reussir.ref<f128 rigid>) : f128
    return %value : f128
  }

  func.func @rc_borrow_then_load_2(%rc : !reussir.rc<!list>) -> !list {
    %borrowed = reussir.rc.borrow(%rc : !reussir.rc<!list>) : !reussir.ref<!list shared>
    // CHECK: load %List, ptr %2, align 16
    %value = reussir.ref.load(%borrowed : !reussir.ref<!list shared>) : !list
    return %value : !list
  }
}
