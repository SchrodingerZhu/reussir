// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

!list_incomplete = !reussir.record<variant "List" incomplete>
!cons = !reussir.record<compound "List::Cons" [value] { i32, !list_incomplete }>
!nil = !reussir.record<compound "List::Nil" [value] {}>
!list = !reussir.record<variant "List" {!cons, !nil}>

!option_some = !reussir.record<compound "Option::Some" [value] {i32}>
!option_none = !reussir.record<compound "Option::None" [value] {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

!result_ok = !reussir.record<compound "Result::Ok" [value] {i32}>
!result_err = !reussir.record<compound "Result::Err" [value] {i32}>
!result = !reussir.record<variant "Result" {!result_ok, !result_err}>

module {
  func.func @cons(%fst : i32, %tail : !reussir.rc<!list>) -> !reussir.rc<!list> {
    %0 = reussir.record.compound(%fst, %tail : i32, !reussir.rc<!list>) : !cons
    %1 = reussir.record.variant [0] (%0 : !cons) : !list
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %rc = reussir.rc.create 
        value(%1 : !list) 
        token(%token : !reussir.token<align: 8, size: 24>) : !reussir.rc<!list>
    return %rc : !reussir.rc<!list>
  }

  func.func @test_option_tag(%opt_ref : !reussir.ref<!option>) -> index {
    %tag = reussir.record.tag(%opt_ref : !reussir.ref<!option>) : index
    return %tag : index
  }

  func.func @test_result_tag(%result_ref : !reussir.ref<!result>) -> index {
    %tag = reussir.record.tag(%result_ref : !reussir.ref<!result>) : index
    return %tag : index
  }
}

// Members are packed by descending alignment (the tail pointer precedes the
// i32 head); a shared variant carries the fused 8-byte box header
// {i32 count slot, i32 tag} and the box IS the record.
// CHECK-DAG: %"List::Cons" = type { ptr, i32, [4 x i8] }
// CHECK-DAG: %List = type { i32, i32, %"List::Cons" }

// CHECK-LABEL: define ptr @cons(i32 %0, ptr %1)
// Stack slots are hoisted to the entry block (static allocas); the fills
// stay at the use sites.
// CHECK: %[[alloca:[0-9]+]] = alloca %List, i64 1, align 8
// CHECK: %[[cons_alloca:[0-9]+]] = alloca %"List::Cons", i64 1, align 8
// CHECK: %[[cons_ptr0:[0-9]+]] = getelementptr %"List::Cons", ptr %[[cons_alloca]], i32 0, i32 1
// CHECK: store i32 %0, ptr %[[cons_ptr0]], align 4
// CHECK: %[[cons_ptr1:[0-9]+]] = getelementptr %"List::Cons", ptr %[[cons_alloca]], i32 0, i32 0
// CHECK: store ptr %1, ptr %[[cons_ptr1]], align 8
// CHECK: %[[cons_loaded:[0-9]+]] = load %"List::Cons", ptr %[[cons_alloca]], align 8
// CHECK: call void @llvm.lifetime.start.p0({{.*}}ptr %[[alloca]])
// CHECK: %[[tag_ptr:[0-9]+]] = getelementptr %List, ptr %[[alloca]], i32 0, i32 1
// CHECK: store i32 0, ptr %[[tag_ptr]], align 4
// CHECK: %[[value_ptr:[0-9]+]] = getelementptr %List, ptr %[[alloca]], i32 0, i32 2
// CHECK: store %"List::Cons" %[[cons_loaded]], ptr %[[value_ptr]], align 8
// CHECK: %[[loaded:[0-9]+]] = load %List, ptr %[[alloca]], align 8
// CHECK: call void @llvm.lifetime.end.p0({{.*}}ptr %[[alloca]])
// CHECK: %[[allocated:[0-9]+]] = call ptr @__reussir_allocate_small(i64 24)
// CHECK: %[[count_ptr:[0-9]+]] = getelementptr %List, ptr %[[allocated]], i32 0, i32 0
// CHECK: %[[box_ptr:[0-9]+]] = getelementptr %List, ptr %[[allocated]], i32 0, i32 0
// CHECK: store %List %[[loaded]], ptr %[[box_ptr]], align 8
// CHECK: store i32 1, ptr %[[count_ptr]], align 4
// CHECK: ret ptr %[[allocated]]

// CHECK-LABEL: define i64 @test_option_tag(ptr %0)
// CHECK: %[[tag_ptr:[0-9]+]] = getelementptr %Option, ptr %0, i32 0, i32 1
// CHECK: %[[narrow_tag:[0-9]+]] = load i32, ptr %[[tag_ptr]], align 4
// CHECK: %[[tag_value:[0-9]+]] = zext i32 %[[narrow_tag]] to i64
// CHECK: ret i64 %[[tag_value]]

// CHECK-LABEL: define i64 @test_result_tag(ptr %0)
// CHECK: %[[tag_ptr:[0-9]+]] = getelementptr %Result, ptr %0, i32 0, i32 1
// CHECK: %[[narrow_tag:[0-9]+]] = load i32, ptr %[[tag_ptr]], align 4
// CHECK: %[[tag_value:[0-9]+]] = zext i32 %[[narrow_tag]] to i64
// CHECK: ret i64 %[[tag_value]]
