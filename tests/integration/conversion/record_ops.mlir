// RUN: %reussir-opt %s --reussir-lowering-basic-ops | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!list_incomplete = !reussir.record<variant "List" incomplete>
!cons = !reussir.record<compound "List::Cons" { i32, [shared] !list_incomplete }>
!nil = !reussir.record<compound "List::Nil" {}>
!list = !reussir.record<variant "List" {!cons, !nil}>

module {
  func.func @cons(%fst : i32, %tail : !reussir.rc<!list>) -> !reussir.rc<!list> {
    %0 = reussir.record.compound(%fst, %tail : i32, !reussir.rc<!list>) : !cons
    %1 = reussir.record.variant [0] (%0 : !cons) : !list
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %rc = reussir.rc.create 
        value(%1 : !list) 
        token(%token : !reussir.token<align: 8, size: 32>) : !reussir.rc<!list>
    return %rc : !reussir.rc<!list>
  }
}

// CHECK-LABEL: define ptr @cons(i32 %0, ptr %1)
// CHECK: %[[cons_undef:[0-9]+]] = insertvalue %"List::Cons" undef, i32 %0, 0
// CHECK: %[[cons_with_tail:[0-9]+]] = insertvalue %"List::Cons" %[[cons_undef]], ptr %1, 1
// CHECK: %[[alloca:[0-9]+]] = alloca %List, i64 1, align 8
// CHECK: %[[tag_ptr:[0-9]+]] = getelementptr %List, ptr %[[alloca]], i32 0, i32 0
// CHECK: store i64 0, ptr %[[tag_ptr]], align 4
// CHECK: %[[value_ptr:[0-9]+]] = getelementptr %List, ptr %[[alloca]], i32 0, i32 1
// CHECK: store %"List::Cons" %[[cons_with_tail]], ptr %[[value_ptr]], align 8
// CHECK: %[[loaded:[0-9]+]] = load %List, ptr %[[alloca]], align 8
// CHECK: %[[allocated:[0-9]+]] = call ptr @__reussir_allocate(i64 8, i64 32)
// CHECK: %[[rc_tag_ptr:[0-9]+]] = getelementptr { i64, %List }, ptr %[[allocated]], i32 0, i32 0
// CHECK: %[[rc_value_ptr:[0-9]+]] = getelementptr { i64, %List }, ptr %[[allocated]], i32 0, i32 1
// CHECK: store i64 1, ptr %[[rc_value_ptr]], align 4
// CHECK: store %List %[[loaded]], ptr %[[rc_tag_ptr]], align 8
// CHECK: ret ptr %[[allocated]]
