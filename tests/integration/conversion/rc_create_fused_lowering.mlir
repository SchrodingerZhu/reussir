// RUN: %reussir-opt %s --reussir-rc-create-fusion | %FileCheck %s --check-prefix=FUSION
// RUN: %reussir-opt %s --reussir-token-instantiation --reussir-rc-create-fusion --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

!cons = !reussir.record<compound "List::Cons" [value] { i32, i64 }>
!cons_alt = !reussir.record<compound "List::ConsAlt" [value] { i32, i64 }>
!nil = !reussir.record<compound "List::Nil" [value] {}>
!list = !reussir.record<variant "List" {!cons, !nil}>
!list_alt = !reussir.record<variant "ListAlt" {!cons, !cons_alt, !nil}>
!cons_clone = !reussir.record<compound "List::ConsClone" [value] { i32, i64 }>
!cons_alt_clone = !reussir.record<compound "List::ConsAltClone" [value] { i32, i64 }>
!nil_clone = !reussir.record<compound "List::NilClone" [value] {}>
!list_clone = !reussir.record<variant "ListClone" {!cons_clone, !cons_alt_clone, !nil_clone}>
!triple_cons = !reussir.record<compound "Triple::Cons" [value] { i32, i64, !list_alt }>
!triple_nil = !reussir.record<compound "Triple::Nil" [value] {}>
!triple = !reussir.record<variant "Triple" {!triple_cons, !triple_nil}>
!triple_clone_cons = !reussir.record<compound "Triple::ConsClone" [value] { i32, i64, !list }>
!triple_clone_nil = !reussir.record<compound "Triple::NilClone" [value] {}>
!triple_clone = !reussir.record<variant "TripleClone" {!triple_clone_cons, !triple_clone_nil}>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>> } {
  func.func @mk_compound(%a: i32, %b: i64) -> !reussir.rc<!cons> {
    %c = reussir.record.compound(%a, %b : i32, i64) : !cons
    %rc = reussir.rc.create value(%c : !cons) : !reussir.rc<!cons>
    return %rc : !reussir.rc<!cons>
  }

  func.func @mk_variant(%a: i32, %b: i64) -> !reussir.rc<!list> {
    %c = reussir.record.compound(%a, %b : i32, i64) : !cons
    %v = reussir.record.variant [0] (%c : !cons) : !list
    %rc = reussir.rc.create value(%v : !list) : !reussir.rc<!list>
    return %rc : !reussir.rc<!list>
  }

  func.func @reuse_compound(%rc: !reussir.rc<!cons>) -> !reussir.rc<!cons> {
    %borrow = reussir.rc.borrow (%rc : !reussir.rc<!cons>) : !reussir.ref<!cons>
    %field0_ref = reussir.ref.project (%borrow : !reussir.ref<!cons>) [0] : !reussir.ref<i32>
    %field0 = reussir.ref.load (%field0_ref : !reussir.ref<i32>) : i32
    %field1_ref = reussir.ref.project (%borrow : !reussir.ref<!cons>) [1] : !reussir.ref<i64>
    %field1 = reussir.ref.load (%field1_ref : !reussir.ref<i64>) : i64
    %c = reussir.record.compound(%field0, %field1 : i32, i64) : !cons
    %token0 = reussir.rc.reinterpret (%rc : !reussir.rc<!cons>) : !reussir.token<align: 8, size: 24>
    %token = reussir.token.launder (%token0 : !reussir.token<align: 8, size: 24>) : !reussir.token<align: 8, size: 24>
    %new = reussir.rc.create value(%c : !cons) token(%token : !reussir.token<align: 8, size: 24>) skip_rc : !reussir.rc<!cons>
    return %new : !reussir.rc<!cons>
  }

  func.func @reuse_variant(%rc: !reussir.rc<!list>) -> !reussir.rc<!list> {
    %borrow = reussir.rc.borrow (%rc : !reussir.rc<!list>) : !reussir.ref<!list>
    %cons_ref = reussir.record.coerce [0] (%borrow : !reussir.ref<!list>) : !reussir.ref<!cons>
    %field0_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons>) [0] : !reussir.ref<i32>
    %field0 = reussir.ref.load (%field0_ref : !reussir.ref<i32>) : i32
    %field1_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons>) [1] : !reussir.ref<i64>
    %field1 = reussir.ref.load (%field1_ref : !reussir.ref<i64>) : i64
    %c = reussir.record.compound(%field0, %field1 : i32, i64) : !cons
    %v = reussir.record.variant [0] (%c : !cons) : !list
    %token0 = reussir.rc.reinterpret (%rc : !reussir.rc<!list>) : !reussir.token<align: 8, size: 32>
    %token = reussir.token.launder (%token0 : !reussir.token<align: 8, size: 32>) : !reussir.token<align: 8, size: 32>
    %new = reussir.rc.create value(%v : !list) token(%token : !reussir.token<align: 8, size: 32>) skip_rc : !reussir.rc<!list>
    return %new : !reussir.rc<!list>
  }

  func.func @reuse_variant_other_tag(%rc: !reussir.rc<!list_alt>) -> !reussir.rc<!list_alt> {
    %borrow = reussir.rc.borrow (%rc : !reussir.rc<!list_alt>) : !reussir.ref<!list_alt>
    %cons_ref = reussir.record.coerce [1] (%borrow : !reussir.ref<!list_alt>) : !reussir.ref<!cons_alt>
    %field0_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons_alt>) [0] : !reussir.ref<i32>
    %field0 = reussir.ref.load (%field0_ref : !reussir.ref<i32>) : i32
    %field1_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons_alt>) [1] : !reussir.ref<i64>
    %field1 = reussir.ref.load (%field1_ref : !reussir.ref<i64>) : i64
    %c = reussir.record.compound(%field0, %field1 : i32, i64) : !cons
    %v = reussir.record.variant [0] (%c : !cons) : !list_alt
    %token0 = reussir.rc.reinterpret (%rc : !reussir.rc<!list_alt>) : !reussir.token<align: 8, size: 32>
    %token = reussir.token.launder (%token0 : !reussir.token<align: 8, size: 32>) : !reussir.token<align: 8, size: 32>
    %new = reussir.rc.create value(%v : !list_alt) token(%token : !reussir.token<align: 8, size: 32>) skip_rc : !reussir.rc<!list_alt>
    return %new : !reussir.rc<!list_alt>
  }

  func.func @reuse_variant_other_type(%rc: !reussir.rc<!list_clone>) -> !reussir.rc<!list_alt> {
    %borrow = reussir.rc.borrow (%rc : !reussir.rc<!list_clone>) : !reussir.ref<!list_clone>
    %cons_ref = reussir.record.coerce [1] (%borrow : !reussir.ref<!list_clone>) : !reussir.ref<!cons_alt_clone>
    %field0_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons_alt_clone>) [0] : !reussir.ref<i32>
    %field0 = reussir.ref.load (%field0_ref : !reussir.ref<i32>) : i32
    %field1_ref = reussir.ref.project (%cons_ref : !reussir.ref<!cons_alt_clone>) [1] : !reussir.ref<i64>
    %field1 = reussir.ref.load (%field1_ref : !reussir.ref<i64>) : i64
    %c = reussir.record.compound(%field0, %field1 : i32, i64) : !cons
    %v = reussir.record.variant [0] (%c : !cons) : !list_alt
    %token0 = reussir.rc.reinterpret (%rc : !reussir.rc<!list_clone>) : !reussir.token<align: 8, size: 32>
    %token = reussir.token.launder (%token0 : !reussir.token<align: 8, size: 32>) : !reussir.token<align: 8, size: 32>
    %new = reussir.rc.create value(%v : !list_alt) token(%token : !reussir.token<align: 8, size: 32>) skip_rc : !reussir.rc<!list_alt>
    return %new : !reussir.rc<!list_alt>
  }

  func.func @reuse_variant_prefix(%rc: !reussir.rc<!triple_clone>, %tail: !reussir.rc<!list_alt>) -> !reussir.rc<!triple> {
    %borrow = reussir.rc.borrow (%rc : !reussir.rc<!triple_clone>) : !reussir.ref<!triple_clone>
    %node_ref = reussir.record.coerce [0] (%borrow : !reussir.ref<!triple_clone>) : !reussir.ref<!triple_clone_cons>
    %field0_ref = reussir.ref.project (%node_ref : !reussir.ref<!triple_clone_cons>) [0] : !reussir.ref<i32>
    %field0 = reussir.ref.load (%field0_ref : !reussir.ref<i32>) : i32
    %field1_ref = reussir.ref.project (%node_ref : !reussir.ref<!triple_clone_cons>) [1] : !reussir.ref<i64>
    %field1 = reussir.ref.load (%field1_ref : !reussir.ref<i64>) : i64
    %c = reussir.record.compound(%field0, %field1, %tail : i32, i64, !reussir.rc<!list_alt>) : !triple_cons
    %v = reussir.record.variant [0] (%c : !triple_cons) : !triple
    %token0 = reussir.rc.reinterpret (%rc : !reussir.rc<!triple_clone>) : !reussir.token<align: 8, size: 40>
    %token = reussir.token.launder (%token0 : !reussir.token<align: 8, size: 40>) : !reussir.token<align: 8, size: 40>
    %new = reussir.rc.create value(%v : !triple) token(%token : !reussir.token<align: 8, size: 40>) skip_rc : !reussir.rc<!triple>
    return %new : !reussir.rc<!triple>
  }
}

// CHECK-LABEL: define ptr @mk_compound
// CHECK-NOT: alloca %"List::Cons"
// CHECK: %[[ALLOC:.*]] = call ptr @__reussir_allocate(i64 8, i64 24)
// CHECK: %[[COUNT:.*]] = getelementptr { i64, %"List::Cons" }, ptr %[[ALLOC]], i32 0, i32 0
// CHECK: store i64 1, ptr %[[COUNT]], align 8
// CHECK: %[[PAYLOAD:.*]] = getelementptr { i64, %"List::Cons" }, ptr %[[ALLOC]], i32 0, i32 1
// CHECK: %[[FIELD0:.*]] = getelementptr %"List::Cons", ptr %[[PAYLOAD]], i32 0, i32 0
// CHECK: store i32 %0, ptr %[[FIELD0]], align 4, !invariant.group
// CHECK: %[[FIELD1:.*]] = getelementptr %"List::Cons", ptr %[[PAYLOAD]], i32 0, i32 1
// CHECK: store i64 %1, ptr %[[FIELD1]], align 8, !invariant.group

// CHECK-LABEL: define ptr @mk_variant
// CHECK-NOT: alloca %List
// CHECK-NOT: alloca %"List::Cons"
// CHECK: %[[ALLOC:.*]] = call ptr @__reussir_allocate(i64 8, i64 32)
// CHECK: %[[COUNT:.*]] = getelementptr { i64, %List }, ptr %[[ALLOC]], i32 0, i32 0
// CHECK: store i64 1, ptr %[[COUNT]], align 8
// CHECK: %[[VARIANT:.*]] = getelementptr { i64, %List }, ptr %[[ALLOC]], i32 0, i32 1
// CHECK: %[[TAGPTR:.*]] = getelementptr %List, ptr %[[VARIANT]], i32 0, i32 0
// CHECK: store i8 0, ptr %[[TAGPTR]], align 1
// CHECK: %[[PAYLOAD:.*]] = getelementptr %List, ptr %[[VARIANT]], i32 0, i32 1
// CHECK: %[[FIELD0:.*]] = getelementptr %"List::Cons", ptr %[[PAYLOAD]], i32 0, i32 0
// CHECK: store i32 %0, ptr %[[FIELD0]], align 4, !invariant.group
// CHECK: %[[FIELD1:.*]] = getelementptr %"List::Cons", ptr %[[PAYLOAD]], i32 0, i32 1
// CHECK: store i64 %1, ptr %[[FIELD1]], align 8, !invariant.group

// FUSION-LABEL: func.func @reuse_compound
// FUSION: reussir.rc.create_compound
// FUSION-SAME: skipFields = array<i64: 0, 1>

// FUSION-LABEL: func.func @reuse_variant
// FUSION: "reussir.rc.create_variant"
// FUSION-SAME: skipFields = array<i64: 0, 1>

// FUSION-LABEL: func.func @reuse_variant_other_tag
// FUSION: "reussir.rc.create_variant"
// FUSION-SAME: skipFields = array<i64: 0, 1>

// FUSION-LABEL: func.func @reuse_variant_other_type
// FUSION: "reussir.rc.create_variant"
// FUSION-SAME: skipFields = array<i64: 0, 1>

// FUSION-LABEL: func.func @reuse_variant_prefix
// FUSION: "reussir.rc.create_variant"
// FUSION-SAME: skipFields = array<i64: 0, 1>

// CHECK-LABEL: define ptr @reuse_compound
// CHECK: call void @llvm.assume
// CHECK: %[[PAYLOAD:.*]] = getelementptr { i64, %"List::Cons" }, ptr %{{.*}}, i32 0, i32 1
// CHECK-NOT: getelementptr %"List::Cons", ptr %[[PAYLOAD]]
// CHECK: ret ptr

// CHECK-LABEL: define ptr @reuse_variant
// CHECK: call void @llvm.assume
// CHECK: %[[VARIANT:.*]] = getelementptr { i64, %List }, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[TAGPTR:.*]] = getelementptr %List, ptr %[[VARIANT]], i32 0, i32 0
// CHECK: store i8 0, ptr %[[TAGPTR]], align 1
// CHECK: %[[PAYLOAD:.*]] = getelementptr %List, ptr %[[VARIANT]], i32 0, i32 1
// CHECK-NOT: getelementptr %"List::Cons", ptr %[[PAYLOAD]]
// CHECK: ret ptr

// CHECK-LABEL: define ptr @reuse_variant_other_tag
// CHECK: call void @llvm.assume
// CHECK: %[[VARIANT2:.*]] = getelementptr { i64, %ListAlt }, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[TAGPTR2:.*]] = getelementptr %ListAlt, ptr %[[VARIANT2]], i32 0, i32 0
// CHECK: store i8 0, ptr %[[TAGPTR2]], align 1
// CHECK: %[[PAYLOAD2:.*]] = getelementptr %ListAlt, ptr %[[VARIANT2]], i32 0, i32 1
// CHECK-NOT: getelementptr %"List::Cons", ptr %[[PAYLOAD2]]
// CHECK: ret ptr

// CHECK-LABEL: define ptr @reuse_variant_other_type
// CHECK: call void @llvm.assume
// CHECK: %[[VARIANT3:.*]] = getelementptr { i64, %ListAlt }, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[TAGPTR3:.*]] = getelementptr %ListAlt, ptr %[[VARIANT3]], i32 0, i32 0
// CHECK: store i8 0, ptr %[[TAGPTR3]], align 1
// CHECK: %[[PAYLOAD3:.*]] = getelementptr %ListAlt, ptr %[[VARIANT3]], i32 0, i32 1
// CHECK-NOT: getelementptr %"List::Cons", ptr %[[PAYLOAD3]]
// CHECK: ret ptr

// CHECK-LABEL: define ptr @reuse_variant_prefix
// CHECK: call void @llvm.assume
// CHECK: %[[VARIANT4:.*]] = getelementptr { i64, %Triple }, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[TAGPTR4:.*]] = getelementptr %Triple, ptr %[[VARIANT4]], i32 0, i32 0
// CHECK: store i8 0, ptr %[[TAGPTR4]], align 1
// CHECK: %[[PAYLOAD4:.*]] = getelementptr %Triple, ptr %[[VARIANT4]], i32 0, i32 1
// CHECK-NOT: getelementptr %"Triple::Cons", ptr %[[PAYLOAD4]], i32 0, i32 0
// CHECK-NOT: getelementptr %"Triple::Cons", ptr %[[PAYLOAD4]], i32 0, i32 1
// CHECK: %[[FIELD24:.*]] = getelementptr %"Triple::Cons", ptr %[[PAYLOAD4]], i32 0, i32 2
// CHECK: store ptr %1, ptr %[[FIELD24]], align 8, !invariant.group
// CHECK: ret ptr
