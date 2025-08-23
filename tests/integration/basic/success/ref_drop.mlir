// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s
// Test basic reference drop operations
!test = !reussir.record<compound "Test" { [shared] i64, [shared] i64, [field] i64 }>
!inner = !reussir.record<variant "Inner" { [value] i64, [shared] i64 }>
!tree_ = !reussir.record<variant "Tree" incomplete>
!branch = !reussir.record<compound "Tree::Branch" { [shared] !tree_, [value] !inner, [shared] !tree_ }>
!leaf = !reussir.record<compound "Tree::Leaf" { }>
!tree = !reussir.record<variant "Tree" { !branch, !leaf }>
module {
  // CHECK: func.func @drop_basic(%arg0: !reussir.ref<i32>)
  func.func @drop_basic(%ref : !reussir.ref<i32>) {
    // CHECK: reussir.ref.drop(%arg0 : !reussir.ref<i32>)
    reussir.ref.drop (%ref : !reussir.ref<i32>)
    return
  }

  func.func @drop_rc(%ref : !reussir.ref<!reussir.rc<i64>>) {
    // CHECK: reussir.ref.drop(%arg0 : !reussir.ref<!reussir.rc<i64>>)
    reussir.ref.drop (%ref : !reussir.ref<!reussir.rc<i64>>)
    return
  }

  func.func @drop_compound(%ref : !reussir.ref<!test>) {
    reussir.ref.drop (%ref : !reussir.ref<!test>)
    return
  }
  func.func @drop_variant(%ref : !reussir.ref<!inner>) {
    reussir.ref.drop (%ref : !reussir.ref<!inner>)
    return
  }
  func.func @drop_tree(%ref : !reussir.ref<!tree>) {
    reussir.ref.drop (%ref : !reussir.ref<!tree>)
    return
  }
}
