// Pass twice to make sure the output is stable
// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s
!foo = !reussir.record<compound "foo" {i32, i32}>
!list_cons = !reussir.record<compound "list.cons" {i32, [shared] !reussir.record<variant "list" incomplete>}>
!list_nil = !reussir.record<compound "list.nil" {}>
!list = !reussir.record<variant "list" {!list_cons, !list_nil}>
!tree = !reussir.record<variant "tree" {!reussir.record<compound "tree.branch" {[shared] !reussir.record<variant "tree">, i32, [shared] !reussir.record<variant "tree">}>, !reussir.record<compound "tree.leaf" {}>}>
module @test {
  // CHECK: func.func private @foo() -> !reussir.record<compound "foo" {i32, i32}>
  func.func private @foo() -> !foo
  // CHECK: func.func private @list() -> !reussir.record<variant "list" {!reussir.record<compound "list.cons" {i32, [shared] !reussir.record<variant "list">}>, !reussir.record<compound "list.nil" {}>}>
  func.func private @list() -> !list
  // CHECK: func.func private @tree() -> !reussir.record<variant "tree" {!reussir.record<compound "tree.branch" {[shared] !reussir.record<variant "tree">, i32, [shared] !reussir.record<variant "tree">}>, !reussir.record<compound "tree.leaf" {}>}
  func.func private @tree() -> !tree
}
