// RUN: %reussir-opt %s --reussir-token-instantiation | %FileCheck %s

// Per-constructor box sizing (#325): with the module opted in via
// `reussir.per_constructor_box_sizing`, a fused-header variant construction
// sizes its allocation token for the *constructed arm* (`header + arm[k]`)
// instead of the uniform max-arm width. Field offsets are unchanged — the
// payload offset comes from the variant-wide alignment — so only the
// trailing padding up to the max arm is dropped. Without the attribute
// every existing test pins the uniform behavior.
//
// List: Cons {i64, rc} => 8 (header) + 16 = 24; Var {i64} => 8 + 8 = 16;
// under the uniform contract both would be 24.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Var" [value] {i64}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

module @test attributes { reussir.per_constructor_box_sizing, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK-LABEL: @cons
  // CHECK: reussir.token.alloc : <align : 8, size : 24>
  func.func @cons(%x: i64, %t: !rclist) -> !rclist {
    %0 = "reussir.rc.create_variant"(%x, %t) <{operandSegmentSizes = array<i32: 0, 2, 0, 0>, tag = 0 : index}> : (i64, !rclist) -> !rclist
    return %0 : !rclist
  }
  // CHECK-LABEL: @leaf
  // CHECK: reussir.token.alloc : <align : 8, size : 16>
  func.func @leaf(%x: i64) -> !rclist {
    %0 = "reussir.rc.create_variant"(%x) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, tag = 1 : index}> : (i64) -> !rclist
    return %0 : !rclist
  }
}
