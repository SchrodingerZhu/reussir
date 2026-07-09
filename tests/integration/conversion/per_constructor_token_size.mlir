// RUN: %reussir-opt %s --reussir-token-instantiation | %FileCheck %s

// Per-constructor variant box sizing (#325) is the DEFAULT: a fused-header
// variant construction sizes its allocation token for the *constructed arm*
// (`header + arm[k]`) instead of the uniform max-arm width. A variant marked
// `fixed` (the dialect flag behind the source `#[repr(fixed)]`) opts back into
// uniform max-arm sizing. Field offsets are unchanged either way — the payload
// offset comes from the variant-wide alignment — so per-arm sizing only drops
// the trailing padding up to the max arm.
//
// List: Cons {i64, rc} => 8 (header) + 16 = 24; Var {i64} => 8 + 8 = 16.
// Under `fixed` both arms allocate the uniform 24.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Var" [value] {i64}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

!listf = !reussir.record<variant "ListF" fixed {!reussir.record<compound "ListF.Cons" [value] {i64, !reussir.record<variant "ListF">}>, !reussir.record<compound "ListF.Var" [value] {i64}>, !reussir.record<compound "ListF.Nil" [value] {}>}>
!rclistf = !reussir.rc<!listf>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // Default per-constructor sizing: the Cons arm needs the full 24 bytes.
  // CHECK-LABEL: @cons
  // CHECK: reussir.token.alloc : <align : 8, size : 24>
  func.func @cons(%x: i64, %t: !rclist) -> !rclist {
    %0 = "reussir.rc.create_variant"(%x, %t) <{operandSegmentSizes = array<i32: 0, 2, 0, 0>, tag = 0 : index}> : (i64, !rclist) -> !rclist
    return %0 : !rclist
  }
  // Default per-constructor sizing: the leaf arm drops to 16 bytes.
  // CHECK-LABEL: @leaf
  // CHECK: reussir.token.alloc : <align : 8, size : 16>
  func.func @leaf(%x: i64) -> !rclist {
    %0 = "reussir.rc.create_variant"(%x) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, tag = 1 : index}> : (i64) -> !rclist
    return %0 : !rclist
  }
  // `fixed` opt-out: the same leaf arm keeps the uniform 24-byte max-arm size.
  // CHECK-LABEL: @leaf_fixed
  // CHECK: reussir.token.alloc : <align : 8, size : 24>
  func.func @leaf_fixed(%x: i64) -> !rclistf {
    %0 = "reussir.rc.create_variant"(%x) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, tag = 1 : index}> : (i64) -> !rclistf
    return %0 : !rclistf
  }
}
