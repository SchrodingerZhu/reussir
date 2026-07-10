// RUN: %reussir-opt %s | %FileCheck %s

// The `fixed` flag (source `#[repr(fixed)]`) drives the rc.dec token type. A
// non-uniform variant that is NOT fixed decrements to a dynamic `token<?>`
// (per-constructor sizing, the default: the freed size is only known at
// runtime). The same variant marked `fixed` decrements to a STATIC uniform
// max-arm token, because every cell is the max-arm width. Arms:
// Cons {i64, rc} => 16, Var {i64} => 8, Nil {} => 0; header 8 + max arm 16 = 24.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Var" [value] {i64}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

!listf = !reussir.record<variant "ListF" fixed {!reussir.record<compound "ListF.Cons" [value] {i64, !reussir.record<variant "ListF">}>, !reussir.record<compound "ListF.Var" [value] {i64}>, !reussir.record<compound "ListF.Nil" [value] {}>}>
!rclistf = !reussir.rc<!listf>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // Non-fixed, non-uniform variant: dynamic token.
  // CHECK-LABEL: @dec_variable
  // CHECK: reussir.rc.dec{{.*}}: !reussir.nullable<!reussir.token<align : 8, size : ?>>
  func.func @dec_variable(%0: !rclist) -> !reussir.nullable<!reussir.token<align: 8, size: ?>> {
    %t = reussir.rc.dec (%0 : !rclist) : !reussir.nullable<!reussir.token<align: 8, size: ?>>
    return %t : !reussir.nullable<!reussir.token<align: 8, size: ?>>
  }

  // `fixed` variant: static uniform max-arm token (24), never dynamic.
  // CHECK-LABEL: @dec_fixed
  // CHECK: reussir.rc.dec{{.*}}: !reussir.nullable<!reussir.token<align : 8, size : 24>>
  func.func @dec_fixed(%0: !rclistf) -> !reussir.nullable<!reussir.token<align: 8, size: 24>> {
    %t = reussir.rc.dec (%0 : !rclistf) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    return %t : !reussir.nullable<!reussir.token<align: 8, size: 24>>
  }
}
