// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-closure-outlining,convert-scf-to-cf,reussir-lowering-basic-ops{closure-wpd=1})' \
// RUN: | %FileCheck %s
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-closure-outlining,convert-scf-to-cf,reussir-lowering-basic-ops)' \
// RUN: | %FileCheck %s --check-prefix=CHECK-OFF

// Closure WPD type-id tiers (docs/design/closure-wpd.md). A vtable for a
// closure of original signature (i32, i32) -> i32 carries one id per suffix
// of that signature — family root `() -> i32`, `(i32) -> i32`, the full
// signature — plus its own uid tier. Ids follow the project's v0 mangling
// scheme: the closure-type production `FK…E…` over one content-addressed
// `C<digest>` nominal per type, so the ids are structural — both parameters
// are i32 and the same segment appears in every tier ([[ARG]]), and the
// return segment ([[RET]]) roots the family. The uid tier is the path
// `<vtable>::wpd` nested under the vtable's own v0 symbol. Without the
// `closure-wpd` option nothing is recorded.
//
// This test is also the ABI canary for the hierarchy itself: the outlined
// `evaluate` must take exactly ONE argument (the rc'd box) and read every
// closure argument from the payload. If evaluation ever passes remaining
// arguments in registers, a vtable can no longer stand in for its suffix
// types and the tiered ids above become UNSOUND — regenerate the scheme
// before appeasing this test.

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @make() -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%a : i32, %b : i32):
          %add = arith.addi %a, %b : i32
          reussir.closure.yield %add : i32
      }
    }
    return %closure : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>
  }
}

// CHECK: module @test attributes
// CHECK-SAME: reussir.wpd.vtables = {
// CHECK-SAME: {{"?}}_R[[VTPATH:[0-9a-zA-Z$_]+]]{{"?}} = [
// CHECK-SAME: "_RFK17ReussirClosureWpdE[[RET:C[0-9a-zA-Z_]+]]",
// CHECK-SAME: "_RFK17ReussirClosureWpd[[ARG:C[0-9a-zA-Z_]+]]E[[RET]]",
// CHECK-SAME: "_RFK17ReussirClosureWpd[[ARG]][[ARG]]E[[RET]]",
// CHECK-SAME: "_RNv[[VTPATH]]3wpd"]

// The ABI canary: exactly one parameter — the rc'd closure box, no comma.
// CHECK: func.func private @[[EVAL:[^(]*8evaluate]](%{{[a-z0-9]+}}: !reussir.rc<!reussir.closure_box<i32, i32> unspecified> {{{[^}]*}}}) -> i32

// CHECK-OFF-NOT: reussir.wpd.vtables
