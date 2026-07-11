// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-closure-outlining,convert-scf-to-cf,reussir-lowering-basic-ops{closure-wpd=1})' \
// RUN: | %FileCheck %s
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-closure-outlining,convert-scf-to-cf,reussir-lowering-basic-ops)' \
// RUN: | %FileCheck %s --check-prefix=CHECK-OFF

// Closure WPD type ids (docs/design/closure-wpd.md). Every outlined closure
// vtable carries exactly ONE id, keyed by the closure's RETURN type:
// apply/uniqify/clone never change a vtable — only the static type — so the
// return type is the one component every value the vtable backs agrees on.
// Two closures with different signatures but the same return type therefore
// share their id ([[ID]] below), which is precisely what lets a call site
// that only knows a weaker static view still assert something every
// reachable vtable carries. The module is marked (`reussir.wpd`) so the
// dispatch conversion patterns emit the call-site assertions. Without the
// `closure-wpd` option nothing is stamped.
//
// This test is also the ABI canary the id scheme rests on: the outlined
// `evaluate` must take exactly ONE argument (the rc'd box) and read every
// closure argument from the payload. If evaluation ever passes remaining
// arguments in registers, the slot ABI depends on more than the return type
// and per-return-type ids become UNSOUND — rework the scheme before
// appeasing this test.

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

  func.func @make_other() -> !reussir.rc<!reussir.closure<(i64) -> i32>> {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          %trunc = arith.trunci %a : i64 to i32
          reussir.closure.yield %trunc : i32
      }
    }
    return %closure : !reussir.rc<!reussir.closure<(i64) -> i32>>
  }
}

// CHECK: module @test attributes
// CHECK-SAME: reussir.wpd

// The ABI canary: exactly one parameter — the rc'd closure box, no comma.
// CHECK: func.func private @{{[^(]*}}8evaluate{{[^(]*}}(%{{[a-z0-9]+}}: !reussir.rc<!reussir.closure_box<i32, i32> unspecified> {{{[^}]*}}}) -> i32

// Both vtables — different signatures, same return type — carry the SAME
// return-type family id. (The vtable ops print at the end of the module.)
// CHECK: reussir.closure.vtable
// CHECK-SAME: reussir.wpd.type_id = "reussir.closure.wpd.[[ID:[0-9A-Za-z]+]]"
// CHECK: reussir.closure.vtable
// CHECK-SAME: reussir.wpd.type_id = "reussir.closure.wpd.[[ID]]"

// CHECK-OFF-NOT: reussir.wpd
