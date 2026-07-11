// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// Regression for a copy-on-write miscompile (issue #344): an inc/dec pair
// around a count-*observing* op — `reussir.array.with_unique_view`,
// `reussir.closure.uniqify`, or a standalone `reussir.rc.is_unique` — is not
// redundant: the extra count is exactly what makes the observation see the
// value as shared and take the defensive-clone path. Cancelling across it
// turned `let b = a.set(…)` with `a` still live into an in-place mutation of
// `a`. The pass must keep the pair when the observing op's operand may alias
// the incremented pointer, and may still cancel across observations of
// *unrelated* values.

!arr = !reussir.array<4 x f64>
!rc_arr = !reussir.rc<!arr>
!clo = !reussir.closure<(f64) -> f64>
!rc_clo = !reussir.rc<!clo>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK-LABEL: @cow_set
  // CHECK: reussir.rc.inc
  // CHECK: reussir.array.with_unique_view
  // CHECK: reussir.rc.dec
  func.func @cow_set(%a: !rc_arr) -> !rc_arr {
    reussir.rc.inc (%a : !rc_arr)
    %b = reussir.array.with_unique_view (%a : !rc_arr) -> !rc_arr {
      ^bb0(%view: memref<4xf64>):
        %c0 = arith.constant 0 : index
        %v = arith.constant 9.9e+01 : f64
        memref.store %v, %view[%c0] : memref<4xf64>
        reussir.scf.yield
    }
    %tk = reussir.rc.dec (%a : !rc_arr) : !reussir.nullable<!reussir.token<align: 8, size: 40>>
    return %b : !rc_arr
  }

  // CHECK-LABEL: @observed_unique
  // CHECK: reussir.rc.inc
  // CHECK: reussir.rc.is_unique
  // CHECK: reussir.rc.dec
  func.func @observed_unique(%a: !rc_arr) -> i1 {
    reussir.rc.inc (%a : !rc_arr)
    %u = reussir.rc.is_unique (%a : !rc_arr) : i1
    %tk = reussir.rc.dec (%a : !rc_arr) : !reussir.nullable<!reussir.token<align: 8, size: 40>>
    return %u : i1
  }

  // CHECK-LABEL: @uniqify_blocks
  // CHECK: reussir.rc.inc
  // CHECK: reussir.closure.uniqify
  // CHECK: reussir.rc.dec
  func.func @uniqify_blocks(%f: !rc_clo) -> !rc_clo {
    reussir.rc.inc (%f : !rc_clo)
    %u = reussir.closure.uniqify (%f : !rc_clo) : !rc_clo
    reussir.rc.dec (%f : !rc_clo)
    return %u : !rc_clo
  }

  // An adjacent inc/dec pair with nothing observing in between still cancels
  // (the barrier is the observing op, not the array type).
  // CHECK-LABEL: @plain_pair_still_cancels
  // CHECK-NOT: reussir.rc.inc
  // CHECK-NOT: reussir.rc.dec
  func.func @plain_pair_still_cancels(%a: !rc_arr) {
    reussir.rc.inc (%a : !rc_arr)
    %tk = reussir.rc.dec (%a : !rc_arr) : !reussir.nullable<!reussir.token<align: 8, size: 40>>
    return
  }
}
