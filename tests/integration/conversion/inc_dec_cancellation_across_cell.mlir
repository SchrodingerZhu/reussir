// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// `reussir-inc-dec-cancellation` must not cancel an `rc.inc` against a later
// aliased `rc.dec` across cell mutation ops. `cell.set` implicitly releases
// the old cell element — the released rc is reachable only *through* the
// cell, so no operand alias query can prove it disjoint from the incremented
// pointer. A `cell.rmw` body is arbitrary nested code the same-block scan
// never visits. Both are opaque, exactly like an unexpanded `ref.drop`.

!box = !reussir.record<compound "Box" {i64}>
!rc_box = !reussir.rc<!box>
!cell = !reussir.rc<!reussir.cell<!rc_box>>
!excl_cell = !reussir.rc<!reussir.cell<!rc_box exclusive>>
!scalar_cell = !reussir.rc<!reussir.cell<i64>>

module {
  func.func @retain_across_cell_set(%v: !rc_box, %rc: !rc_box, %cell: !cell) {
    // The cell may hold the very rc being retained; the set's implicit drop
    // of the old element must keep this pair alive.
    reussir.rc.inc (%rc : !rc_box)
    reussir.cell.set(%v : !rc_box, %cell : !cell)
    reussir.rc.dec (%rc : !rc_box)
    return
  }

  func.func @retain_across_cell_rmw(%v: !rc_box, %rc: !rc_box, %cell: !excl_cell) {
    reussir.rc.inc (%rc : !rc_box)
    reussir.cell.rmw(%cell : !excl_cell) {
      ^bb0(%old: !rc_box):
        reussir.rc.dec (%old : !rc_box)
        reussir.cell.yield(%v : !rc_box)
    }
    reussir.rc.dec (%rc : !rc_box)
    return
  }

  func.func @cancel_across_trivial_cell_set(%v: i64, %rc: !rc_box, %cell: !scalar_cell) {
    // A set of a trivially-copyable element expands to a plain store — it
    // cannot release anything, so it stays transparent and the pair cancels.
    reussir.rc.inc (%rc : !rc_box)
    reussir.cell.set(%v : i64, %cell : !scalar_cell)
    reussir.rc.dec (%rc : !rc_box)
    return
  }
}

// CHECK-LABEL: func.func @retain_across_cell_set
// CHECK:       reussir.rc.inc
// CHECK:       reussir.cell.set
// CHECK:       reussir.rc.dec

// CHECK-LABEL: func.func @retain_across_cell_rmw
// CHECK:       reussir.rc.inc
// CHECK:       reussir.cell.rmw
// CHECK:       reussir.rc.dec

// CHECK-LABEL: func.func @cancel_across_trivial_cell_set
// CHECK-NOT:   reussir.rc.inc
// CHECK:       reussir.cell.set
// CHECK-NOT:   reussir.rc.dec
