// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// Regression (#290): `reussir-inc-dec-cancellation` must not cancel an
// `rc.inc` against a later aliased `rc.dec` when a `reussir.ref.drop` of a
// managed-content value sits between them. The drop releases every rc
// reachable *through* the referenced value — and the incremented rc may be
// stored inside it (a spilled `[value]` enum whose payload was just loaded),
// which no operand alias query can prove disjoint. Cancelling across the drop
// lets it take the count to zero while the payload is still live: this is the
// `[value]`-enum-with-shared-payload use-after-free (match loads the payload
// rc, drops the spilled scrutinee, then reads through the freed pointer).

!box = !reussir.record<compound "Box" {i64}>
!just = !reussir.record<compound "Just" [value] {!box}>
!nothing = !reussir.record<compound "Nothing" [value] {}>
!maybebox = !reussir.record<variant "MaybeBox" [value] {!nothing, !just}>
!plain = !reussir.record<compound "Plain" [value] {i64, i1}>

module {
  func.func @retain_across_managed_drop(%v: !maybebox, %rc: !reussir.rc<!box>) {
    // The dropped variant may hold the very rc being retained, so this pair
    // must survive: cancelling it lets the drop free the box while the
    // caller's reference is still live.
    reussir.rc.inc (%rc : !reussir.rc<!box>)
    %spill = reussir.ref.spilled(%v : !maybebox) : !reussir.ref<!maybebox>
    reussir.ref.drop(%spill : !reussir.ref<!maybebox>)
    reussir.rc.dec (%rc : !reussir.rc<!box>)
    return
  }

  func.func @retain_across_region_cleanup(%r: !reussir.region, %rc: !reussir.rc<!box>) {
    // Region cleanup releases the region's objects wholesale; it is just as
    // opaque to operand alias queries as the drop above.
    reussir.rc.inc (%rc : !reussir.rc<!box>)
    reussir.region.cleanup(%r : !reussir.region)
    reussir.rc.dec (%rc : !reussir.rc<!box>)
    return
  }

  func.func @cancel_across_trivial_drop(%v: !plain, %rc: !reussir.rc<!box>) {
    // A drop of a trivially-copyable value expands to nothing — it cannot
    // release anything, so it stays transparent and the pair cancels.
    reussir.rc.inc (%rc : !reussir.rc<!box>)
    %spill = reussir.ref.spilled(%v : !plain) : !reussir.ref<!plain>
    reussir.ref.drop(%spill : !reussir.ref<!plain>)
    reussir.rc.dec (%rc : !reussir.rc<!box>)
    return
  }
}

// CHECK-LABEL: func.func @retain_across_managed_drop
// CHECK:       reussir.rc.inc
// CHECK:       reussir.ref.drop
// CHECK:       reussir.rc.dec

// CHECK-LABEL: func.func @retain_across_region_cleanup
// CHECK:       reussir.rc.inc
// CHECK:       reussir.region.cleanup
// CHECK:       reussir.rc.dec

// CHECK-LABEL: func.func @cancel_across_trivial_drop
// CHECK-NOT:   reussir.rc.inc
// CHECK:       reussir.ref.drop
// CHECK-NOT:   reussir.rc.dec
