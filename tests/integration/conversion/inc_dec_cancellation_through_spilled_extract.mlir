// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// A by-value result struct is consumed by extracting its rc member for
// further use while the struct itself is spilled so its release can be
// decomposed field by field:
//
//   %m = record.extract %s[0] ; rc.inc(%m)            // retain the member
//   %f = ref.load(ref.project(ref.spilled %s, [0]))   // the release's view
//   rc.dec(%f)                                        // release the member
//
// `%m` and `%f` read the same stored pointer, so the pair is borrow
// semantics in disguise and must cancel. Alias analysis proves the two
// reads MustAlias (same record value, same field index); without that the
// surviving decrement observes a shared count at runtime, never yields a
// reuse token, and poisons token-donor selection with a candidate that is
// null on every execution — an allocation plus a free of the genuinely
// reusable block per call on result-struct shapes like std's
// `Changed`/`Inserted`.

!box = !reussir.record<compound "Box" {i64}>
!pair = !reussir.record<compound "Pair" [value] {!box, i1}>
!two = !reussir.record<compound "Two" [value] {!box, !box}>

module {
  // CHECK-LABEL: @cancel_extract_spill_roundtrip
  func.func @cancel_extract_spill_roundtrip(%s: !pair) -> !reussir.rc<!box> {
    // CHECK-NOT: reussir.rc.inc
    // CHECK-NOT: reussir.rc.dec
    %m = reussir.record.extract(%s : !pair)[0] : !reussir.rc<!box>
    reussir.rc.inc (%m : !reussir.rc<!box>)
    %spill = reussir.ref.spilled(%s : !pair) : !reussir.ref<!pair>
    %slot = reussir.ref.project(%spill : !reussir.ref<!pair>) [0] : !reussir.ref<!reussir.rc<!box>>
    %f = reussir.ref.load(%slot : !reussir.ref<!reussir.rc<!box>>) : !reussir.rc<!box>
    reussir.rc.dec (%f : !reussir.rc<!box>)
    return %m : !reussir.rc<!box>
  }

  // Distinct fields of the same spill hold distinct pointers: the retain of
  // field 0 must not cancel against the release of field 1.
  // CHECK-LABEL: @keep_distinct_fields
  func.func @keep_distinct_fields(%s: !two) -> !reussir.rc<!box> {
    // CHECK: reussir.rc.inc
    // CHECK: reussir.rc.dec
    %m = reussir.record.extract(%s : !two)[0] : !reussir.rc<!box>
    reussir.rc.inc (%m : !reussir.rc<!box>)
    %spill = reussir.ref.spilled(%s : !two) : !reussir.ref<!two>
    %slot = reussir.ref.project(%spill : !reussir.ref<!two>) [1] : !reussir.ref<!reussir.rc<!box>>
    %f = reussir.ref.load(%slot : !reussir.ref<!reussir.rc<!box>>) : !reussir.rc<!box>
    reussir.rc.dec (%f : !reussir.rc<!box>)
    return %m : !reussir.rc<!box>
  }

  // Spills of different record values stay unrelated even at equal indices.
  // CHECK-LABEL: @keep_distinct_records
  func.func @keep_distinct_records(%s: !pair, %t: !pair) -> !reussir.rc<!box> {
    // CHECK: reussir.rc.inc
    // CHECK: reussir.rc.dec
    %m = reussir.record.extract(%s : !pair)[0] : !reussir.rc<!box>
    reussir.rc.inc (%m : !reussir.rc<!box>)
    %spill = reussir.ref.spilled(%t : !pair) : !reussir.ref<!pair>
    %slot = reussir.ref.project(%spill : !reussir.ref<!pair>) [0] : !reussir.ref<!reussir.rc<!box>>
    %f = reussir.ref.load(%slot : !reussir.ref<!reussir.rc<!box>>) : !reussir.rc<!box>
    reussir.rc.dec (%f : !reussir.rc<!box>)
    return %m : !reussir.rc<!box>
  }
}
