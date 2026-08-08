// RUN: %reussir-opt %s --reussir-default-inliner --reussir-partial-move \
// RUN:   | %FileCheck %s --check-prefix=LATE-ONLY
// RUN: %reussir-opt %s --reussir-early-partial-move \
// RUN:   --reussir-default-inliner --reussir-partial-move \
// RUN:   | %FileCheck %s --check-prefix=SPLIT

// A natural match-arm spelling constructs a fresh owned tail after a recursive
// call even though every input is already available. The early phase moves the
// closed compound/variant/create chain before the call, matching the source
// ordering that measured faster without changing RC operation counts. The late
// phase keeps its existing release/rebuild responsibilities.

!leaf = !reussir.record<compound "Leaf" [value] {i64}>
!elem = !reussir.record<variant "Elem" {!leaf}>
!rc_elem = !reussir.rc<!elem>
!pair = !reussir.record<compound "Pair" [value] {!elem, !elem}>
!digit = !reussir.record<variant "Digit" {!pair}>
!rc_digit = !reussir.rc<!digit>
!holder = !reussir.record<compound "Holder" {!elem}>
!rc_holder = !reussir.rc<!holder>
!dependent_pair = !reussir.record<compound "DependentPair" [value] {!elem, !digit}>
!dependent = !reussir.record<variant "Dependent" {!dependent_pair}>
!rc_dependent = !reussir.rc<!dependent>

func.func private @recurse(!rc_digit, !rc_elem) -> !rc_digit

// LATE-ONLY-LABEL: func.func @natural(
// LATE-ONLY: %[[CALL:.+]] = call @recurse
// LATE-ONLY-NEXT: %[[PAIR:.+]] = reussir.record.compound
// LATE-ONLY-NEXT: %[[VARIANT:.+]] = reussir.record.variant
// LATE-ONLY-NEXT: %[[TAIL:.+]] = reussir.rc.create
// LATE-ONLY: return %[[TAIL]]

// SPLIT-LABEL: func.func @natural(
// SPLIT: %[[PAIR:.+]] = reussir.record.compound
// SPLIT-NEXT: %[[VARIANT:.+]] = reussir.record.variant
// SPLIT-NEXT: %[[TAIL:.+]] = reussir.rc.create
// SPLIT-NEXT: %[[CALL:.+]] = call @recurse
// SPLIT: return %[[TAIL]]
func.func @natural(%holder: !rc_holder, %other: !rc_elem,
                   %middle: !rc_digit, %node: !rc_elem) -> !rc_digit {
  %ref = reussir.rc.borrow (%holder : !rc_holder) : !reussir.ref<!holder>
  %slot = reussir.ref.project (%ref : !reussir.ref<!holder>) [0] : !reussir.ref<!rc_elem>
  %member = reussir.ref.load (%slot : !reussir.ref<!rc_elem>) : !rc_elem
  reussir.rc.inc (%member : !rc_elem)
  reussir.rc.dec (%holder : !rc_holder)
  %updated = func.call @recurse(%middle, %node) : (!rc_digit, !rc_elem) -> !rc_digit
  %pair = reussir.record.compound(%member, %other : !rc_elem, !rc_elem) : !pair
  %variant = reussir.record.variant[0] (%pair : !pair) : !digit
  %tail = reussir.rc.create value(%variant : !digit) : !rc_digit
  reussir.rc.dec (%updated : !rc_digit)
  return %tail : !rc_digit
}

// A construction that consumes the call result cannot move before that call.
// This is the dominance guard exercised by the natural fingertree corpus when
// a broader first implementation tried to schedule unrelated result carriers.
// SPLIT-LABEL: func.func @depends_on_call(
// SPLIT: %[[CALL:.+]] = call @recurse
// SPLIT-NEXT: %[[PAIR:.+]] = reussir.record.compound
// SPLIT-NEXT: %[[VARIANT:.+]] = reussir.record.variant
// SPLIT-NEXT: %[[RESULT:.+]] = reussir.rc.create
// SPLIT: return %[[RESULT]]
func.func @depends_on_call(%holder: !rc_holder, %middle: !rc_digit,
                           %node: !rc_elem) -> !rc_dependent {
  %ref = reussir.rc.borrow (%holder : !rc_holder) : !reussir.ref<!holder>
  %slot = reussir.ref.project (%ref : !reussir.ref<!holder>) [0] : !reussir.ref<!rc_elem>
  %member = reussir.ref.load (%slot : !reussir.ref<!rc_elem>) : !rc_elem
  reussir.rc.inc (%member : !rc_elem)
  reussir.rc.dec (%holder : !rc_holder)
  %updated = func.call @recurse(%middle, %node) : (!rc_digit, !rc_elem) -> !rc_digit
  %pair = reussir.record.compound(%member, %updated : !rc_elem, !rc_digit) : !dependent_pair
  %variant = reussir.record.variant[0] (%pair : !dependent_pair) : !dependent
  %result = reussir.rc.create value(%variant : !dependent) : !rc_dependent
  return %result : !rc_dependent
}
