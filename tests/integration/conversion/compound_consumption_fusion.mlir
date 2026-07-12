// RUN: %reussir-opt %s --reussir-rc-dispatch-fusion | %FileCheck %s

// A rebuild-style consumption of a compound box: fields are loaded and
// retained, then the box is released. The fusion folds the retains into a
// tagless destructuring decrement (boundMembers, no destructureTag) so the
// unique path transfers the kept members instead of round-tripping their
// counts.

!qlist_ = !reussir.record<variant "QList" incomplete>
!qlist_cons = !reussir.record<compound "QList::Cons" [value] {i64, !qlist_}>
!qlist_nil = !reussir.record<compound "QList::Nil" [value] {}>
!qlist = !reussir.record<variant "QList" {!qlist_cons, !qlist_nil}>
!rc_qlist = !reussir.rc<!qlist>
!state = !reussir.record<compound "State" [value] {i64, !qlist}>
!queue = !reussir.record<compound "Queue" {i64, !qlist, !state}>
!rc_queue = !reussir.rc<!queue>

// CHECK-LABEL: func.func @rebuild(
// CHECK-NOT: reussir.rc.inc
// CHECK-NOT: reussir.ref.acquire
// CHECK: reussir.rc.dec(%{{.+}} : !reussir.rc<!reussir.record<compound "Queue"
// CHECK-SAME: boundMembers = array<i64: 2, 1>
// CHECK: reussir.record.compound
func.func @rebuild(%q: !rc_queue, %len: i64) -> !queue {
  %ref = reussir.rc.borrow (%q : !rc_queue) : !reussir.ref<!queue>
  %front_slot = reussir.ref.project (%ref : !reussir.ref<!queue>) [1] : !reussir.ref<!rc_qlist>
  %front = reussir.ref.load (%front_slot : !reussir.ref<!rc_qlist>) : !rc_qlist
  reussir.rc.inc (%front : !rc_qlist)
  %state_slot = reussir.ref.project (%ref : !reussir.ref<!queue>) [2] : !reussir.ref<!state>
  %state = reussir.ref.load (%state_slot : !reussir.ref<!state>) : !state
  %spilled = reussir.ref.spilled (%state : !state) : !reussir.ref<!state>
  reussir.ref.acquire (%spilled : !reussir.ref<!state>)
  reussir.rc.dec (%q : !rc_queue)
  %rebuilt = reussir.record.compound(%len, %front, %state : i64, !rc_qlist, !state) : !queue
  return %rebuilt : !queue
}

// A call between the retain and the release blocks the fusion: the callee
// could release the box or observe its count.
// CHECK-LABEL: func.func @blocked(
// CHECK: reussir.rc.inc
// CHECK: call @opaque
// CHECK: reussir.rc.dec
// CHECK-NOT: boundMembers
func.func private @opaque()
func.func @blocked(%q: !rc_queue) -> !rc_qlist {
  %ref = reussir.rc.borrow (%q : !rc_queue) : !reussir.ref<!queue>
  %front_slot = reussir.ref.project (%ref : !reussir.ref<!queue>) [1] : !reussir.ref<!rc_qlist>
  %front = reussir.ref.load (%front_slot : !reussir.ref<!rc_qlist>) : !rc_qlist
  reussir.rc.inc (%front : !rc_qlist)
  func.call @opaque() : () -> ()
  reussir.rc.dec (%q : !rc_queue)
  return %front : !rc_qlist
}
