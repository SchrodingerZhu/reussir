// RUN: %reussir-opt %s --reussir-partial-move --reussir-rc-dispatch-fusion | %FileCheck %s

// Projection-only compound owners can move at their first consuming use even
// when a late scalar projection originally kept the carrier alive across a
// call. Every nontrivial member transfers, so the advanced dec only frees the
// box on its unique path and rematerializes the transferred retain when shared.

!leaf = !reussir.record<compound "Leaf" [value] {i64}>
!tree = !reussir.record<variant "Tree" {!leaf}>
!rc_tree = !reussir.rc<!tree>
!carrier = !reussir.record<compound "Carrier" {!tree, i1, i1}>
!rc_carrier = !reussir.rc<!carrier>

func.func private @consume_tree(!rc_tree, i1) -> !rc_tree
func.func private @consume_one(!rc_tree)

// CHECK-LABEL: func.func @lazy_projection(
// CHECK: %[[ROOT_REF:.+]] = reussir.rc.borrow
// CHECK: %[[TREE_SLOT:.+]] = reussir.ref.project (%[[ROOT_REF]]
// CHECK: %[[TREE:.+]] = reussir.ref.load (%[[TREE_SLOT]]
// CHECK-NOT: reussir.rc.inc
// CHECK: reussir.ref.project
// CHECK: reussir.ref.load
// CHECK: reussir.ref.project
// CHECK: %[[LATE:.+]] = reussir.ref.load
// CHECK: reussir.rc.dec
// CHECK-SAME: boundMembers = array<i64: 0>
// CHECK-NEXT: %{{.+}} = call @consume_tree(%[[TREE]],
// CHECK: return %[[LATE]]
func.func @lazy_projection(%carrier: !rc_carrier) -> i1 {
  %ref = reussir.rc.borrow (%carrier : !rc_carrier) : !reussir.ref<!carrier>
  %tree_slot = reussir.ref.project (%ref : !reussir.ref<!carrier>) [0] : !reussir.ref<!rc_tree>
  %tree = reussir.ref.load (%tree_slot : !reussir.ref<!rc_tree>) : !rc_tree
  reussir.rc.inc (%tree : !rc_tree)
  %early_slot = reussir.ref.project (%ref : !reussir.ref<!carrier>) [1] : !reussir.ref<i1>
  %early = reussir.ref.load (%early_slot : !reussir.ref<i1>) : i1
  %updated = func.call @consume_tree(%tree, %early) : (!rc_tree, i1) -> !rc_tree
  %late_slot = reussir.ref.project (%ref : !reussir.ref<!carrier>) [2] : !reussir.ref<i1>
  %late = reussir.ref.load (%late_slot : !reussir.ref<i1>) : i1
  reussir.rc.dec (%carrier : !rc_carrier)
  reussir.rc.dec (%updated : !rc_tree)
  return %late : i1
}

// Repeated extraction of one field transfers the box's one owner and keeps
// the extra copy's retain. Duplicate boundMembers would be unsound because the
// unique box contains only one reference to transfer.
// CHECK-LABEL: func.func @duplicate_projection(
// CHECK-COUNT-1: reussir.rc.inc
// CHECK: reussir.rc.dec
// CHECK-SAME: boundMembers = array<i64: 0>
// CHECK-NEXT: call @consume_one
func.func @duplicate_projection(%carrier: !rc_carrier) -> !rc_tree {
  %ref0 = reussir.rc.borrow (%carrier : !rc_carrier) : !reussir.ref<!carrier>
  %slot0 = reussir.ref.project (%ref0 : !reussir.ref<!carrier>) [0] : !reussir.ref<!rc_tree>
  %tree0 = reussir.ref.load (%slot0 : !reussir.ref<!rc_tree>) : !rc_tree
  reussir.rc.inc (%tree0 : !rc_tree)
  func.call @consume_one(%tree0) : (!rc_tree) -> ()
  %ref1 = reussir.rc.borrow (%carrier : !rc_carrier) : !reussir.ref<!carrier>
  %slot1 = reussir.ref.project (%ref1 : !reussir.ref<!carrier>) [0] : !reussir.ref<!rc_tree>
  %tree1 = reussir.ref.load (%slot1 : !reussir.ref<!rc_tree>) : !rc_tree
  reussir.rc.inc (%tree1 : !rc_tree)
  %late_slot = reussir.ref.project (%ref1 : !reussir.ref<!carrier>) [2] : !reussir.ref<i1>
  %late = reussir.ref.load (%late_slot : !reussir.ref<i1>) : i1
  reussir.rc.dec (%carrier : !rc_carrier)
  return %tree1 : !rc_tree
}

!two_trees = !reussir.record<compound "TwoTrees" {!tree, !tree, i1}>
!rc_two_trees = !reussir.rc<!two_trees>

// A nontrivial unbound member could run arbitrary drop glue before the call,
// so incomplete transfers stay untouched.
// CHECK-LABEL: func.func @incomplete_transfer(
// CHECK: reussir.rc.inc
// CHECK: call @consume_one
// CHECK: reussir.rc.dec
// CHECK-NOT: boundMembers
func.func @incomplete_transfer(%carrier: !rc_two_trees) -> i1 {
  %ref = reussir.rc.borrow (%carrier : !rc_two_trees) : !reussir.ref<!two_trees>
  %tree_slot = reussir.ref.project (%ref : !reussir.ref<!two_trees>) [0] : !reussir.ref<!rc_tree>
  %tree = reussir.ref.load (%tree_slot : !reussir.ref<!rc_tree>) : !rc_tree
  reussir.rc.inc (%tree : !rc_tree)
  func.call @consume_one(%tree) : (!rc_tree) -> ()
  %late_slot = reussir.ref.project (%ref : !reussir.ref<!two_trees>) [2] : !reussir.ref<i1>
  %late = reussir.ref.load (%late_slot : !reussir.ref<i1>) : i1
  reussir.rc.dec (%carrier : !rc_two_trees)
  return %late : i1
}

// A count observer on a same-typed MayAlias value blocks the advanced release.
// CHECK-LABEL: func.func @count_observer(
// CHECK: reussir.rc.inc
// CHECK: reussir.rc.is_unique
// CHECK: call @consume_one
// CHECK: reussir.rc.dec
// CHECK-NOT: boundMembers
func.func @count_observer(%carrier: !rc_carrier, %alias: !rc_carrier) -> i1 {
  %ref = reussir.rc.borrow (%carrier : !rc_carrier) : !reussir.ref<!carrier>
  %tree_slot = reussir.ref.project (%ref : !reussir.ref<!carrier>) [0] : !reussir.ref<!rc_tree>
  %tree = reussir.ref.load (%tree_slot : !reussir.ref<!rc_tree>) : !rc_tree
  reussir.rc.inc (%tree : !rc_tree)
  %unique = reussir.rc.is_unique (%alias : !rc_carrier) : i1
  func.call @consume_one(%tree) : (!rc_tree) -> ()
  %late_slot = reussir.ref.project (%ref : !reussir.ref<!carrier>) [2] : !reussir.ref<i1>
  %late = reussir.ref.load (%late_slot : !reussir.ref<i1>) : i1
  reussir.rc.dec (%carrier : !rc_carrier)
  reussir.rc.dec (%alias : !rc_carrier)
  return %late : i1
}
