// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// A destructuring decrement transfers a retain to each bound member
// (`reussir-rc-dispatch-fusion` folded the member's `rc.inc` into the
// release). Cancelling `inc %box; ...; dec %box {boundMembers}` must
// rematerialize those member retains: erasing the pair outright leaves the
// consuming call below with a reference nobody paid for, and the member's
// box is freed while still live (heap-use-after-free at run time).

!list_ = !reussir.record<variant "List" incomplete>
!list_nil = !reussir.record<compound "List::Nil" [value] { }>
!list_cons = !reussir.record<compound "List::Cons" [value]{ i64, !list_ }>
!list = !reussir.record<variant "List" { !list_nil, !list_cons }>
!pair = !reussir.record<compound "Pair" { i64, !list }>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> }  {
  func.func private @consume(!reussir.rc<!list>) -> i64

  func.func @cancelled_release_keeps_bound_retain(%box: !reussir.rc<!pair>) -> i64 {
    reussir.rc.inc(%box : !reussir.rc<!pair>)
    %ref = reussir.rc.borrow(%box : !reussir.rc<!pair>) : !reussir.ref<!pair shared>
    %slot = reussir.ref.project(%ref : !reussir.ref<!pair shared>) [1] : !reussir.ref<!reussir.rc<!list> shared>
    %member = reussir.ref.load(%slot : !reussir.ref<!reussir.rc<!list> shared>) : !reussir.rc<!list>
    %token = reussir.rc.dec(%box : !reussir.rc<!pair>) : !reussir.nullable<!reussir.token<align: 8, size: 24>> {boundMembers = array<i64: 1>}
    %res = func.call @consume(%member) : (!reussir.rc<!list>) -> i64
    return %res : i64
  }
}

// The inc/dec pair on the box cancels, but the bound member's retain must
// survive as a rematerialized project + load + inc feeding the consuming
// call.
// CHECK-LABEL: func.func @cancelled_release_keeps_bound_retain
// CHECK-NOT: reussir.rc.dec
// CHECK: reussir.ref.load
// CHECK: %[[REMAT:.+]] = reussir.ref.load
// CHECK-NEXT: reussir.rc.inc(%[[REMAT]]
// CHECK-NOT: reussir.rc.dec
// CHECK: call @consume
