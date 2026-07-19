// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=EXPAND
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1})' | %FileCheck %s --check-prefix=OUTLINE
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},reussir-convert-to-std,convert-scf-to-cf,convert-to-llvm)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

// The whole-subtree atomicity rule (thread-safety design §3.3). Two shapes:
//
// 1. An `Arc` island in normal space: `Pair` is a normal-context value
//    holding one explicitly atomic rc member (an `Arc` member — the member
//    type spells `rc<… atomic>` because the container cannot derive it) and
//    one plain shared member. Acquire/drop of the pair must retain/release
//    the two links at their own disciplines.
//
// 2. The arc'd recursive spine: dropping `rc<AList, …, atomic>` floods the
//    atomic context down — the Cons tail member is *declared bare* (one
//    record serves both colorings) but its link derives atomic from the
//    parent box, and the outlined drop glue forks per atomic kind (its
//    argument reference carries the context its body derives from).

!inner = !reussir.record<compound "Inner" [shared] { i64 }>
!pair = !reussir.record<compound "Pair" [value] { !reussir.rc<!inner atomic>, !inner }>

!alist_ = !reussir.record<variant "AList" incomplete>
!alist_nil = !reussir.record<compound "AList::Nil" [value] { }>
!alist_cons = !reussir.record<compound "AList::Cons" [value] { i64, !alist_ }>
!alist = !reussir.record<variant "AList" { !alist_nil, !alist_cons }>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // The acquire glue loads each member link and bumps it at its own
  // discipline: member 0 through the written atomic rc type, member 1
  // through the derived normal link.
  // EXPAND-LABEL: @acquire_pair
  // EXPAND: reussir.ref.project{{.*}}[0]
  // EXPAND: reussir.ref.load{{.*}} : !reussir.rc<!reussir.record<compound "Inner" {i64}> atomic>
  // EXPAND: reussir.rc.inc(%{{[0-9a-z]+}} : !reussir.rc<!reussir.record<compound "Inner" {i64}> atomic>)
  // EXPAND: reussir.ref.project{{.*}}[1]
  // EXPAND: reussir.ref.load{{.*}} : !reussir.rc<!reussir.record<compound "Inner" {i64}>>
  // EXPAND: reussir.rc.inc(%{{[0-9a-z]+}} : !reussir.rc<!reussir.record<compound "Inner" {i64}>>)

  // LLVM-LABEL: @_RINvNvC4core9intrinsic16acquire_in_placePairE
  // LLVM: atomicrmw add ptr %{{[0-9]+}}, i32 1 monotonic
  func.func @acquire_pair(%ref : !reussir.ref<!pair>) {
    reussir.ref.acquire (%ref : !reussir.ref<!pair>)
    return
  }

  // Dropping the pair releases the atomic member through the acq_rel
  // `atomicrmw sub` discipline and the plain member nonatomically.
  // EXPAND-LABEL: @drop_pair
  // EXPAND: reussir.rc.dec(%{{[0-9a-z]+}} : !reussir.rc<!reussir.record<compound "Inner" {i64}> atomic>)
  // EXPAND: reussir.rc.dec(%{{[0-9a-z]+}} : !reussir.rc<!reussir.record<compound "Inner" {i64}>>)
  func.func @drop_pair(%ref : !reussir.ref<!pair>) {
    reussir.ref.drop (%ref : !reussir.ref<!pair>)
    return
  }

  // The recursive spine outlines to a per-kind dtor: the atomic glue takes
  // an *atomic* reference, dispatches, and releases the bare-declared Cons
  // tail by calling itself — the whole spine shares one coloring.
  // OUTLINE-LABEL: func.func private @_RINvNvC4core9intrinsic20drop_in_place_atomicAListE
  // OUTLINE-SAME: !reussir.ref<!reussir.record<variant "AList" {{.*}}> atomic>
  // OUTLINE: reussir.record.dispatch
  // OUTLINE: func.call @_RINvNvC4core9intrinsic20drop_in_place_atomicAListE({{.*}} atomic>) -> ()
  // OUTLINE-LABEL: @drop_alist
  // OUTLINE: func.call @_RINvNvC4core9intrinsic20drop_in_place_atomicAListE

  // LLVM-LABEL: @_RINvNvC4core9intrinsic20drop_in_place_atomicAListE
  // LLVM: atomicrmw sub ptr %{{[0-9]+}}, i32 1 acq_rel
  func.func @drop_alist(%rc : !reussir.rc<!alist atomic>) {
    %t = reussir.rc.dec (%rc : !reussir.rc<!alist atomic>) : !reussir.nullable<!reussir.token<align: 8, size: ?>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: ?>>)
    return
  }
}
