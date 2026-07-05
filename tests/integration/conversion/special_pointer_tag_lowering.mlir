// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

// LLVM lowering under the special-pointer-tag scheme (module stamped with
// `reussir.special_ptr_tag`). A tagged immediate's top byte is `tag + 1`
// and its low bits hold the address of the per-tag dummy box
// `{refcount = 2, tag}`. TBI makes loads through the tagged value land in
// the dummy box, so the hot reads (rc.fetch, rc.is_unique, record.tag)
// lower to plain loads with no immediate-check; only `rc.set` — the one
// store that could decay the dummy's refcount to 1 — steers a tagged
// access to the scratch word.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

// CHECK-DAG: @__reussir_tag_scratch = internal global i64 0
// CHECK-DAG: @__reussir_tag_dummy_1 = internal global [2 x i64] [i64 2, i64 1]
module @test attributes { reussir.special_ptr_tag, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {

  // Immediate materialization: dummy address | (tag + 1) << 56, no
  // allocation. (2 << 56 = 144115188075855872.)
  // CHECK-LABEL: define ptr @make_nil()
  // CHECK: %[[ENC:.+]] = or i64 ptrtoint (ptr @__reussir_tag_dummy_1 to i64), 144115188075855872
  // CHECK: %[[PTR:.+]] = inttoptr i64 %[[ENC]] to ptr
  // CHECK: ret ptr %[[PTR]]
  func.func @make_nil() -> !rclist {
    %0 = reussir.rc.tagged tag(1) : !rclist
    return %0 : !rclist
  }

  // Hot read: plain refcount load, no top-byte decode, no select.
  // CHECK-LABEL: define i64 @fetch(ptr %0)
  // CHECK-NOT: lshr
  // CHECK-NOT: select
  // CHECK: %[[CNT:.+]] = load i64, ptr
  // CHECK: ret i64 %[[CNT]]
  func.func @fetch(%rc: !rclist) -> index {
    %cnt = reussir.rc.fetch (%rc : !rclist) : index
    return %cnt : index
  }

  // Hot read: plain tag load — a tagged immediate's load lands in its
  // per-tag dummy box and reads the real tag, so no decode is emitted.
  // CHECK-LABEL: define i64 @tag(ptr %0)
  // CHECK-NOT: lshr
  // CHECK-NOT: select
  // CHECK: %[[TAG:.+]] = load i64, ptr
  // CHECK: ret i64 %[[TAG]]
  func.func @tag(%ref: !reussir.ref<!list shared>) -> index {
    %tag = reussir.record.tag(%ref : !reussir.ref<!list shared>) : index
    return %tag : index
  }

  // The decrementing store: a tagged access is steered to the scratch word
  // so the dummy refcount can never decay below 2.
  // CHECK-LABEL: define void @set(ptr %0, i64 %1)
  // CHECK: %[[INT:.+]] = ptrtoint ptr %0 to i64
  // CHECK: %[[TOP:.+]] = lshr i64 %[[INT]], 56
  // CHECK: %[[ISTAG:.+]] = icmp ne i64 %[[TOP]], 0
  // CHECK: %[[ADDR:.+]] = select i1 %[[ISTAG]], ptr @__reussir_tag_scratch, ptr %0
  // CHECK: store i64 %1, ptr %[[ADDR]]
  func.func @set(%rc: !rclist, %v: index) {
    reussir.rc.set(%rc : !rclist, %v : index)
    return
  }
}
