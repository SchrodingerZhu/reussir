// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

// LLVM lowering under the *arch-independent* (immortal) special-pointer-tag
// encoding (module stamped `reussir.special_ptr_tag = "immortal"`). The
// immediate is the per-tag dummy box's address itself — no TBI/LAM pointer
// tricks — and the dummy's refcount is initialized to the immortal value
// 3 * 2^62 = 0xC000000000000000 for a 64-bit refcount word. Hot reads
// (rc.fetch, rc.is_unique, record.tag) are plain loads; `rc.set` recognizes
// a dummy by the magnitude of the stored count (>= 2^63; a real refcount
// can never get there) and steers it to the scratch word.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

// CHECK-DAG: @__reussir_tag_scratch = internal global i64 0
// CHECK-DAG: @__reussir_tag_dummy_1 = internal global [2 x i64] [i64 -4611686018427387904, i64 1]
module @test attributes { reussir.special_ptr_tag = "immortal", dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {

  // Immediate materialization: the plain dummy address — no or/inttoptr.
  // CHECK-LABEL: define ptr @make_nil()
  // CHECK-NOT: or i64
  // CHECK-NOT: inttoptr
  // CHECK: ret ptr @__reussir_tag_dummy_1
  func.func @make_nil() -> !rclist {
    %0 = reussir.rc.tagged tag(1) : !rclist
    return %0 : !rclist
  }

  // Hot read: plain refcount load, no pointer decode, no select.
  // CHECK-LABEL: define i64 @fetch(ptr %0)
  // CHECK-NOT: lshr
  // CHECK-NOT: select
  // CHECK: %[[CNT:.+]] = load i64, ptr
  // CHECK: ret i64 %[[CNT]]
  func.func @fetch(%rc: !rclist) -> index {
    %cnt = reussir.rc.fetch (%rc : !rclist) : index
    return %cnt : index
  }

  // Hot read: plain tag load — an immediate's load lands in its per-tag
  // dummy box and reads the real tag.
  // CHECK-LABEL: define i64 @tag(ptr %0)
  // CHECK-NOT: lshr
  // CHECK-NOT: select
  // CHECK: %[[TAG:.+]] = load i64, ptr
  // CHECK: ret i64 %[[TAG]]
  func.func @tag(%ref: !reussir.ref<!list shared>) -> index {
    %tag = reussir.record.tag(%ref : !reussir.ref<!list shared>) : index
    return %tag : index
  }

  // The decrementing store: a count at or above 2^63 identifies a dummy
  // (its stored value is `immortal - 1`, still over the threshold), and the
  // store is steered to the scratch word so the dummy never decays.
  // CHECK-LABEL: define void @set(ptr %0, i64 %1)
  // CHECK-NOT: lshr
  // CHECK: %[[ISDUMMY:.+]] = icmp uge i64 %1, -9223372036854775808
  // CHECK: %[[ADDR:.+]] = select i1 %[[ISDUMMY]], ptr @__reussir_tag_scratch, ptr %0
  // CHECK: store i64 %1, ptr %[[ADDR]]
  func.func @set(%rc: !rclist, %v: index) {
    reussir.rc.set(%rc : !rclist, %v : index)
    return
  }
}
