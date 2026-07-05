// RUN: %reussir-opt %s --reussir-special-pointer-tag | %FileCheck %s --check-prefixes=CHECK,TBI
// RUN: %reussir-opt %s --reussir-special-pointer-tag="encoding=immortal" | \
// RUN:   %FileCheck %s --check-prefixes=CHECK,IMMORTAL

// The special-pointer-tag pass rewrites nullary variant constructions of
// taggable shared rc-boxed enums into `reussir.rc.tagged` immediates — the
// rewrite is identical for both encodings; only the stamped attribute (and
// later the LLVM lowering) differs. It
// must match the *unfused* frontend chain (`record.compound {}` →
// `record.variant` → `rc.create`), because the fused `rc.create_variant`
// form only appears after token instantiation — too late to elide the
// allocation. The module is stamped with `reussir.special_ptr_tag` so the
// LLVM lowering knows immediates may flow.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Nil" [value] {}>}>
!nil = !reussir.record<compound "List.Nil" [value] {}>
!cons = !reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List" {!reussir.record<compound "List.Cons">, !reussir.record<compound "List.Nil" [value] {}>}>}>
!rclist = !reussir.rc<!list>

// TBI: module @test attributes {{{.*}}reussir.special_ptr_tag = "tbi"{{.*}}}
// IMMORTAL: module @test attributes {{{.*}}reussir.special_ptr_tag = "immortal"{{.*}}}
module @test {
  // The nullary arm becomes an immediate; no box is created.
  // CHECK-LABEL: func.func @make_nil
  // CHECK: %[[IMM:.+]] = reussir.rc.tagged tag(1) : !reussir.rc<!reussir.record<variant "List"
  // CHECK-NOT: reussir.rc.create
  // CHECK: return %[[IMM]]
  func.func @make_nil() -> !rclist {
    %0 = reussir.record.compound : !nil
    %1 = reussir.record.variant[1] (%0 : !nil) : !list
    %2 = reussir.rc.create value(%1 : !list) : !rclist
    return %2 : !rclist
  }

  // An inhabited arm keeps its ordinary boxed construction.
  // CHECK-LABEL: func.func @make_cons
  // CHECK: reussir.record.variant
  // CHECK: reussir.rc.create value
  // CHECK-NOT: reussir.rc.tagged
  func.func @make_cons(%head: i64, %tail: !rclist) -> !rclist {
    %0 = reussir.record.compound(%head, %tail : i64, !rclist) : !cons
    %1 = reussir.record.variant[0] (%0 : !cons) : !list
    %2 = reussir.rc.create value(%1 : !list) : !rclist
    return %2 : !rclist
  }
}
