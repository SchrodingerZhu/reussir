// RUN: %reussir-opt %s --reussir-lowering-scf-ops | %FileCheck %s

// Under the immortal encoding a nullary arm's value is always the per-tag
// immediate, so dispatch peels such arms into `rc.compare_immortal` guards
// (a pointer compare — no load); when a single arm remains it inlines as
// the residual else with no tag read at all.

!list_ = !reussir.record<variant "List" incomplete>
!list_nil = !reussir.record<compound "List::Nil" [value] {}>
!list_cons = !reussir.record<compound "List::Cons" [value] {i64, !list_}>
!list = !reussir.record<variant "List" {!list_nil, !list_cons}>
!rc_list = !reussir.rc<!list>

module attributes {reussir.special_ptr_tag = "immortal"} {
  // CHECK-LABEL: func.func @sum_head(
  // CHECK: %[[GUARD:.+]] = reussir.rc.compare_immortal(%{{.+}} : !reussir.rc<{{.*}}) tag(0)
  // CHECK: scf.if %[[GUARD]] -> (i64) {
  // CHECK:   arith.constant 0
  // CHECK: } else {
  // CHECK-NOT: reussir.record.tag
  // CHECK-NOT: scf.index_switch
  // CHECK:   reussir.record.coerce
  // CHECK:   reussir.ref.load
  // CHECK: }
  func.func @sum_head(%l: !rc_list) -> i64 {
    %ref = reussir.rc.borrow (%l : !rc_list) : !reussir.ref<!list>
    %r = reussir.record.dispatch (%ref : !reussir.ref<!list>) -> i64 {
      [0] -> {
        ^bb0(%nil: !reussir.ref<!list_nil>):
        %zero = arith.constant 0 : i64
        reussir.scf.yield %zero : i64
      }
      [1] -> {
        ^bb1(%cons: !reussir.ref<!list_cons>):
        %slot = reussir.ref.project (%cons : !reussir.ref<!list_cons>) [0] : !reussir.ref<i64>
        %v = reussir.ref.load (%slot : !reussir.ref<i64>) : i64
        reussir.scf.yield %v : i64
      }
    }
    return %r : i64
  }
}
