// RUN: %reussir-opt %s \
// RUN:   --reussir-rc-create-fusion \
// RUN:   --reussir-trmc-recursion-analysis | %FileCheck %s

!list_ = !reussir.record<variant "List" incomplete>
!list_cons = !reussir.record<compound "List::Cons" [value] {i64, !list_}>
!list_nil = !reussir.record<compound "List::Nil" [value] {}>
!list = !reussir.record<variant "List" {!list_cons, !list_nil}>
!rc_list = !reussir.rc<!list>

module {
  // The original function becomes a thin wrapper entering the helper through
  // the empty constructor context.
  // CHECK-LABEL: func.func @map_plus1(
  // CHECK: %[[EMPTY:.+]] = reussir.cctx.empty : !reussir.cctx<!reussir.rc<!reussir.record<variant "List"
  // CHECK: %[[RES:.+]] = call @map_plus1.trmc(%{{.+}}, %[[EMPTY]])
  // CHECK: return %[[RES]]
  func.func @map_plus1(%list: !rc_list) -> !rc_list {
    %list_ref = reussir.rc.borrow (%list : !rc_list) : !reussir.ref<!list>
    %result = reussir.record.dispatch (%list_ref : !reussir.ref<!list>) -> !rc_list {
      [0] -> {
        ^bb0(%cons_ref: !reussir.ref<!list_cons>):
        %head_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [0] : !reussir.ref<i64>
        %head = reussir.ref.load (%head_ref : !reussir.ref<i64>) : i64
        %tail_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [1] : !reussir.ref<!rc_list>
        %tail = reussir.ref.load (%tail_ref : !reussir.ref<!rc_list>) : !rc_list
        reussir.rc.inc (%tail : !rc_list)
        %one = arith.constant 1 : i64
        %next_head = arith.addi %head, %one : i64
        %mapped_tail = func.call @map_plus1(%tail) : (!rc_list) -> !rc_list
        %new_cons = reussir.record.compound(%next_head, %mapped_tail : i64, !rc_list) : !list_cons
        %new_variant = reussir.record.variant[0] (%new_cons : !list_cons) : !list
        %new_list = reussir.rc.create value(%new_variant : !list) : !rc_list
        reussir.scf.yield %new_list : !rc_list
      }
      [1] -> {
        ^bb1(%nil_ref: !reussir.ref<!list_nil>):
        reussir.scf.yield %list : !rc_list
      }
    }
    return %result : !rc_list
  }

  // The helper threads a `!reussir.cctx` accumulator: the constructor arm
  // allocates with a hole, extends the context, and tail-calls itself; the
  // base-case arm completes the context with `cctx.apply`.
  // CHECK-LABEL: func.func private @map_plus1.trmc(
  // CHECK-SAME: %[[LIST:.+]]: !reussir.rc<!reussir.record<variant "List"
  // CHECK-SAME: %[[CTX:.+]]: !reussir.cctx<!reussir.rc<!reussir.record<variant "List"
  // CHECK-SAME: {llvm.noundef})
  // CHECK-SAME: attributes {llvm.linkage = #llvm.linkage<internal>}
  // CHECK: reussir.record.dispatch
  // CHECK: %[[NEW:.+]], %[[TAILHOLE:.+]] = "reussir.rc.create_variant"(%{{.+}}, %{{.+}}) <{holeFields = array<i64: 1>, operandSegmentSizes = array<i32: 0, 2, 0, 0>, tag = 0 : index}>
  // CHECK-SAME: : (i64, !reussir.rc<!reussir.record<variant "List"
  // CHECK-SAME: -> (!reussir.rc<!reussir.record<variant "List"
  // CHECK-SAME: !reussir.hole<!reussir.rc<!reussir.record<variant "List"
  // CHECK: %[[EXT:.+]] = reussir.cctx.extend(%[[CTX]] : !reussir.cctx<{{.+}}) plug(%[[NEW]] : {{.+}}) hole(%[[TAILHOLE]] : {{.+}})
  // CHECK: %[[TAIL:.+]] = func.call @map_plus1.trmc(%{{.+}}, %[[EXT]])
  // CHECK: reussir.scf.yield %[[TAIL]]
  // CHECK: %[[DONE:.+]] = reussir.cctx.apply(%[[CTX]] : !reussir.cctx<{{.+}}) (%[[LIST]] : {{.+}})
  // CHECK: reussir.scf.yield %[[DONE]]
}
