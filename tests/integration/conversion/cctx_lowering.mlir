// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

// LLVM lowering of first-class constructor contexts: a `!reussir.cctx` is a
// `{root, hole}` pointer pair with a null hole denoting the empty context.
// Extend and apply are branch-free — the plugging store is steered to the
// per-module scratch word when the context is empty, and the root/result
// is selected instead.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>
!cctx = !reussir.cctx<!rclist>
!hole = !reussir.hole<!rclist>

// CHECK-DAG: @_RNvC19REUSSIR_TAG_SCRATCH43DvQR3LzfbDLEt0P0zamCLh5N8ob8mGkcKPkikTHzGH2 = internal global i64 0
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {

  // The empty context is all-null.
  // CHECK-LABEL: define { ptr, ptr } @empty()
  // CHECK: ret { ptr, ptr } zeroinitializer
  func.func @empty() -> !cctx {
    %0 = reussir.cctx.empty : !cctx
    return %0 : !cctx
  }

  // Extend plugs the constructor into the hole (scratch-steered when empty),
  // selects the new root, and refocuses on the constructor's own hole.
  // CHECK-LABEL: define { ptr, ptr } @extend({ ptr, ptr } %0, ptr %1, ptr %2)
  // CHECK: %[[ROOT0:.+]] = extractvalue { ptr, ptr } %0, 0
  // CHECK: %[[HOLE0:.+]] = extractvalue { ptr, ptr } %0, 1
  // CHECK: %[[ISEMPTY:.+]] = icmp eq ptr %[[HOLE0]], null
  // CHECK: %[[ADDR:.+]] = select i1 %[[ISEMPTY]], ptr @_RNvC19REUSSIR_TAG_SCRATCH43DvQR3LzfbDLEt0P0zamCLh5N8ob8mGkcKPkikTHzGH2, ptr %[[HOLE0]]
  // CHECK: store ptr %1, ptr %[[ADDR]]
  // CHECK: %[[ROOT:.+]] = select i1 %[[ISEMPTY]], ptr %1, ptr %[[ROOT0]]
  // CHECK: insertvalue { ptr, ptr } undef, ptr %[[ROOT]], 0
  // CHECK: insertvalue { ptr, ptr } {{.+}}, ptr %2, 1
  func.func @extend(%ctx: !cctx, %rc: !rclist, %h: !hole) -> !cctx {
    %0 = reussir.cctx.extend(%ctx : !cctx) plug(%rc : !rclist) hole(%h : !hole) : !cctx
    return %0 : !cctx
  }

  // Apply plugs the final value (scratch-steered when empty) and yields the
  // root — or the value itself for the empty context.
  // CHECK-LABEL: define ptr @apply({ ptr, ptr } %0, ptr %1)
  // CHECK: %[[AROOT:.+]] = extractvalue { ptr, ptr } %0, 0
  // CHECK: %[[AHOLE:.+]] = extractvalue { ptr, ptr } %0, 1
  // CHECK: %[[AEMPTY:.+]] = icmp eq ptr %[[AHOLE]], null
  // CHECK: %[[AADDR:.+]] = select i1 %[[AEMPTY]], ptr @_RNvC19REUSSIR_TAG_SCRATCH43DvQR3LzfbDLEt0P0zamCLh5N8ob8mGkcKPkikTHzGH2, ptr %[[AHOLE]]
  // CHECK: store ptr %1, ptr %[[AADDR]]
  // CHECK: %[[ARES:.+]] = select i1 %[[AEMPTY]], ptr %1, ptr %[[AROOT]]
  // CHECK: ret ptr %[[ARES]]
  func.func @apply(%ctx: !cctx, %v: !rclist) -> !rclist {
    %0 = reussir.cctx.apply(%ctx : !cctx) (%v : !rclist) : !rclist
    return %0 : !rclist
  }
}
