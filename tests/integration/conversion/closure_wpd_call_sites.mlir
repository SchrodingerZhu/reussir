// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-lowering-scf-ops,convert-scf-to-cf,reussir-lowering-basic-ops{closure-wpd=1},reussir-convert-to-llvm,reconcile-unrealized-casts)' \
// RUN: | %FileCheck %s

// Closure WPD call-site type tests (docs/design/closure-wpd.md). Every
// indirect vtable call site — eval (evaluate slot), the uniqify expansion's
// closure.clone (clone slot), rc.dec of a closure (drop slot) — asserts the
// type id of its operand's STATIC type with a `reussir.closure.wpd_test`,
// which the dialect's LLVM translation interface lowers to `llvm.type.test`
// + `llvm.assume`.
//
// The ids are STRENGTHENED: apply/uniqify/clone never change the vtable, so
// the walk chases the operand back through them — and through the uniqify
// expansion's control-flow diamond, meeting over both predecessors — to the
// fullest statically visible suffix. In @chain every site therefore tests
// the full two-argument id even though the eval's operand is typed
// `() -> i32`: parameter (i32,i32) -> apply -> uniqify diamond -> apply ->
// eval. In @root no stronger fact is visible and the eval tests the family
// root (no parameter segment between the binder and `E`).

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @root(%g : !reussir.rc<!reussir.closure<() -> i32>>) -> i32 {
    %r = reussir.closure.eval (%g : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    return %r : i32
  }

  func.func @chain(%f : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) -> i32 {
    %a = arith.constant 1 : i32
    %b = arith.constant 2 : i32
    %f1 = reussir.closure.apply (%a : i32) to (%f : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %u = reussir.closure.uniqify (%f1 : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %f2 = reussir.closure.apply (%b : i32) to (%u : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %r = reussir.closure.eval (%f2 : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    return %r : i32
  }
}

// @root — nothing to strengthen: the eval tests the family root, and its
// return segment ([[RET]]) roots the whole family.
// CHECK-LABEL: llvm.func @root
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("_RFK17ReussirClosureWpdE[[RET:C[^"]+]]")

// @chain — the uniqify expansion's clone (clone slot), the shared-path
// rc.dec (drop slot), and the eval (evaluate slot) all strengthened to the
// full signature: parameter segments ([[ARGS]]) before the `E`, same family
// root after it.
// CHECK-LABEL: llvm.func @chain
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("_RFK17ReussirClosureWpd[[ARGS:C[^"]+]]E[[RET]]")
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("_RFK17ReussirClosureWpd[[ARGS]]E[[RET]]")
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("_RFK17ReussirClosureWpd[[ARGS]]E[[RET]]")
