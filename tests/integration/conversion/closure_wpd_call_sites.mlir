// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-lowering-scf-ops,convert-scf-to-cf,reussir-lowering-basic-ops{closure-wpd=1},reussir-convert-to-llvm,reconcile-unrealized-casts)' \
// RUN: | %FileCheck %s

// Closure WPD call-site type tests (docs/design/closure-wpd.md). Every
// indirect vtable call site — eval (evaluate slot), the uniqify expansion's
// closure.clone (clone slot), rc.dec of a closure (drop slot) — asserts its
// operand's return-type family id with a `reussir.closure.wpd_test`, which
// the dialect's LLVM translation interface lowers to `llvm.type.test` +
// `llvm.assume`.
//
// The test is emitted ON THE LOADED VTABLE POINTER — the same SSA value the
// slot GEP and call hang off. That discipline is load-bearing:
// WholeProgramDevirt discovers devirtualizable calls by a pure def-use walk
// from the tested pointer, so testing a separate load of the same slot
// would disconnect the assertion from the call. All sites in both functions
// share one id ([[ID]]): the family is keyed by the return type alone, and
// apply/uniqify never change the backing vtable.

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

// @root — the eval asserts the return-type family id on the vtable pointer
// it just loaded, right before the slot GEP.
// CHECK-LABEL: llvm.func @root
// CHECK: %[[VT:[0-9]+]] = llvm.load %{{[0-9]+}} invariant_group
// CHECK-NEXT: reussir.closure.wpd_test(%[[VT]] : !llvm.ptr) id("reussir.closure.wpd.[[ID:[0-9A-Za-z]+]]")
// CHECK-NEXT: llvm.getelementptr %[[VT]]

// @chain — the uniqify expansion's clone (clone slot), the shared-path
// rc.dec (drop slot), and the eval (evaluate slot) all assert the same
// return-type family id.
// CHECK-LABEL: llvm.func @chain
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("reussir.closure.wpd.[[ID]]")
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("reussir.closure.wpd.[[ID]]")
// CHECK: reussir.closure.wpd_test(%{{[0-9]+}} : !llvm.ptr) id("reussir.closure.wpd.[[ID]]")
