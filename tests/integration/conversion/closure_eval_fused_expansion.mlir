// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-lowering-scf-ops)' \
// RUN: | %FileCheck %s

// A fused `closure.eval` (`with` pack) expands to unchecked applies plus the
// plain eval — and nothing else. In particular NO `closure.uniqify` is
// introduced: the fused form carries `apply`'s contract (uniqueness is the
// producer's obligation), and whenever a check was genuinely needed the
// beta-reduction pass left the original bottom-most uniqify in place as this
// op's operand.

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @expand(%f: !reussir.rc<!reussir.closure<(i32, i32) -> i32>>, %x: i32, %y: i32) -> i32 {
    %r = reussir.closure.eval (%f : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) with (%x, %y : i32, i32) : i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @expand
// CHECK-NOT: reussir.closure.uniqify
// CHECK: %[[C1:[0-9a-z]+]] = reussir.closure.apply(%arg1 : i32) to(%arg0 : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
// CHECK-NOT: reussir.closure.uniqify
// CHECK: %[[C2:[0-9a-z]+]] = reussir.closure.apply(%arg2 : i32) to(%[[C1]] : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
// CHECK-NOT: reussir.closure.uniqify
// CHECK: reussir.closure.eval(%[[C2]] : !reussir.rc<!reussir.closure<() -> i32>>) : i32
