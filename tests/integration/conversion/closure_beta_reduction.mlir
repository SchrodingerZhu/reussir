// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(reussir-closure-beta-reduction))' \
// RUN: | %FileCheck %s

// Closure beta reduction. The pass walks each eval's operand up the def-use
// chain: single-use applies fold into the eval's `with` pack; a uniqify is
// stripped only when it is a provable runtime no-op (its operand is another
// single-use apply/create link of the chain); the bottom-most uniqify — the
// genuine guard of the first application — is kept and becomes the fused
// eval's operand. A chain bottoming at a single-use inlined create is a
// visible beta redex and is inlined outright, erasing every intermediate
// operation including the create's token.

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Full redex: create → uniqify → apply → apply → eval, all single-use.
  // Everything collapses to the body arithmetic; the token dies with the
  // create.
  func.func @redex(%x: i32, %y: i32) -> i32 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%a : i32, %b : i32):
          %s = arith.addi %a, %b : i32
          reussir.closure.yield %s : i32
      }
    }
    %u = reussir.closure.uniqify (%c : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>
    %c1 = reussir.closure.apply (%x : i32) to (%u : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %c2 = reussir.closure.apply (%y : i32) to (%c1 : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %r = reussir.closure.eval (%c2 : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    return %r : i32
  }

  // Chains merged by inlining: the intermediate uniqify (over the single-use
  // apply result, unique by construction) is removed; the bottom one (over
  // the possibly-shared argument) survives as the fused eval's operand.
  func.func @chain(%f: !reussir.rc<!reussir.closure<(i32, i32) -> i32>>, %x: i32, %y: i32) -> i32 {
    %u1 = reussir.closure.uniqify (%f : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>
    %c1 = reussir.closure.apply (%x : i32) to (%u1 : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %u2 = reussir.closure.uniqify (%c1 : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %c2 = reussir.closure.apply (%y : i32) to (%u2 : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %r = reussir.closure.eval (%c2 : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    return %r : i32
  }

  // A lone frontend-shaped uniqify/apply/eval unit: the apply folds into
  // the fused form, but the uniqify is the genuine guard of the first
  // application (its operand is a possibly-shared function argument) and
  // must survive as the fused eval's operand.
  func.func @unit(%f: !reussir.rc<!reussir.closure<(i32) -> i32>>, %x: i32) -> i32 {
    %u = reussir.closure.uniqify (%f : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    %c = reussir.closure.apply (%x : i32) to (%u : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %r = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @redex
// CHECK-NOT: reussir.token.alloc
// CHECK-NOT: reussir.closure
// CHECK: %[[S:[0-9a-z]+]] = arith.addi %arg0, %arg1 : i32
// CHECK-NOT: reussir.closure
// CHECK: return %[[S]] : i32

// CHECK-LABEL: func.func @chain
// CHECK: %[[U:[0-9a-z]+]] = reussir.closure.uniqify(%arg0
// CHECK-NOT: reussir.closure.uniqify
// CHECK-NOT: reussir.closure.apply
// CHECK: reussir.closure.eval(%[[U]] : {{.*}}) with(%arg1, %arg2 : i32, i32) : i32

// CHECK-LABEL: func.func @unit
// CHECK: %[[G:[0-9a-z]+]] = reussir.closure.uniqify(%arg0
// CHECK-NOT: reussir.closure.apply
// CHECK: reussir.closure.eval(%[[G]] : {{.*}}) with(%arg1 : i32) : i32
