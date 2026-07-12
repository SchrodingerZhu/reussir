// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(reussir-closure-beta-reduction))' \
// RUN: | %FileCheck %s

// Regression test for issue #396. An identity closure yields one of its block
// arguments directly. `inlineBlockBefore` replaces that argument and destroys
// the source block, so both the direct and loop-carried beta reductions must
// use the substituted value rather than retain the old block argument.

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @direct_identity(%x: i64) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          reussir.closure.yield %a : i64
      }
    }
    %r = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%x : i64) : i64
    return %r : i64
  }

  func.func @loop_identity(%n: index) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          reussir.closure.yield %a : i64
      }
    }
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %lb to %n step %step iter_args(%acc = %zero) -> (i64) {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %next = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      scf.yield %next : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %r : i64
  }
}

// CHECK-LABEL: func.func @direct_identity
// CHECK-NOT: reussir
// CHECK: return %arg0 : i64

// CHECK-LABEL: func.func @loop_identity
// CHECK-NOT: reussir
// CHECK: %[[RESULT:[0-9a-z_]+]] = scf.for
// CHECK: %[[IV:[0-9a-z_]+]] = arith.index_cast
// CHECK-NEXT: scf.yield %[[IV]] : i64
// CHECK-NOT: reussir
// CHECK: return %[[RESULT]] : i64
