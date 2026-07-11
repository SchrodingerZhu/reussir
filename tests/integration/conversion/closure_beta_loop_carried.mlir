// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(reussir-closure-beta-reduction))' \
// RUN: | %FileCheck %s

// The loop-carried beta redex (LOOPINLINE): a closure built once outside a
// loop and called once per iteration — per-iteration `rc.inc` + `eval`
// inside, one `rc.dec` after — inlines into the loop body and the box
// disappears. Loops are matched through `LoopLikeOpInterface`, so `scf.for`
// and `affine.for` reduce alike. Capture counts are preserved: an rc-typed
// capture gains a per-iteration inc at the splice point and a dec where the
// box's dec stood.

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // A fold-shaped loop: the closure body lands inside the loop, every
  // closure op and the token vanish.
  func.func @fold(%n: index) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          %s = arith.addi %a, %a : i64
          reussir.closure.yield %s : i64
      }
    }
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %lb to %n step %step iter_args(%acc = %zero) -> (i64) {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %y = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      %acc2 = arith.addi %acc, %y : i64
      scf.yield %acc2 : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %r : i64
  }

  // A rank-2 nest: the redex sits two loops out; both levels are loop-like,
  // so it still fuses.
  func.func @rank2(%n: index) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          %s = arith.muli %a, %a : i64
          reussir.closure.yield %s : i64
      }
    }
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %lb to %n step %step iter_args(%acc = %zero) -> (i64) {
      %inner = scf.for %j = %lb to %n step %step iter_args(%iacc = %acc) -> (i64) {
        reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
        %iv = arith.index_cast %j : index to i64
        %y = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
        %iacc2 = arith.addi %iacc, %y : i64
        scf.yield %iacc2 : i64
      }
      scf.yield %inner : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %r : i64
  }

  // The same shape under `affine.for`: nothing in the pattern is scf-specific.
  func.func @affine_loop() -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%a : i64):
          %s = arith.addi %a, %a : i64
          reussir.closure.yield %s : i64
      }
    }
    %zero = arith.constant 0 : i64
    %r = affine.for %i = 0 to 8 iter_args(%acc = %zero) -> (i64) {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %y = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      %acc2 = arith.addi %acc, %y : i64
      affine.yield %acc2 : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %r : i64
  }

  // An rc-typed capture bound by an apply outside the loop: the inlined
  // body consumes it per iteration, so it gains a per-iteration inc at the
  // splice point and a dec where the closure's dec stood.
  func.func @rc_capture(%box: !reussir.rc<i64>, %n: index) {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64)>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%b : !reussir.rc<i64>, %a : i64):
          reussir.rc.dec (%b : !reussir.rc<i64>)
          reussir.closure.yield
      }
    }
    %c1 = reussir.closure.apply (%box : !reussir.rc<i64>) to (%c : !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64)>>) : !reussir.rc<!reussir.closure<(i64)>>
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %i = %lb to %n step %step {
      reussir.rc.inc (%c1 : !reussir.rc<!reussir.closure<(i64)>>)
      %iv = arith.index_cast %i : index to i64
      reussir.closure.eval (%c1 : !reussir.rc<!reussir.closure<(i64)>>) with (%iv : i64)
    }
    reussir.rc.dec (%c1 : !reussir.rc<!reussir.closure<(i64)>>)
    return
  }

  // The general frontend shape: a capture bound outside the loop through
  // its own uniqify+apply — rc arguments are owned by the callee, so with
  // no later use the apply consumes %box's original ref directly — and a
  // genuine per-iteration uniqify+apply+eval chain over the shared box
  // inside the loop. The per-iteration uniqify clones; the clone is
  // invisible to the fused form, which reproduces its capture accounting
  // with the splice-point inc and the trailing dec.
  func.func @general(%box: !reussir.rc<i64>, %n: index) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 48>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64, i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 48>)
      body {
        ^bb0(%b : !reussir.rc<i64>, %x : i64, %a : i64):
          reussir.rc.dec (%b : !reussir.rc<i64>)
          %s = arith.muli %x, %a : i64
          reussir.closure.yield %s : i64
      }
    }
    %u0 = reussir.closure.uniqify (%c : !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64, i64) -> i64>>
    %c1 = reussir.closure.apply (%box : !reussir.rc<i64>) to (%u0 : !reussir.rc<!reussir.closure<(!reussir.rc<i64>, i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %lb to %n step %step iter_args(%acc = %zero) -> (i64) {
      reussir.rc.inc (%c1 : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %u1 = reussir.closure.uniqify (%c1 : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>
      %b1 = reussir.closure.apply (%iv : i64) to (%u1 : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
      %y = reussir.closure.eval (%b1 : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      %acc2 = arith.addi %acc, %y : i64
      scf.yield %acc2 : i64
    }
    reussir.rc.dec (%c1 : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>)
    return %r : i64
  }

  // A capture bound INSIDE the loop: a per-iteration apply on the hoisted
  // create. LOOPINLINE alone cannot fire (the eval's operand is loop-local);
  // it composes with FOLD, which folds the apply into the eval's pack, after
  // which the chain is the bare create and the loop-varying capture rides
  // the substituted pack.
  func.func @loop_varying_capture(%n: index) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %c = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64, i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%cap : i64, %a : i64):
          %s = arith.muli %cap, %a : i64
          reussir.closure.yield %s : i64
      }
    }
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %lb to %n step %step iter_args(%acc = %zero) -> (i64) {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %bound = reussir.closure.apply (%iv : i64) to (%c : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
      %y = reussir.closure.eval (%bound : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      %acc2 = arith.addi %acc, %y : i64
      scf.yield %acc2 : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64, i64) -> i64>>)
    return %r : i64
  }

  // NEGATIVE: a second eval of the same closure in the loop — the use set
  // is not the carried-redex shape; everything stays.
  func.func @two_evals(%n: index) {
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
    scf.for %i = %lb to %n step %step {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %y1 = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
      %y2 = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
    }
    reussir.rc.dec (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return
  }

  // NEGATIVE: the closure escapes through a call — not a self-contained
  // use set; everything stays.
  func.func private @sink(!reussir.rc<!reussir.closure<(i64) -> i64>>)
  func.func @escapes(%n: index) {
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
    scf.for %i = %lb to %n step %step {
      reussir.rc.inc (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %iv = arith.index_cast %i : index to i64
      %y = reussir.closure.eval (%c : !reussir.rc<!reussir.closure<(i64) -> i64>>) with (%iv : i64) : i64
    }
    func.call @sink(%c) : (!reussir.rc<!reussir.closure<(i64) -> i64>>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @fold
// CHECK-NOT: reussir.token.alloc
// CHECK-NOT: reussir.closure
// CHECK: scf.for {{.*}} iter_args(%[[ACC:[0-9a-z_]+]] = %{{[0-9a-z_]+}})
// CHECK-NOT: reussir.rc.inc
// CHECK: %[[IV:[0-9a-z]+]] = arith.index_cast
// CHECK: %[[S:[0-9a-z]+]] = arith.addi %[[IV]], %[[IV]] : i64
// CHECK: %[[A2:[0-9a-z]+]] = arith.addi %[[ACC]], %[[S]] : i64
// CHECK: scf.yield %[[A2]] : i64
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @rank2
// CHECK-NOT: reussir.closure
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @affine_loop
// CHECK-NOT: reussir.closure
// CHECK: affine.for
// CHECK: arith.addi
// CHECK: affine.yield
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @rc_capture
// CHECK-NOT: reussir.closure
// CHECK: scf.for
// CHECK: reussir.rc.inc(%arg0 : !reussir.rc<i64>)
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)
// CHECK: }
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)

// CHECK-LABEL: func.func @general
// CHECK-NOT: reussir.closure
// CHECK-NOT: reussir.rc.inc
// CHECK: scf.for
// CHECK: reussir.rc.inc(%arg0 : !reussir.rc<i64>)
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)
// CHECK: arith.muli
// CHECK: scf.yield
// CHECK: }
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)

// CHECK-LABEL: func.func @loop_varying_capture
// CHECK-NOT: reussir.closure
// CHECK: scf.for
// CHECK: %[[LV:[0-9a-z_]+]] = arith.index_cast
// CHECK: arith.muli %[[LV]], %[[LV]] : i64
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @two_evals
// CHECK: reussir.closure.create
// CHECK: scf.for
// CHECK-COUNT-2: reussir.closure.eval
// CHECK: reussir.rc.dec

// CHECK-LABEL: func.func @escapes
// CHECK: reussir.closure.create
// CHECK: reussir.closure.eval
// CHECK: call @sink
