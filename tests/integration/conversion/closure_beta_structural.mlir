// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(reussir-closure-beta-reduction))' \
// RUN: | %FileCheck %s

// Structural beta reduction accepts exactly LoopLike operations and
// reussir.cell.rmw as region boundaries. The existing loop tests cover pure
// loop nests; this file covers rmw alone, rmw nested under a loop, ownership
// accounting for an rc capture, and rejection of an unrelated region op.

!rc_cell = !reussir.rc<!reussir.cell<i64 exclusive>>
!rc_i64 = !reussir.rc<i64>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @cell_rmw(%cell: !rc_cell) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%value: i64):
          %next = arith.addi %value, %value : i64
          reussir.closure.yield %next : i64
      }
    }
    %old = reussir.cell.rmw(%cell : !rc_cell) -> i64 {
      ^bb0(%current: i64):
        reussir.rc.inc(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
        %unique = reussir.closure.uniqify(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
        %bound = reussir.closure.apply(%current : i64) to (%unique : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<() -> i64>>
        %next = reussir.closure.eval(%bound : !reussir.rc<!reussir.closure<() -> i64>>) : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    reussir.rc.dec(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %old : i64
  }

  func.func @cell_rmw_rc_capture(%capture: !rc_i64, %cell: !rc_cell) -> i64 {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 40>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(!rc_i64, i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 40>)
      body {
        ^bb0(%owned: !rc_i64, %value: i64):
          reussir.rc.dec(%owned : !rc_i64)
          %next = arith.addi %value, %value : i64
          reussir.closure.yield %next : i64
      }
    }
    %captured = reussir.closure.apply(%capture : !rc_i64) to (%closure : !reussir.rc<!reussir.closure<(!rc_i64, i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
    %old = reussir.cell.rmw(%cell : !rc_cell) -> i64 {
      ^bb0(%current: i64):
        reussir.rc.inc(%captured : !reussir.rc<!reussir.closure<(i64) -> i64>>)
        %unique = reussir.closure.uniqify(%captured : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
        %bound = reussir.closure.apply(%current : i64) to (%unique : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<() -> i64>>
        %next = reussir.closure.eval(%bound : !reussir.rc<!reussir.closure<() -> i64>>) : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    reussir.rc.dec(%captured : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return %old : i64
  }

  func.func @loop_and_cell_rmw(%cell: !rc_cell, %n: index) {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%value: i64):
          %next = arith.addi %value, %value : i64
          reussir.closure.yield %next : i64
      }
    }
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %i = %lb to %n step %step {
      reussir.cell.rmw(%cell : !rc_cell) {
        ^bb0(%current: i64):
          reussir.rc.inc(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
          %unique = reussir.closure.uniqify(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
          %bound = reussir.closure.apply(%current : i64) to (%unique : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<() -> i64>>
          %next = reussir.closure.eval(%bound : !reussir.rc<!reussir.closure<() -> i64>>) : i64
          reussir.cell.yield(%next : i64)
      }
    }
    reussir.rc.dec(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return
  }

  // scf.if is intentionally not part of the narrow structural set.
  func.func @unsupported_if(%cond: i1) {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%value: i64):
          reussir.closure.yield %value : i64
      }
    }
    scf.if %cond {
      %zero = arith.constant 0 : i64
      reussir.rc.inc(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
      %unique = reussir.closure.uniqify(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<(i64) -> i64>>
      %bound = reussir.closure.apply(%zero : i64) to (%unique : !reussir.rc<!reussir.closure<(i64) -> i64>>) : !reussir.rc<!reussir.closure<() -> i64>>
      %result = reussir.closure.eval(%bound : !reussir.rc<!reussir.closure<() -> i64>>) : i64
    }
    reussir.rc.dec(%closure : !reussir.rc<!reussir.closure<(i64) -> i64>>)
    return
  }
}

// CHECK-LABEL: func.func @cell_rmw
// CHECK-NOT: reussir.token.alloc
// CHECK-NOT: reussir.closure
// CHECK-NOT: reussir.rc.inc
// CHECK: reussir.cell.rmw
// CHECK: %[[NEXT:.+]] = arith.addi %[[CURRENT:.+]], %[[CURRENT]] : i64
// CHECK: reussir.cell.yield(%[[NEXT]] : i64) output(%[[CURRENT]] : i64)
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @cell_rmw_rc_capture
// CHECK-NOT: reussir.closure
// CHECK: reussir.cell.rmw
// CHECK: reussir.rc.inc(%arg0 : !reussir.rc<i64>)
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)
// CHECK: arith.addi
// CHECK: reussir.cell.yield
// CHECK: }
// CHECK: reussir.rc.dec(%arg0 : !reussir.rc<i64>)

// CHECK-LABEL: func.func @loop_and_cell_rmw
// CHECK-NOT: reussir.closure
// CHECK-NOT: reussir.rc.inc
// CHECK: scf.for
// CHECK: reussir.cell.rmw
// CHECK: arith.addi
// CHECK: reussir.cell.yield
// CHECK-NOT: reussir.rc.dec

// CHECK-LABEL: func.func @unsupported_if
// CHECK: reussir.closure.create
// CHECK: scf.if
// CHECK: reussir.closure.eval
// CHECK: reussir.rc.dec
