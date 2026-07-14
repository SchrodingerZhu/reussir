// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

!inner = !reussir.rc<i64>
!cell = !reussir.cell<!inner>
!rc_cell = !reussir.rc<!cell>

module {
  func.func @cell_ops(%initial: !inner, %replacement: !inner) -> i64 {
    %cell = reussir.cell.create value(%initial : !inner) : !rc_cell
    %cloned = reussir.cell.get(%cell : !rc_cell) : !inner
    reussir.cell.set(%replacement : !inner, %cell : !rc_cell)
    %answer = reussir.cell.rmw(%cell : !rc_cell) -> i64 {
      ^bb0(%old: !inner):
        %c42 = arith.constant 42 : i64
        reussir.cell.yield(%old : !inner) output(%c42 : i64)
    }
    reussir.cell.rmw(%cell : !rc_cell) {
      ^bb0(%old: !inner):
        reussir.cell.yield(%old : !inner)
    }
    reussir.rc.dec(%cloned : !inner)
    reussir.rc.dec(%cell : !rc_cell)
    return %answer : i64
  }
}

// CHECK-LABEL: func.func @cell_ops
// CHECK: %[[CELL_VALUE:.+]] = reussir.cell.create value(%{{.+}} : !reussir.rc<i64>) : !reussir.rc<!reussir.cell<!reussir.rc<i64>>>
// CHECK: %{{.+}} = reussir.cell.get(%[[CELL_VALUE]] : !reussir.rc<!reussir.cell<!reussir.rc<i64>>>) : !reussir.rc<i64>
// CHECK: reussir.cell.set(%{{.+}} : !reussir.rc<i64>, %[[CELL_VALUE]] : !reussir.rc<!reussir.cell<!reussir.rc<i64>>>)
// CHECK: reussir.cell.rmw(%[[CELL_VALUE]] : !reussir.rc<!reussir.cell<!reussir.rc<i64>>>) -> i64
// CHECK: reussir.cell.yield(%{{.+}} : !reussir.rc<i64>) output(%{{.+}} : i64)
// CHECK: reussir.cell.rmw(%[[CELL_VALUE]] : !reussir.rc<!reussir.cell<!reussir.rc<i64>>>)
// CHECK: reussir.cell.yield(%{{.+}} : !reussir.rc<i64>)
