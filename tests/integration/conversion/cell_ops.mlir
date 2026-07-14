// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation))' | %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=EXPAND
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion --reussir-invariant-group-analysis | %FileCheck %s --check-prefix=INVARIANT
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion)' | %FileCheck %s --check-prefix=DROP
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-lowering-scf-ops,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},reussir-lowering-scf-ops,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64>
!rc_cell = !reussir.rc<!reussir.cell<!inner>>
!rc_i64_cell = !reussir.rc<!reussir.cell<i64>>
!holder = !reussir.record<compound "CellHolder" [value] {
  !reussir.cell<i64>
}>

module {
  // TOKEN-LABEL: func.func @create_cell
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 16>
  // TOKEN: reussir.cell.create value(%{{.+}} : !reussir.rc<i64>) token(%[[TOKEN]] : !reussir.token<align : 8, size : 16>)
  func.func @create_cell(%value: !inner) -> !rc_cell {
    %cell = reussir.cell.create value(%value : !inner) : !rc_cell
    return %cell : !rc_cell
  }

  // TOKEN-LABEL: func.func @drop_cell
  // TOKEN: reussir.rc.dec{{.*}}: !reussir.nullable<!reussir.token<align : 8, size : 16>>
  // DROP-LABEL: func.func @drop_cell
  // DROP: %[[CELL_REF:.+]] = reussir.rc.borrow{{.*}}!reussir.ref<!reussir.cell<!reussir.rc<i64>>>
  // DROP: %[[INNER_SLOT:.+]] = reussir.ref.project(%[[CELL_REF]]
  // DROP: %[[INNER:.+]] = reussir.ref.load(%[[INNER_SLOT]]
  // DROP: reussir.rc.dec(%[[INNER]] : !reussir.rc<i64>)
  func.func @drop_cell(%cell: !rc_cell) {
    reussir.rc.dec(%cell : !rc_cell)
    return
  }

  // EXPAND-LABEL: func.func @get_cell
  // EXPAND: %[[BORROW:.+]] = reussir.rc.borrow
  // EXPAND: %[[SLOT:.+]] = reussir.ref.project(%[[BORROW]]
  // EXPAND: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // EXPAND-NEXT: reussir.rc.inc(%[[VALUE]]
  // EXPAND-NOT: reussir.ref.load
  // EXPAND: return %[[VALUE]]
  // INVARIANT-LABEL: func.func @get_cell
  // INVARIANT: reussir.ref.load
  // INVARIANT-NOT: invariant_group
  // INVARIANT: reussir.rc.inc
  func.func @get_cell(%cell: !rc_cell) -> !inner {
    %value = reussir.cell.get(%cell : !rc_cell) : !inner
    return %value : !inner
  }

  // EXPAND-LABEL: func.func @set_cell
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND-NEXT: %{{.+}} = reussir.rc.dec(%[[OLD]]
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND: reussir.ref.store
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND: return
  func.func @set_cell(%value: !inner, %cell: !rc_cell) {
    reussir.cell.set(%value : !inner, %cell : !rc_cell)
    return
  }

  // EXPAND-LABEL: func.func @rmw_cell
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: reussir.ref.store
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: return %[[OLD]]
  func.func @rmw_cell(%value: !inner, %cell: !rc_cell) -> !inner {
    %old = reussir.cell.rmw(%cell : !rc_cell) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%value : !inner) output(%current : !inner)
    }
    return %old : !inner
  }

  // EXPAND-LABEL: func.func @rmw_i64
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: %[[NEXT:.+]] = arith.addi %[[OLD]]
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: reussir.ref.store{{.*}}(%[[NEXT]] : i64)
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: return
  func.func @rmw_i64(%cell: !rc_i64_cell) -> i64 {
    %old = reussir.cell.rmw(%cell : !rc_i64_cell) -> i64 {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  // A non-cell borrow remains eligible for invariant.group, proving that the
  // negative cell checks above are caused by the mutable projection boundary.
  // INVARIANT-LABEL: func.func @control_invariant
  // INVARIANT: reussir.ref.load{{.*}}invariant_group
  func.func @control_invariant(%value: !inner) -> i64 {
    %ref = reussir.rc.borrow(%value : !inner) : !reussir.ref<i64>
    %loaded = reussir.ref.load(%ref : !reussir.ref<i64>) : i64
    return %loaded : i64
  }

  // A raw Cell member occupies one shared pointer slot and projects as
  // Rc<Cell>, matching closure/array member materialization.
  // LLVM: %CellHolder = type { ptr }
  // LLVM-LABEL: define ptr @project_cell_member(ptr %{{.+}})
  // LLVM: getelementptr %CellHolder, ptr %{{.+}}, i32 0, i32 0
  // LLVM: load ptr
  func.func @project_cell_member(%holder: !reussir.ref<!holder>) -> !rc_i64_cell {
    %slot = reussir.ref.project(%holder : !reussir.ref<!holder>) [0]
      : !reussir.ref<!rc_i64_cell>
    %cell = reussir.ref.load(%slot : !reussir.ref<!rc_i64_cell>) : !rc_i64_cell
    return %cell : !rc_i64_cell
  }
}
