// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation))' | %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=EXPAND
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion --reussir-invariant-group-analysis | %FileCheck %s --check-prefix=INVARIANT
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion)' | %FileCheck %s --check-prefix=DROP
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-lowering-scf-ops,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},reussir-lowering-scf-ops,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64>
!nullable_inner = !reussir.nullable<!inner>
!rc_cell = !reussir.rc<!reussir.cell<!inner>>
!rc_i64_cell = !reussir.rc<!reussir.cell<i64>>
!rc_nullable_cell = !reussir.rc<!reussir.cell<!nullable_inner>>
!rc_excl_cell = !reussir.rc<!reussir.cell<!inner exclusive>>
!rc_i64_excl_cell = !reussir.rc<!reussir.cell<i64 exclusive>>
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

  // An exclusive cell box carries the trailing in-use flag: the payload
  // widens from 8 to 16 bytes (element + flag padded to alignment), so the
  // box token is 8 bytes larger. The flag starts out false.
  // TOKEN-LABEL: func.func @create_exclusive_cell
  // TOKEN: %[[TOKEN:.+]] = reussir.token.alloc : <align : 8, size : 24>
  // TOKEN: reussir.cell.create value(%{{.+}} : !reussir.rc<i64>) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>)
  // EXPAND-LABEL: func.func @create_exclusive_cell
  // EXPAND: %[[FALSE:.+]] = arith.constant false
  // EXPAND: reussir.ref.store
  // EXPAND: %[[FLAG:.+]] = reussir.ref.project{{.*}}[1] : !reussir.ref<i1 field>
  // EXPAND: reussir.ref.store(%[[FLAG]] : !reussir.ref<i1 field>) (%[[FALSE]] : i1)
  func.func @create_exclusive_cell(%value: !inner) -> !rc_excl_cell {
    %cell = reussir.cell.create value(%value : !inner) : !rc_excl_cell
    return %cell : !rc_excl_cell
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
  // EXPAND-NOT: reussir.ref.project{{.*}}[1]
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

  // A non-trivial, non-RC element (here a nullable RC pointer) must also be
  // acquired on get: the clone dispatches on nullness and retains the wrapped
  // box on the nonnull path.
  // EXPAND-LABEL: func.func @get_nullable_cell
  // EXPAND: %[[VALUE:.+]] = reussir.ref.load
  // EXPAND: %[[SPILL:.+]] = reussir.ref.spilled(%[[VALUE]]
  // EXPAND: %[[RELOAD:.+]] = reussir.ref.load(%[[SPILL]]
  // EXPAND: reussir.nullable.dispatch(%[[RELOAD]]
  // EXPAND: ^{{.+}}(%[[NONNULL:.+]]: !reussir.rc<i64>):
  // EXPAND: reussir.rc.inc(%[[NONNULL]]
  // EXPAND: return %[[VALUE]]
  func.func @get_nullable_cell(%cell: !rc_nullable_cell) -> !nullable_inner {
    %value = reussir.cell.get(%cell : !rc_nullable_cell) : !nullable_inner
    return %value : !nullable_inner
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

  // Every payload access to an exclusive cell lowers to a branch on the
  // in-use flag: the in-use arm panics and yields poison, the free arm
  // performs the access.
  // EXPAND-LABEL: func.func @get_exclusive_cell
  // EXPAND: %[[POISON:.+]] = ub.poison : !reussir.rc<i64>
  // EXPAND: %[[FLAG:.+]] = reussir.ref.project{{.*}}[1] : !reussir.ref<i1 field>
  // EXPAND: %[[INUSE:.+]] = reussir.ref.load(%[[FLAG]]
  // EXPAND: %[[EXPECT:.+]] = reussir.expect(%[[INUSE]] : i1, false)
  // EXPAND: %[[RESULT:.+]] = scf.if %[[EXPECT]] -> (!reussir.rc<i64>)
  // EXPAND: reussir.panic "cell.get on an exclusive cell that is in use"
  // EXPAND: scf.yield %[[POISON]]
  // EXPAND: } else {
  // EXPAND: %[[VALUE:.+]] = reussir.ref.load
  // EXPAND: reussir.rc.inc(%[[VALUE]]
  // EXPAND: scf.yield %[[VALUE]]
  // EXPAND: return %[[RESULT]]
  func.func @get_exclusive_cell(%cell: !rc_excl_cell) -> !inner {
    %value = reussir.cell.get(%cell : !rc_excl_cell) : !inner
    return %value : !inner
  }

  // EXPAND-LABEL: func.func @set_exclusive_cell
  // EXPAND: %[[EXPECT:.+]] = reussir.expect
  // EXPAND: scf.if %[[EXPECT]]
  // EXPAND: reussir.panic "cell.set on an exclusive cell that is in use"
  // EXPAND: } else {
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND: reussir.rc.dec(%[[OLD]]
  // EXPAND: reussir.ref.store
  func.func @set_exclusive_cell(%value: !inner, %cell: !rc_excl_cell) {
    reussir.cell.set(%value : !inner, %cell : !rc_excl_cell)
    return
  }

  // The whole read-modify-write is the free arm of the flag branch: raise
  // the flag, move the element through the body, clear the flag with the
  // replacement store. The moved element sees no retain or release, and the
  // panicked arm yields poison.
  // EXPAND-LABEL: func.func @rmw_cell
  // EXPAND-DAG: %[[FALSE:.+]] = arith.constant false
  // EXPAND-DAG: %[[TRUE:.+]] = arith.constant true
  // EXPAND-DAG: %[[POISON:.+]] = ub.poison : !reussir.rc<i64>
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: %[[RESULT:.+]] = scf.if %{{.+}} -> (!reussir.rc<i64>)
  // EXPAND: reussir.panic "cell.rmw on an exclusive cell that is in use"
  // EXPAND: scf.yield %[[POISON]]
  // EXPAND: } else {
  // EXPAND: reussir.ref.store(%{{.+}} : !reussir.ref<i1 field>) (%[[TRUE]] : i1)
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: reussir.ref.store
  // EXPAND: reussir.ref.store(%{{.+}} : !reussir.ref<i1 field>) (%[[FALSE]] : i1)
  // EXPAND: scf.yield %[[OLD]]
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: return %[[RESULT]]
  func.func @rmw_cell(%value: !inner, %cell: !rc_excl_cell) -> !inner {
    %old = reussir.cell.rmw(%cell : !rc_excl_cell) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%value : !inner) output(%current : !inner)
    }
    return %old : !inner
  }

  // EXPAND-LABEL: func.func @rmw_i64
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: reussir.ref.store{{.*}}(%{{.+}} : i1)
  // EXPAND: %[[OLD:.+]] = reussir.ref.load
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: %[[NEXT:.+]] = arith.addi %[[OLD]]
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: reussir.ref.store{{.*}}(%[[NEXT]] : i64)
  // EXPAND-NOT: reussir.rc.inc
  // EXPAND-NOT: reussir.rc.dec
  // EXPAND: scf.yield %[[OLD]]
  // EXPAND: return
  func.func @rmw_i64(%cell: !rc_i64_excl_cell) -> i64 {
    %old = reussir.cell.rmw(%cell : !rc_i64_excl_cell) -> i64 {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  // EXPAND-LABEL: func.func @in_use
  // EXPAND: %[[FLAG:.+]] = reussir.ref.project{{.*}}[1] : !reussir.ref<i1 field>
  // EXPAND: %[[LOADED:.+]] = reussir.ref.load(%[[FLAG]] : !reussir.ref<i1 field>) : i1
  // EXPAND: return %[[LOADED]]
  func.func @in_use(%cell: !rc_excl_cell) -> i1 {
    %flag = reussir.cell.in_use(%cell : !rc_excl_cell) : i1
    return %flag : i1
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

  // The exclusive flag lives at struct index 1: the guarded fast path loads
  // it, and the box payload lowers as {element, i1}.
  // LLVM-LABEL: define i1 @in_use_lowering(ptr %[[CELL:.+]])
  // LLVM: %[[FLAG_PTR:.+]] = getelementptr { ptr, i1 }, ptr %{{.+}}, i32 0, i32 1
  // LLVM: %[[FLAG:.+]] = load i1, ptr %[[FLAG_PTR]]
  // LLVM: ret i1 %[[FLAG]]
  func.func @in_use_lowering(%cell: !rc_excl_cell) -> i1 {
    %flag = reussir.cell.in_use(%cell : !rc_excl_cell) : i1
    return %flag : i1
  }
}
