// REQUIRES: asan
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-lowering-scf-ops,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-lowering-scf-ops,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %cc %asan_flags -c -x ir %t.ll -o %t.o
// RUN: %cc %asan_flags %t.o %reussir_rt_asan -o %t.exe \
// RUN:   %rpath_flag %rpath_san_flag %extra_sys_libs
// RUN: %asan_env %t.exe

// This one lifecycle catches all Cell ownership edges with ASan+LSan:
// - get must retain the nested Rc before its returned clone is released;
// - set must release the old nested Rc but must not retain the incoming one;
// - rmw must move old/new values without either retain or release;
// - the final cell decrement must recursively release its nested Rc.

!inner = !reussir.rc<i64>
!cell = !reussir.rc<!reussir.cell<!inner>>
!scalar_cell = !reussir.rc<!reussir.cell<i64>>

module attributes {reussir.sanitize = ["sanitize_address"]} {
  func.func private @read_inner(%value: !inner) -> i64
      attributes {llvm.linkage = #llvm.linkage<internal>,
                  passthrough = ["sanitize_address"]} {
    %ref = reussir.rc.borrow(%value : !inner) : !reussir.ref<i64>
    %loaded = reussir.ref.load(%ref : !reussir.ref<i64>) : i64
    return %loaded : i64
  }

  func.func private @require_equal(%actual: i64, %expected: i64)
      attributes {llvm.linkage = #llvm.linkage<internal>,
                  passthrough = ["sanitize_address"]} {
    %wrong = arith.cmpi ne, %actual, %expected : i64
    scf.if %wrong {
      reussir.panic "cell value mismatch"
    }
    return
  }

  func.func @main() -> i32 attributes {passthrough = ["sanitize_address"]} {
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i64
    %c6 = arith.constant 6 : i64
    %c9 = arith.constant 9 : i64
    %c11 = arith.constant 11 : i64
    %c22 = arith.constant 22 : i64
    %c33 = arith.constant 33 : i64
    %c44 = arith.constant 44 : i64

    %v1 = reussir.rc.create value(%c11 : i64) : !inner
    %cell = reussir.cell.create value(%v1 : !inner) : !cell

    // A returned get value owns a clone independent of the cell.
    %clone1 = reussir.cell.get(%cell : !cell) : !inner
    %got11 = func.call @read_inner(%clone1) : (!inner) -> i64
    func.call @require_equal(%got11, %c11) : (i64, i64) -> ()
    reussir.rc.dec(%clone1 : !inner)

    // set drops v1 and transfers v2 into the slot without retaining v2.
    %v2 = reussir.rc.create value(%c22 : i64) : !inner
    reussir.cell.set(%v2 : !inner, %cell : !cell)
    %clone2 = reussir.cell.get(%cell : !cell) : !inner
    %got22 = func.call @read_inner(%clone2) : (!inner) -> i64
    func.call @require_equal(%got22, %c22) : (i64, i64) -> ()
    reussir.rc.dec(%clone2 : !inner)

    // rmw moves v2 out and v3 in. Returning the old value must not retain it.
    %v3 = reussir.rc.create value(%c33 : i64) : !inner
    %old = reussir.cell.rmw(%cell : !cell) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%v3 : !inner) output(%current : !inner)
    }
    %old22 = func.call @read_inner(%old) : (!inner) -> i64
    func.call @require_equal(%old22, %c22) : (i64, i64) -> ()
    reussir.rc.dec(%old : !inner)

    %clone3 = reussir.cell.get(%cell : !cell) : !inner
    %got33 = func.call @read_inner(%clone3) : (!inner) -> i64
    func.call @require_equal(%got33, %c33) : (i64, i64) -> ()
    reussir.rc.dec(%clone3 : !inner)

    // Replacing v3 checks another old-value drop. The cell's final decrement
    // must recursively release v4.
    %v4 = reussir.rc.create value(%c44 : i64) : !inner
    reussir.cell.set(%v4 : !inner, %cell : !cell)
    reussir.rc.dec(%cell : !cell)

    // Also execute both rmw and ordinary get/set for a trivial payload.
    %scalar = reussir.cell.create value(%c5 : i64) : !scalar_cell
    %before = reussir.cell.rmw(%scalar : !scalar_cell) -> i64 {
      ^bb0(%current: i64):
        %one = arith.constant 1 : i64
        %next = arith.addi %current, %one : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    func.call @require_equal(%before, %c5) : (i64, i64) -> ()
    %after = reussir.cell.get(%scalar : !scalar_cell) : i64
    func.call @require_equal(%after, %c6) : (i64, i64) -> ()
    reussir.cell.set(%c9 : i64, %scalar : !scalar_cell)
    %set_value = reussir.cell.get(%scalar : !scalar_cell) : i64
    func.call @require_equal(%set_value, %c9) : (i64, i64) -> ()
    reussir.rc.dec(%scalar : !scalar_cell)

    return %c0_i32 : i32
  }
}
