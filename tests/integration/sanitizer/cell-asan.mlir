// REQUIRES: asan
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %cc %asan_flags -c -x ir %t.ll -o %t.o
// RUN: %cc %asan_flags %t.o %reussir_rt_asan -o %t.exe \
// RUN:   %rpath_flag %rpath_san_flag %extra_sys_libs
// RUN: %asan_env %t.exe

// This one lifecycle catches all Cell ownership edges with ASan+LSan:
// - get must retain the nested Rc before its returned clone is released;
// - get of a non-RC managed element (a nullable RC pointer) must equally
//   retain through the acquisition glue — the case that used to silently
//   lose its +1 and read freed memory;
// - set must release the old nested Rc but must not retain the incoming one;
// - rmw (exclusive cells only) must move old/new values without either
//   retain or release, and must maintain the in-use flag;
// - the final cell decrement must recursively release its nested Rc.

!inner = !reussir.rc<i64>
!nullable_inner = !reussir.nullable<!inner>
!cell = !reussir.rc<!reussir.cell<!inner>>
!excl_cell = !reussir.rc<!reussir.cell<!inner exclusive>>
!nullable_cell = !reussir.rc<!reussir.cell<!nullable_inner>>
!scalar_cell = !reussir.rc<!reussir.cell<i64 exclusive>>

module attributes {reussir.sanitize = ["sanitize_address"]} {
  func.func private @read_inner(%value: !inner) -> i64
      attributes {llvm.linkage = #llvm.linkage<internal>,
                  llvm.passthrough = ["sanitize_address"]} {
    %ref = reussir.rc.borrow(%value : !inner) : !reussir.ref<i64>
    %loaded = reussir.ref.load(%ref : !reussir.ref<i64>) : i64
    return %loaded : i64
  }

  func.func private @require_equal(%actual: i64, %expected: i64)
      attributes {llvm.linkage = #llvm.linkage<internal>,
                  llvm.passthrough = ["sanitize_address"]} {
    %wrong = arith.cmpi ne, %actual, %expected : i64
    scf.if %wrong {
      reussir.panic "cell value mismatch"
    }
    return
  }

  func.func @main() -> i32 attributes {llvm.passthrough = ["sanitize_address"]} {
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i64
    %c6 = arith.constant 6 : i64
    %c9 = arith.constant 9 : i64
    %c11 = arith.constant 11 : i64
    %c22 = arith.constant 22 : i64
    %c33 = arith.constant 33 : i64
    %c44 = arith.constant 44 : i64
    %c55 = arith.constant 55 : i64
    %c66 = arith.constant 66 : i64

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

    // rmw needs an exclusive cell: it moves v3 out and v4 in. Returning the
    // old value must not retain it, and the in-use flag must read false
    // outside the body.
    %v3 = reussir.rc.create value(%c33 : i64) : !inner
    %ecell = reussir.cell.create value(%v3 : !inner) : !excl_cell
    %v4 = reussir.rc.create value(%c44 : i64) : !inner
    %old = reussir.cell.rmw(%ecell : !excl_cell) -> !inner {
      ^bb0(%current: !inner):
        reussir.cell.yield(%v4 : !inner) output(%current : !inner)
    }
    %old33 = func.call @read_inner(%old) : (!inner) -> i64
    func.call @require_equal(%old33, %c33) : (i64, i64) -> ()
    reussir.rc.dec(%old : !inner)

    %idle = reussir.cell.in_use(%ecell : !excl_cell) : i1
    scf.if %idle {
      reussir.panic "exclusive cell must be idle outside rmw"
    }

    %clone3 = reussir.cell.get(%ecell : !excl_cell) : !inner
    %got44 = func.call @read_inner(%clone3) : (!inner) -> i64
    func.call @require_equal(%got44, %c44) : (i64, i64) -> ()
    reussir.rc.dec(%clone3 : !inner)

    // Replacing v2 checks another old-value drop. The cells' final
    // decrements must recursively release their elements.
    %v5 = reussir.rc.create value(%c55 : i64) : !inner
    reussir.cell.set(%v5 : !inner, %cell : !cell)
    reussir.rc.dec(%cell : !cell)
    reussir.rc.dec(%ecell : !excl_cell)

    // A nullable element is managed but not a bare RC pointer: its get-clone
    // must retain through the acquisition glue. Without that +1 the set(null)
    // below frees the box while %nclone still reads it (ASan catches the
    // use-after-free), and the trailing decrements double-release.
    %v6 = reussir.rc.create value(%c66 : i64) : !inner
    %nonnull = reussir.nullable.create (%v6 : !inner) : !nullable_inner
    %ncell = reussir.cell.create value(%nonnull : !nullable_inner) : !nullable_cell
    %nclone = reussir.cell.get(%ncell : !nullable_cell) : !nullable_inner
    %null = reussir.nullable.create : !nullable_inner
    reussir.cell.set(%null : !nullable_inner, %ncell : !nullable_cell)
    %ncoerced = reussir.nullable.coerce (%nclone : !nullable_inner) : !inner
    %got66 = func.call @read_inner(%ncoerced) : (!inner) -> i64
    func.call @require_equal(%got66, %c66) : (i64, i64) -> ()
    reussir.rc.dec(%ncoerced : !inner)
    reussir.rc.dec(%ncell : !nullable_cell)

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
