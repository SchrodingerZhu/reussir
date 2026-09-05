// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Executable proof of the ownership traversal over a dynamic-extent array of
// managed elements: a rank-2 `array<? x 2 x rc<i32>>` whose drop walks a loop
// nest bounded by `memref.dim` on the leading dim and projects through the
// dynamic strided layout at every rank. The array is shared and updated
// through a unique view, so the clone's acquire traversal retains every
// element once and the two drops release each element exactly twice.
!e = !reussir.rc<i32>
!dv = !reussir.array<? x 2 x !e>
!rc_dv = !reussir.rc<!dv>
!view = memref<?x2x!e, strided<[?, ?], offset: ?>>
!row = memref<2x!e, strided<[?], offset: ?>>
module {
  // Slot [i][j] holds rc(2i + j).
  func.func private @make(%n: index) -> !rc_dv attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !dv
    %rc = reussir.rc.create value(%poison : !dv) extents(%n) : !rc_dv
    %ref = reussir.rc.borrow (%rc : !rc_dv) : !reussir.ref<!dv>
    %v = reussir.array.view(%ref : !reussir.ref<!dv>) : !view
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %n step %c1 {
      %row = reussir.array.project(%v : !view) [%i : index] : !row
      scf.for %j = %c0 to %c2 step %c1 {
        %slot = reussir.array.project(%row : !row) [%j : index] : !reussir.ref<!e field>
        %i2 = arith.muli %i, %c2 : index
        %k = arith.addi %i2, %j : index
        %kv = arith.index_cast %k : index to i32
        %elem = reussir.rc.create value(%kv : i32) : !e
        reussir.ref.store (%slot : !reussir.ref<!e field>) (%elem : !e)
      }
    }
    return %rc : !rc_dv
  }

  func.func private @sum(%rc: !rc_dv) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %ref = reussir.rc.borrow (%rc : !rc_dv) : !reussir.ref<!dv>
    %v = reussir.array.view(%ref : !reussir.ref<!dv>) : !view
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n = memref.dim %v, %c0 : !view
    %c0i = arith.constant 0 : i32
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0i) -> i32 {
      %row = reussir.array.project(%v : !view) [%i : index] : !row
      %s2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%acc2 = %acc) -> i32 {
        %slot = reussir.array.project(%row : !row) [%j : index] : !reussir.ref<!e field>
        %elem = reussir.ref.load (%slot : !reussir.ref<!e field>) : !e
        %eref = reussir.rc.borrow (%elem : !e) : !reussir.ref<i32>
        %x = reussir.ref.load (%eref : !reussir.ref<i32>) : i32
        %a = arith.addi %acc2, %x : i32
        scf.yield %a : i32
      }
      scf.yield %s2 : i32
    }
    return %s : i32
  }

  func.func private @set00(%rc: !rc_dv, %value: !e) -> !rc_dv attributes {llvm.linkage = #llvm.linkage<internal>} {
    %updated = reussir.array.with_unique_view (%rc : !rc_dv) -> !rc_dv {
      ^bb0(%view: !view):
        %c0 = arith.constant 0 : index
        %row = reussir.array.project(%view : !view) [%c0 : index] : !row
        %slot = reussir.array.project(%row : !row) [%c0 : index] : !reussir.ref<!e field>
        // Release the old occupant, then move the new element in.
        %old = reussir.ref.load (%slot : !reussir.ref<!e field>) : !e
        reussir.rc.dec (%old : !e)
        reussir.ref.store (%slot : !reussir.ref<!e field>) (%value : !e)
        reussir.scf.yield
    }
    return %updated : !rc_dv
  }

  func.func @main() -> i32 {
    %c5 = arith.constant 5 : index
    %c100 = arith.constant 100 : i32
    // sum(0..9) = 45; replacing slot [0][0] (value 0) with 100 gives 145.
    %c45 = arith.constant 45 : i32
    %c145 = arith.constant 145 : i32

    %shared = func.call @make(%c5) : (index) -> !rc_dv
    reussir.rc.inc (%shared : !rc_dv)
    %hundred = reussir.rc.create value(%c100 : i32) : !e
    %updated = func.call @set00(%shared, %hundred) : (!rc_dv, !e) -> !rc_dv

    %original_sum = func.call @sum(%shared) : (!rc_dv) -> i32
    %original_bad = arith.cmpi ne, %original_sum, %c45 : i32
    scf.if %original_bad {
      reussir.panic "shared managed dynamic array was mutated through the clone"
    }
    %updated_sum = func.call @sum(%updated) : (!rc_dv) -> i32
    %updated_bad = arith.cmpi ne, %updated_sum, %c145 : i32
    scf.if %updated_bad {
      reussir.panic "managed dynamic array clone did not carry the update"
    }
    reussir.rc.dec (%shared : !rc_dv)
    reussir.rc.dec (%updated : !rc_dv)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
