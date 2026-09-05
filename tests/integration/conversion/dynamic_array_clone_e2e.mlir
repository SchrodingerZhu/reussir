// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Executable proof of the dynamic-extent clone branch of
// `array.with_unique_view` (docs/design/dynamic-extent-arrays.md, "Clone
// compacts"): a shared [0, 1, ..., n-1] is updated through a unique view, so
// the box is cloned with a runtime-sized token and a runtime-length payload
// copy. The source must keep its sum (45) and the clone carry the update
// (145); both boxes are released through their dynamic tokens.
!dv = !reussir.array<? x i32>
!rc_dv = !reussir.rc<!dv>
module {
  func.func private @iota(%n: index) -> !rc_dv attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !dv
    %rc = reussir.rc.create value(%poison : !dv) extents(%n) : !rc_dv
    %ref = reussir.rc.borrow (%rc : !rc_dv) : !reussir.ref<!dv>
    %v = reussir.array.view(%ref : !reussir.ref<!dv>) : memref<?xi32, strided<[?], offset: ?>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %iv = arith.index_cast %i : index to i32
      memref.store %iv, %v[%i] : memref<?xi32, strided<[?], offset: ?>>
    }
    return %rc : !rc_dv
  }

  func.func private @sum(%rc: !rc_dv) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %ref = reussir.rc.borrow (%rc : !rc_dv) : !reussir.ref<!dv>
    %v = reussir.array.view(%ref : !reussir.ref<!dv>) : memref<?xi32, strided<[?], offset: ?>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = memref.dim %v, %c0 : memref<?xi32, strided<[?], offset: ?>>
    %c0i = arith.constant 0 : i32
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0i) -> i32 {
      %x = memref.load %v[%i] : memref<?xi32, strided<[?], offset: ?>>
      %a = arith.addi %acc, %x : i32
      scf.yield %a : i32
    }
    return %s : i32
  }

  func.func private @set0(%rc: !rc_dv, %value: i32) -> !rc_dv attributes {llvm.linkage = #llvm.linkage<internal>} {
    %updated = reussir.array.with_unique_view (%rc : !rc_dv) -> !rc_dv {
      ^bb0(%view: memref<?xi32, strided<[?], offset: ?>>):
        %c0 = arith.constant 0 : index
        memref.store %value, %view[%c0] : memref<?xi32, strided<[?], offset: ?>>
        reussir.scf.yield
    }
    return %updated : !rc_dv
  }

  func.func @main() -> i32 {
    %c10 = arith.constant 10 : index
    %c100 = arith.constant 100 : i32
    %c45 = arith.constant 45 : i32
    %c145 = arith.constant 145 : i32

    %shared = func.call @iota(%c10) : (index) -> !rc_dv
    reussir.rc.inc (%shared : !rc_dv)
    %updated = func.call @set0(%shared, %c100) : (!rc_dv, i32) -> !rc_dv

    %original_sum = func.call @sum(%shared) : (!rc_dv) -> i32
    %original_bad = arith.cmpi ne, %original_sum, %c45 : i32
    scf.if %original_bad {
      reussir.panic "shared dynamic array was mutated through the clone"
    }
    %updated_sum = func.call @sum(%updated) : (!rc_dv) -> i32
    %updated_bad = arith.cmpi ne, %updated_sum, %c145 : i32
    scf.if %updated_bad {
      reussir.panic "dynamic array clone did not carry the update"
    }
    reussir.rc.dec (%shared : !rc_dv)
    reussir.rc.dec (%updated : !rc_dv)

    // A unique array is updated in place: no clone, same box.
    %unique = func.call @iota(%c10) : (index) -> !rc_dv
    %unique_updated = func.call @set0(%unique, %c100) : (!rc_dv, i32) -> !rc_dv
    %unique_sum = func.call @sum(%unique_updated) : (!rc_dv) -> i32
    %unique_bad = arith.cmpi ne, %unique_sum, %c145 : i32
    scf.if %unique_bad {
      reussir.panic "unique dynamic array update did not persist"
    }
    reussir.rc.dec (%unique_updated : !rc_dv)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
