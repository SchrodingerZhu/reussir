// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Executable proof of the dynamic-extent array lifecycle: construct with a
// runtime extent (token size computed as header + n * elemsize), fill and
// reduce through the strided view, release through the dynamic token and
// the unsized free. Asserts sum(0..9) == 45.
!dv = !reussir.array<? x i32>
!rc_dv = !reussir.rc<!dv>
module {
  // Build [0, 1, ..., n-1], sum it through the strided view, release the box.
  func.func private @iota_sum(%n: index) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
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
    %c0i = arith.constant 0 : i32
    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0i) -> i32 {
      %x = memref.load %v[%i] : memref<?xi32, strided<[?], offset: ?>>
      %a = arith.addi %acc, %x : i32
      scf.yield %a : i32
    }
    reussir.rc.dec (%rc : !rc_dv)
    return %sum : i32
  }

  func.func @main() -> i32 {
    %c10 = arith.constant 10 : index
    %s = func.call @iota_sum(%c10) : (index) -> i32
    // sum 0..9 = 45
    %c45 = arith.constant 45 : i32
    %bad = arith.cmpi ne, %s, %c45 : i32
    scf.if %bad {
      reussir.panic "dynamic-extent iota sum mismatch"
    }
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
