// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O3 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

!arr4 = !reussir.array<4 x i8>
!rc_arr4 = !reussir.rc<!arr4>

module {
  func.func private @make_zero_array() -> !rc_arr4 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !arr4
    %xs = reussir.rc.create value(%poison : !arr4) : !rc_arr4
    %zeroed = reussir.array.with_unique_view (%xs : !rc_arr4) -> !rc_arr4 {
      ^bb0(%view: memref<4xi8>):
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0 : i8
        scf.for %i = %c0 to %c4 step %c1 {
          memref.store %zero, %view[%i] : memref<4xi8>
        }
        reussir.scf.yield
    }
    return %zeroed : !rc_arr4
  }

  func.func private @set0(%xs: !rc_arr4, %value: i8) -> !rc_arr4 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %updated = reussir.array.with_unique_view (%xs : !rc_arr4) -> !rc_arr4 {
      ^bb0(%view: memref<4xi8>):
        %c0 = arith.constant 0 : index
        memref.store %value, %view[%c0] : memref<4xi8>
        reussir.scf.yield
    }
    return %updated : !rc_arr4
  }

  func.func private @read0_borrowed(%xs: !rc_arr4) -> i8 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %borrow = reussir.rc.borrow (%xs : !rc_arr4) : !reussir.ref<!arr4>
    %view = reussir.array.view(%borrow : !reussir.ref<!arr4>) : memref<4xi8>
    %c0 = arith.constant 0 : index
    %value = memref.load %view[%c0] : memref<4xi8>
    return %value : i8
  }

  func.func private @read0_unique(%xs: !rc_arr4) -> i8 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %value = reussir.array.with_unique_view (%xs : !rc_arr4) -> i8 {
      ^bb0(%view: memref<4xi8>):
        %c0 = arith.constant 0 : index
        %loaded = memref.load %view[%c0] : memref<4xi8>
        reussir.scf.yield %loaded : i8
    }
    return %value : i8
  }

  func.func @main() -> i32 {
    %c0_i8 = arith.constant 0 : i8
    %c7_i8 = arith.constant 7 : i8
    %c9_i8 = arith.constant 9 : i8
    %c0_i32 = arith.constant 0 : i32

    %unique = func.call @make_zero_array() : () -> !rc_arr4
    %unique_updated = func.call @set0(%unique, %c7_i8) : (!rc_arr4, i8) -> !rc_arr4
    %unique_head = func.call @read0_unique(%unique_updated) : (!rc_arr4) -> i8
    %unique_failed = arith.cmpi ne, %unique_head, %c7_i8 : i8
    scf.if %unique_failed {
      reussir.panic "unique array update did not persist"
    }
    reussir.rc.dec (%unique_updated : !rc_arr4)

    %shared = func.call @make_zero_array() : () -> !rc_arr4
    reussir.rc.inc (%shared : !rc_arr4)
    %shared_updated = func.call @set0(%shared, %c9_i8) : (!rc_arr4, i8) -> !rc_arr4
    %shared_original_head = func.call @read0_borrowed(%shared) : (!rc_arr4) -> i8
    %shared_original_failed = arith.cmpi ne, %shared_original_head, %c0_i8 : i8
    scf.if %shared_original_failed {
      reussir.panic "shared source array was mutated"
    }
    %shared_updated_head = func.call @read0_borrowed(%shared_updated) : (!rc_arr4) -> i8
    %shared_updated_failed = arith.cmpi ne, %shared_updated_head, %c9_i8 : i8
    scf.if %shared_updated_failed {
      reussir.panic "shared clone result was not updated"
    }
    reussir.rc.dec (%shared : !rc_arr4)
    reussir.rc.dec (%shared_updated : !rc_arr4)

    %shared_scalar = func.call @make_zero_array() : () -> !rc_arr4
    reussir.rc.inc (%shared_scalar : !rc_arr4)
    %shared_scalar_head = func.call @read0_unique(%shared_scalar) : (!rc_arr4) -> i8
    %shared_scalar_failed = arith.cmpi ne, %shared_scalar_head, %c0_i8 : i8
    scf.if %shared_scalar_failed {
      reussir.panic "scalar array.with_unique_view read returned the wrong value"
    }
    %shared_scalar_original_head = func.call @read0_borrowed(%shared_scalar) : (!rc_arr4) -> i8
    %shared_scalar_original_failed = arith.cmpi ne, %shared_scalar_original_head, %c0_i8 : i8
    scf.if %shared_scalar_original_failed {
      reussir.panic "scalar array.with_unique_view read mutated the shared source"
    }
    reussir.rc.dec (%shared_scalar : !rc_arr4)

    return %c0_i32 : i32
  }
}
