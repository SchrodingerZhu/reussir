// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,reussir-unique-carrying-recursion-analysis,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O3 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

!arr512 = !reussir.array<512 x i8>
!rc_arr512 = !reussir.rc<!arr512>

module {
  func.func private @inc_1(%xs: !rc_arr512, %i: index) -> !rc_arr512 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c512 = arith.constant 512 : index
    %cond = arith.cmpi ult, %i, %c512 : index
    %result = scf.if %cond -> (!rc_arr512) {
      %updated = reussir.array.with_unique_view (%xs : !rc_arr512) -> !rc_arr512 {
        ^bb0(%view: memref<512xi8>):
          %old = memref.load %view[%i] : memref<512xi8>
          %one = arith.constant 1 : i8
          %new = arith.addi %old, %one : i8
          memref.store %new, %view[%i] : memref<512xi8>
          reussir.scf.yield
      }
      %c1 = arith.constant 1 : index
      %next = arith.addi %i, %c1 : index
      %rec = func.call @inc_1(%updated, %next) : (!rc_arr512, index) -> !rc_arr512
      scf.yield %rec : !rc_arr512
    } else {
      scf.yield %xs : !rc_arr512
    }
    return %result : !rc_arr512
  }

  func.func private @make_zero_array() -> !rc_arr512 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !arr512
    %xs = reussir.rc.create value(%poison : !arr512) : !rc_arr512
    %zeroed = reussir.array.with_unique_view (%xs : !rc_arr512) -> !rc_arr512 {
      ^bb0(%view: memref<512xi8>):
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0 : i8
        scf.for %i = %c0 to %c512 step %c1 {
          memref.store %zero, %view[%i] : memref<512xi8>
        }
        reussir.scf.yield
    }
    return %zeroed : !rc_arr512
  }

  func.func @inc_1_export(%xs: !rc_arr512) -> !rc_arr512 {
    %c0 = arith.constant 0 : index
    %res = func.call @inc_1(%xs, %c0) : (!rc_arr512, index) -> !rc_arr512
    return %res : !rc_arr512
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    %unique = func.call @make_zero_array() : () -> !rc_arr512
    %unique_result = func.call @inc_1_export(%unique) : (!rc_arr512) -> !rc_arr512
    %unique_borrow = reussir.rc.borrow (%unique_result : !rc_arr512) : !reussir.ref<!arr512>
    %unique_view = reussir.array.view(%unique_borrow : !reussir.ref<!arr512>) : memref<512xi8>
    %unique_sum = scf.for %i = %c0 to %c512 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
      %value = memref.load %unique_view[%i] : memref<512xi8>
      %value_i32 = arith.extui %value : i8 to i32
      %next = arith.addi %acc, %value_i32 : i32
      scf.yield %next : i32
    }
    %unique_failed = arith.cmpi ne, %unique_sum, %c512_i32 : i32
    scf.if %unique_failed {
      reussir.panic "unique fast path did not update all 512 elements"
    }
    reussir.rc.dec (%unique_result : !rc_arr512)

    %shared = func.call @make_zero_array() : () -> !rc_arr512
    reussir.rc.inc (%shared : !rc_arr512)
    %shared_result = func.call @inc_1_export(%shared) : (!rc_arr512) -> !rc_arr512
    %shared_borrow = reussir.rc.borrow (%shared : !rc_arr512) : !reussir.ref<!arr512>
    %shared_view = reussir.array.view(%shared_borrow : !reussir.ref<!arr512>) : memref<512xi8>
    %shared_sum = scf.for %i = %c0 to %c512 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
      %value = memref.load %shared_view[%i] : memref<512xi8>
      %value_i32 = arith.extui %value : i8 to i32
      %next = arith.addi %acc, %value_i32 : i32
      scf.yield %next : i32
    }
    %shared_original_failed = arith.cmpi ne, %shared_sum, %c0_i32 : i32
    scf.if %shared_original_failed {
      reussir.panic "shared source array was mutated"
    }
    %shared_result_borrow = reussir.rc.borrow (%shared_result : !rc_arr512) : !reussir.ref<!arr512>
    %shared_result_view = reussir.array.view(%shared_result_borrow : !reussir.ref<!arr512>) : memref<512xi8>
    %shared_result_sum = scf.for %i = %c0 to %c512 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
      %value = memref.load %shared_result_view[%i] : memref<512xi8>
      %value_i32 = arith.extui %value : i8 to i32
      %next = arith.addi %acc, %value_i32 : i32
      scf.yield %next : i32
    }
    %shared_result_failed = arith.cmpi ne, %shared_result_sum, %c512_i32 : i32
    scf.if %shared_result_failed {
      reussir.panic "shared fallback result did not update all 512 elements"
    }
    reussir.rc.dec (%shared : !rc_arr512)
    reussir.rc.dec (%shared_result : !rc_arr512)

    return %c0_i32 : i32
  }
}
