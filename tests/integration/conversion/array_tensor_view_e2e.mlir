// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=LOOPS
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O3 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

!mat2 = !reussir.array<2 x 2 x i32>
!rc_mat2 = !reussir.rc<!mat2>

module {
  func.func private @make_zero_matrix() -> !rc_mat2 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !mat2
    %xs = reussir.rc.create value(%poison : !mat2) : !rc_mat2
    %zeroed = reussir.array.with_unique_view (%xs : !rc_mat2) -> !rc_mat2 {
      ^bb0(%view: memref<2x2xi32>):
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0 : i32
        scf.for %i = %c0 to %c2 step %c1 {
          scf.for %j = %c0 to %c2 step %c1 {
            memref.store %zero, %view[%i, %j] : memref<2x2xi32>
          }
        }
        reussir.scf.yield
    }
    return %zeroed : !rc_mat2
  }

  func.func private @write_matmul_result(%xs: !rc_mat2) -> !rc_mat2 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %updated = reussir.array.with_unique_view (%xs : !rc_mat2) -> !rc_mat2 {
      ^bb0(%view: tensor<2x2xi32>):
        %lhs = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
        %rhs = arith.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi32>
        %product = linalg.matmul
          ins(%lhs, %rhs : tensor<2x2xi32>, tensor<2x2xi32>)
          outs(%view : tensor<2x2xi32>) -> tensor<2x2xi32>
        %materialized = bufferization.materialize_in_destination %product in %view
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %head = tensor.extract %materialized[%c0, %c0] : tensor<2x2xi32>
        %tail = tensor.extract %materialized[%c1, %c1] : tensor<2x2xi32>
        %c19 = arith.constant 19 : i32
        %c50 = arith.constant 50 : i32
        %head_failed = arith.cmpi ne, %head, %c19 : i32
        scf.if %head_failed {
          reussir.panic "matmul materialization produced the wrong [0,0] entry"
        }
        %tail_failed = arith.cmpi ne, %tail, %c50 : i32
        scf.if %tail_failed {
          reussir.panic "matmul materialization produced the wrong [1,1] entry"
        }
        reussir.scf.yield
    }
    return %updated : !rc_mat2
  }

  func.func private @read00_tensor(%xs: !rc_mat2) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %borrow = reussir.rc.borrow (%xs : !rc_mat2) : !reussir.ref<!mat2>
    %view = reussir.array.view(%borrow : !reussir.ref<!mat2>) : tensor<2x2xi32>
    %c0 = arith.constant 0 : index
    %value = tensor.extract %view[%c0, %c0] : tensor<2x2xi32>
    return %value : i32
  }

  func.func private @read11_tensor(%xs: !rc_mat2) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %borrow = reussir.rc.borrow (%xs : !rc_mat2) : !reussir.ref<!mat2>
    %view = reussir.array.view(%borrow : !reussir.ref<!mat2>) : tensor<2x2xi32>
    %c1 = arith.constant 1 : index
    %value = tensor.extract %view[%c1, %c1] : tensor<2x2xi32>
    return %value : i32
  }

  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c19 = arith.constant 19 : i32
    %c50 = arith.constant 50 : i32

    %unique = func.call @make_zero_matrix() : () -> !rc_mat2
    %unique_updated = func.call @write_matmul_result(%unique) : (!rc_mat2) -> !rc_mat2
    %unique_head = func.call @read00_tensor(%unique_updated) : (!rc_mat2) -> i32
    %unique_head_failed = arith.cmpi ne, %unique_head, %c19 : i32
    scf.if %unique_head_failed {
      reussir.panic "unique matmul result did not persist at [0,0]"
    }
    %unique_tail = func.call @read11_tensor(%unique_updated) : (!rc_mat2) -> i32
    %unique_tail_failed = arith.cmpi ne, %unique_tail, %c50 : i32
    scf.if %unique_tail_failed {
      reussir.panic "unique matmul result did not persist at [1,1]"
    }
    reussir.rc.dec (%unique_updated : !rc_mat2)

    %shared = func.call @make_zero_matrix() : () -> !rc_mat2
    reussir.rc.inc (%shared : !rc_mat2)
    %shared_updated = func.call @write_matmul_result(%shared) : (!rc_mat2) -> !rc_mat2
    %shared_original_head = func.call @read00_tensor(%shared) : (!rc_mat2) -> i32
    %shared_original_failed = arith.cmpi ne, %shared_original_head, %c0_i32 : i32
    scf.if %shared_original_failed {
      reussir.panic "shared source matrix was mutated"
    }
    %shared_updated_head = func.call @read00_tensor(%shared_updated) : (!rc_mat2) -> i32
    %shared_updated_head_failed = arith.cmpi ne, %shared_updated_head, %c19 : i32
    scf.if %shared_updated_head_failed {
      reussir.panic "shared matmul clone result did not persist at [0,0]"
    }
    %shared_updated_tail = func.call @read11_tensor(%shared_updated) : (!rc_mat2) -> i32
    %shared_updated_tail_failed = arith.cmpi ne, %shared_updated_tail, %c50 : i32
    scf.if %shared_updated_tail_failed {
      reussir.panic "shared matmul clone result did not persist at [1,1]"
    }
    reussir.rc.dec (%shared : !rc_mat2)
    reussir.rc.dec (%shared_updated : !rc_mat2)

    return %c0_i32 : i32
  }
}

// LOOPS-LABEL: func.func private @write_matmul_result(
// LOOPS-NOT: linalg.
// LOOPS-NOT: scf.parallel
// LOOPS: scf.for
// LOOPS: memref.load
// LOOPS: memref.store
