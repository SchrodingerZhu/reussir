// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse)' \
// RUN:   | %FileCheck %s

// A non-elementwise DPS chain: fill -> matmul -> bias-add, every stage's outs
// threaded on the destination view's tensor, reading three other rc arrays.
// One-shot bufferize must keep the entire chain on the destination payload:
// no temporary, no copy, in both CoW branches.

#map = affine_map<(d0, d1) -> (d0, d1)>
!mat = !reussir.array<8 x 8 x i32>
!rc_mat = !reussir.rc<!mat>

module {
  // CHECK-LABEL: func.func @mm_bias
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: reussir.rc.is_unique
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: return
  func.func @mm_bias(%out: !rc_mat, %a: !rc_mat, %b: !rc_mat, %bias: !rc_mat) -> !rc_mat {
    %ba = reussir.rc.borrow (%a : !rc_mat) : !reussir.ref<!mat>
    %va = reussir.array.view(%ba : !reussir.ref<!mat>) : tensor<8x8xi32>
    %bb = reussir.rc.borrow (%b : !rc_mat) : !reussir.ref<!mat>
    %vb = reussir.array.view(%bb : !reussir.ref<!mat>) : tensor<8x8xi32>
    %bc = reussir.rc.borrow (%bias : !rc_mat) : !reussir.ref<!mat>
    %vc = reussir.array.view(%bc : !reussir.ref<!mat>) : tensor<8x8xi32>
    %updated = reussir.array.with_unique_view (%out : !rc_mat) -> !rc_mat {
      ^bb0(%view: memref<8x8xi32>):
        %zero = arith.constant 0 : i32
        %dest = bufferization.to_tensor %view restrict writable : memref<8x8xi32> to tensor<8x8xi32>
        %zeroed = linalg.fill ins(%zero : i32) outs(%dest : tensor<8x8xi32>) -> tensor<8x8xi32>
        %mm = linalg.matmul ins(%va, %vb : tensor<8x8xi32>, tensor<8x8xi32>)
                            outs(%zeroed : tensor<8x8xi32>) -> tensor<8x8xi32>
        %biased = linalg.generic
            {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%vc : tensor<8x8xi32>) outs(%mm : tensor<8x8xi32>) {
          ^bb0(%c: i32, %acc: i32):
            %s = arith.addi %acc, %c : i32
            linalg.yield %s : i32
        } -> tensor<8x8xi32>
        bufferization.materialize_in_destination %biased in writable %view
          : (tensor<8x8xi32>, memref<8x8xi32>) -> ()
        reussir.scf.yield
    }
    return %updated : !rc_mat
  }
}
