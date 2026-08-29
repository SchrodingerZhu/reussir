// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse)' \
// RUN:   | %FileCheck %s

// A `modify`-shaped kernel: two elementwise linalg.generic stages threaded in
// destination-passing style through the unique view's tensor (the view value
// is the outs of every stage), anchored with the memref-destination form of
// materialize_in_destination. One-shot bufferize must run the whole chain in
// place on the rc payload: no temporary buffer, no copy, in either CoW branch.
//
// Two contracts this test pins, discovered the hard way:
// - The tensor-destination materialize_in_destination form is dead code when
//   its result is unused (value semantics): the memref-destination form is
//   the only sound write-back anchor for an effectful kernel.
// - The outs-threading is what makes in-place possible. Rewriting the chain
//   through tensor.empty (as linalg-fuse-elementwise-ops does) defeats both
//   eliminate-empty-tensors (the ins alias the destination) and one-shot's
//   in-place analysis, yielding an alloc + copy instead.

#map = affine_map<(d0) -> (d0)>

!vec = !reussir.array<64 x i32>
!rc_vec = !reussir.rc<!vec>

module {
  // CHECK-LABEL: func.func @scale_shift
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: %[[VIEW:.+]] = reussir.array.view
  // CHECK: scf.for
  // CHECK: memref.load %[[VIEW]]
  // CHECK: arith.muli
  // CHECK: memref.store {{.+}}, %[[VIEW]]
  // CHECK: scf.for
  // CHECK: memref.load %[[VIEW]]
  // CHECK: arith.addi
  // CHECK: memref.store {{.+}}, %[[VIEW]]
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: return
  func.func @scale_shift(%xs: !rc_vec) -> !rc_vec {
    %updated = reussir.array.with_unique_view (%xs : !rc_vec) -> !rc_vec {
      ^bb0(%view: memref<64xi32>):
        %c2 = arith.constant 2 : i32
        %c3 = arith.constant 3 : i32
        %t = bufferization.to_tensor %view restrict writable : memref<64xi32> to tensor<64xi32>
        %scaled = linalg.generic
            {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
            ins(%t : tensor<64xi32>) outs(%t : tensor<64xi32>) {
          ^bb0(%in: i32, %out: i32):
            %m = arith.muli %in, %c2 : i32
            linalg.yield %m : i32
        } -> tensor<64xi32>
        %shifted = linalg.generic
            {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
            ins(%scaled : tensor<64xi32>) outs(%scaled : tensor<64xi32>) {
          ^bb0(%in: i32, %out: i32):
            %a = arith.addi %in, %c3 : i32
            linalg.yield %a : i32
        } -> tensor<64xi32>
        bufferization.materialize_in_destination %shifted in writable %view
          : (tensor<64xi32>, memref<64xi32>) -> ()
        reussir.scf.yield
    }
    return %updated : !rc_vec
  }
}
