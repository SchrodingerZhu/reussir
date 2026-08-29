// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse{emit-remarks=1}),reussir-convert-to-std)' \
// RUN:   --remarks-filter=TokenReuse --remark-format=emitRemark 2>&1 | %FileCheck %s --check-prefix=REMARK
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse)' \
// RUN:   | %FileCheck %s

// A fresh-destination tensor kernel over dying inputs. The views are taken,
// the inputs are decremented, and the result box is created before the kernel
// region — so TokenReuse sees two dead exact-size array tokens feeding the
// result's token.alloc, across the pending tensor code.
//
// Both reuses fire at the ensure tier (Score=2), unadjusted: exact-size array
// boxes are already first-class donors. After rc-create-sink/fusion the unique
// path degenerates to reinterpreting the dead box as the new one (`skip_rc`) —
// no free, no alloc, no refcount write.
//
// Contract note (why this is sound here and not in general): the kernel still
// READS the donor's payload after the create — the loops materialize later,
// during bufferization. With pointer-identity reuse the kernel reads and
// writes the same buffer, which is correct only because every stage is
// elementwise with identical index maps. TokenReuse cannot see this: the
// frontend must emit consuming decs before the kernel ONLY for
// elementwise-aligned kernels, and after it otherwise.

// REMARK: remark: [Passed] TokenReused | Category:TokenReuse:OneShot | Function=axpy | AvailableTokens=2, CompatibleTokens=2, Score=2, Source=loc({{.*}}), Strategy=ensure
// REMARK: remark: [Passed] TokenReused | Category:TokenReuse:OneShot | Function=axpy | AvailableTokens=1, CompatibleTokens=1, Score=2, Source=loc({{.*}}), Strategy=ensure

#map = affine_map<(d0) -> (d0)>
!vec = !reussir.array<64 x i32>
!rc_vec = !reussir.rc<!vec>
!vtoken = !reussir.token<align: 4, size: 260>

module {
  // CHECK-LABEL: func.func @axpy
  // CHECK: %[[COUNT:.+]] = reussir.rc.fetch
  // CHECK: %[[ISONE:.+]] = arith.cmpi eq, %[[COUNT]]
  // CHECK: scf.if
  // CHECK: reussir.rc.reinterpret
  // CHECK: reussir.token.launder
  // CHECK: reussir.rc.create {{.*}} skip_rc
  // CHECK: } else {
  // CHECK: reussir.token.alloc
  // CHECK: reussir.rc.create
  func.func @axpy(%xs: !rc_vec, %ys: !rc_vec) -> !rc_vec {
    %bx = reussir.rc.borrow (%xs : !rc_vec) : !reussir.ref<!vec>
    %vx = reussir.array.view(%bx : !reussir.ref<!vec>) : tensor<64xi32>
    %by = reussir.rc.borrow (%ys : !rc_vec) : !reussir.ref<!vec>
    %vy = reussir.array.view(%by : !reussir.ref<!vec>) : tensor<64xi32>
    %t1 = reussir.rc.dec (%xs : !rc_vec) : !reussir.nullable<!vtoken>
    %t2 = reussir.rc.dec (%ys : !rc_vec) : !reussir.nullable<!vtoken>
    %poison = ub.poison : !vec
    %tk = reussir.token.alloc : !vtoken
    %fresh = reussir.rc.create value(%poison : !vec) token(%tk : !vtoken) : !rc_vec
    %result = reussir.array.with_unique_view (%fresh : !rc_vec) -> !rc_vec {
      ^bb0(%view: memref<64xi32>):
        %dest = bufferization.to_tensor %view restrict writable : memref<64xi32> to tensor<64xi32>
        %sum = linalg.generic
            {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
            ins(%vx, %vy : tensor<64xi32>, tensor<64xi32>) outs(%dest : tensor<64xi32>) {
          ^bb0(%a: i32, %b: i32, %out: i32):
            %s = arith.addi %a, %b : i32
            linalg.yield %s : i32
        } -> tensor<64xi32>
        bufferization.materialize_in_destination %sum in writable %view
          : (tensor<64xi32>, memref<64xi32>) -> ()
        reussir.scf.yield
    }
    return %result : !rc_vec
  }
}
