// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/linalg_tile_fuse_schedule.mlir},transform-interpreter,lower-affine,canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=FUSE
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/linalg_tile_fuse_schedule.mlir},transform-interpreter,lower-affine,reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=BUF
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/linalg_tile_fuse_schedule.mlir},transform-interpreter,lower-affine,reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse,scf-forall-to-parallel)' \
// RUN:   | %FileCheck %s --check-prefix=PAR
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/linalg_tile_fuse_schedule.mlir},transform-interpreter,lower-affine,reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse,scf-forall-to-parallel,convert-scf-to-openmp)' \
// RUN:   | %FileCheck %s --check-prefix=OMP
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/linalg_tile_fuse_schedule.mlir},transform-interpreter,lower-affine,reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse,scf-forall-to-for,expand-strided-metadata,lower-affine,canonicalize,control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O3 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Transform-driven tile-and-fuse over an rc array kernel: relu(A@B + bias),
// DPS-threaded onto the destination view. The schedule
// (Inputs/linalg_tile_fuse_schedule.mlir) tiles the trailing elementwise
// stage into an 8x8 scf.forall and fuses matmul and fill into the tile loop
// BEFORE the reussir expansion, so the single kernel body is scheduled once
// and then cloned into both CoW branches. This is the fusion path that works
// where linalg-fuse-elementwise-ops fails: tile-and-fuse keeps the
// destination threaded through shared_outs instead of rebuilding outs from
// tensor.empty.
//
// Pipeline facts this pins:
// - lower-affine must run after the interpreter: tiling emits affine.apply,
//   and the affine dialect is not in reussir-convert-to-std's legal set, so
//   the with_unique_view expansion pattern rolls back on a tiled body.
// - The bufferized forall reads and writes memref.subviews of the rc
//   payloads with runtime offsets; expand-strided-metadata (plus a second
//   lower-affine) is required before reussir-lowering-basic-ops, exactly as
//   docs/design/dynamic-extent-arrays.md anticipates for strided views.
// - scf-forall-to-parallel materializes scf.parallel in both CoW arms, and
//   convert-scf-to-openmp lowers those to omp.parallel/omp.wsloop — the
//   OpenMP direction named in AGENTS.md is reachable from this path.
// - The executable run serializes with scf-forall-to-for and asserts
//   out[i][j] == max(0, j - 4) for every element.

// FUSE-LABEL: func.func private @fused_kernel
// FUSE: reussir.array.with_unique_view
// FUSE: scf.forall (%{{.+}}, %{{.+}}) in (2, 2) shared_outs(%{{.+}} = %{{.+}}) -> (tensor<16x16xi32>)
// FUSE-DAG: tensor.extract_slice
// FUSE: linalg.fill
// FUSE: linalg.matmul
// FUSE: linalg.generic
// FUSE: arith.maxsi
// FUSE: scf.forall.in_parallel
// FUSE: tensor.parallel_insert_slice
// FUSE-NOT: scf.forall

// BUF-LABEL: func.func private @fused_kernel
// BUF-NOT: memref.alloc
// BUF-NOT: memref.copy
// BUF: scf.forall
// BUF: memref.subview {{.*}} strided<[16, 1], offset: ?>
// BUF-NOT: memref.alloc
// BUF-NOT: memref.copy

// PAR-LABEL: func.func private @fused_kernel
// PAR: scf.parallel
// PAR-NOT: scf.forall

// OMP-LABEL: func.func private @fused_kernel
// OMP: omp.parallel
// OMP: omp.wsloop

// PoC 5: transform-driven tile-and-fuse + parallelization.
// relu(A@B + bias): fill -> matmul -> bias+relu, DPS-threaded; the transform
// script tiles the trailing elementwise stage into an 8x8 scf.forall and
// fuses matmul and fill into the tile loop.
#map = affine_map<(d0, d1) -> (d0, d1)>
!mat = !reussir.array<16 x 16 x i32>
!rc_mat = !reussir.rc<!mat>

module {
  func.func private @fused_kernel(%out: !rc_mat, %a: !rc_mat, %b: !rc_mat, %bias: !rc_mat) -> !rc_mat attributes {llvm.linkage = #llvm.linkage<internal>} {
    %ba = reussir.rc.borrow (%a : !rc_mat) : !reussir.ref<!mat>
    %va = reussir.array.view(%ba : !reussir.ref<!mat>) : tensor<16x16xi32>
    %bb = reussir.rc.borrow (%b : !rc_mat) : !reussir.ref<!mat>
    %vb = reussir.array.view(%bb : !reussir.ref<!mat>) : tensor<16x16xi32>
    %bc = reussir.rc.borrow (%bias : !rc_mat) : !reussir.ref<!mat>
    %vbias = reussir.array.view(%bc : !reussir.ref<!mat>) : tensor<16x16xi32>
    %updated = reussir.array.with_unique_view (%out : !rc_mat) -> !rc_mat {
      ^bb0(%view: memref<16x16xi32>):
        %zero = arith.constant 0 : i32
        %dest = bufferization.to_tensor %view restrict writable : memref<16x16xi32> to tensor<16x16xi32>
        %zeroed = linalg.fill ins(%zero : i32) outs(%dest : tensor<16x16xi32>) -> tensor<16x16xi32>
        %mm = linalg.matmul ins(%va, %vb : tensor<16x16xi32>, tensor<16x16xi32>)
                            outs(%zeroed : tensor<16x16xi32>) -> tensor<16x16xi32>
        %relu = linalg.generic
            {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%vbias : tensor<16x16xi32>) outs(%mm : tensor<16x16xi32>) {
          ^bb0(%c: i32, %acc: i32):
            %s = arith.addi %acc, %c : i32
            %r = arith.maxsi %s, %zero : i32
            linalg.yield %r : i32
        } -> tensor<16x16xi32>
        bufferization.materialize_in_destination %relu in writable %view
          : (tensor<16x16xi32>, memref<16x16xi32>) -> ()
        reussir.scf.yield
    }
    return %updated : !rc_mat
  }

  // fill[i][j] = base + jstep*j  (row-independent so bias rows are equal)
  func.func private @fill2d(%base: i32, %jstep: i32) -> !rc_mat attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !mat
    %fresh = reussir.rc.create value(%poison : !mat) : !rc_mat
    %filled = reussir.array.with_unique_view (%fresh : !rc_mat) -> !rc_mat {
      ^bb0(%view: memref<16x16xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {
          scf.for %j = %c0 to %c16 step %c1 {
            %jj = arith.index_cast %j : index to i32
            %scaled = arith.muli %jj, %jstep : i32
            %v = arith.addi %scaled, %base : i32
            memref.store %v, %view[%i, %j] : memref<16x16xi32>
          }
        }
        reussir.scf.yield
    }
    return %filled : !rc_mat
  }

  func.func @main() -> i32 {
    %ret = arith.constant 0 : i32
    %ci0 = arith.constant 0 : i32
    %ci1 = arith.constant 1 : i32
    %cin20 = arith.constant -20 : i32
    // A = all ones, B = all ones, bias[i][j] = j - 20
    %a = func.call @fill2d(%ci1, %ci0) : (i32, i32) -> !rc_mat
    %b = func.call @fill2d(%ci1, %ci0) : (i32, i32) -> !rc_mat
    %bias = func.call @fill2d(%cin20, %ci1) : (i32, i32) -> !rc_mat
    %poison = ub.poison : !mat
    %out = reussir.rc.create value(%poison : !mat) : !rc_mat
    // out[i][j] = relu(16 + j - 20) = max(0, j - 4)
    %r = func.call @fused_kernel(%out, %a, %b, %bias) : (!rc_mat, !rc_mat, !rc_mat, !rc_mat) -> !rc_mat
    %br = reussir.rc.borrow (%r : !rc_mat) : !reussir.ref<!mat>
    %vr = reussir.array.view(%br : !reussir.ref<!mat>) : memref<16x16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %got = memref.load %vr[%i, %j] : memref<16x16xi32>
        %jj = arith.index_cast %j : index to i32
        %shift = arith.subi %jj, %c4 : i32
        %want = arith.maxsi %shift, %ci0 : i32
        %bad = arith.cmpi ne, %got, %want : i32
        scf.if %bad {
          reussir.panic "fused tiled kernel produced a wrong element"
        }
      }
    }
    reussir.rc.dec (%a : !rc_mat)
    reussir.rc.dec (%b : !rc_mat)
    reussir.rc.dec (%bias : !rc_mat)
    reussir.rc.dec (%r : !rc_mat)
    return %ret : i32
  }
}
