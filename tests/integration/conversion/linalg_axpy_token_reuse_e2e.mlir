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

// Executable proof of the axpy token-reuse composition: with unique inputs
// both decs yield tokens, the result box reinterprets a dead donor wholesale
// (skip_rc — no allocation on the hot path), and the elementwise kernel
// reads and writes the aliased buffer with correct results. With shared
// inputs the decs only decrement and the kernel must not clobber the
// still-live sources. Both modes are asserted element by element.

// LOOPS-LABEL: func.func private @axpy(
// LOOPS-NOT: linalg.
// LOOPS-NOT: memref.alloc
// LOOPS: reussir.rc.create {{.*}} skip_rc
// LOOPS: scf.for
// LOOPS: memref.load
// LOOPS: memref.store
#map = affine_map<(d0) -> (d0)>
!vec = !reussir.array<64 x i32>
!rc_vec = !reussir.rc<!vec>
!vtoken = !reussir.token<align: 4, size: 260>
module {
  // out[i] = xs[i] + ys[i]; xs and ys are consumed.
  func.func private @axpy(%xs: !rc_vec, %ys: !rc_vec) -> !rc_vec attributes {llvm.linkage = #llvm.linkage<internal>} {
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

  // fill[i] = base + step*i
  func.func private @iota(%base: i32, %step: i32) -> !rc_vec attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !vec
    %fresh = reussir.rc.create value(%poison : !vec) : !rc_vec
    %filled = reussir.array.with_unique_view (%fresh : !rc_vec) -> !rc_vec {
      ^bb0(%view: memref<64xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        scf.for %i = %c0 to %c64 step %c1 {
          %ii = arith.index_cast %i : index to i32
          %scaled = arith.muli %ii, %step : i32
          %v = arith.addi %scaled, %base : i32
          memref.store %v, %view[%i] : memref<64xi32>
        }
        reussir.scf.yield
    }
    return %filled : !rc_vec
  }

  func.func private @check_iota3(%zs: !rc_vec) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %b = reussir.rc.borrow (%zs : !rc_vec) : !reussir.ref<!vec>
    %v = reussir.array.view(%b : !reussir.ref<!vec>) : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : i32
    scf.for %i = %c0 to %c64 step %c1 {
      %got = memref.load %v[%i] : memref<64xi32>
      %ii = arith.index_cast %i : index to i32
      %want = arith.muli %ii, %c3 : i32
      %bad = arith.cmpi ne, %got, %want : i32
      scf.if %bad {
        reussir.panic "axpy result mismatch"
      }
    }
    return
  }

  func.func @main() -> i32 {
    %ret = arith.constant 0 : i32
    %ci0 = arith.constant 0 : i32
    %ci1 = arith.constant 1 : i32
    %ci2 = arith.constant 2 : i32
    %c1_idx = arith.constant 1 : index

    // Unique inputs: both decs yield tokens; result box reuses one wholesale.
    %xs = func.call @iota(%ci0, %ci1) : (i32, i32) -> !rc_vec
    %ys = func.call @iota(%ci0, %ci2) : (i32, i32) -> !rc_vec
    %zs = func.call @axpy(%xs, %ys) : (!rc_vec, !rc_vec) -> !rc_vec
    func.call @check_iota3(%zs) : (!rc_vec) -> ()
    reussir.rc.dec (%zs : !rc_vec)

    // Shared inputs: decs decrement only; kernel must not clobber the
    // still-live sources.
    %xs2 = func.call @iota(%ci0, %ci1) : (i32, i32) -> !rc_vec
    %ys2 = func.call @iota(%ci0, %ci2) : (i32, i32) -> !rc_vec
    reussir.rc.inc (%xs2 : !rc_vec)
    reussir.rc.inc (%ys2 : !rc_vec)
    %zs2 = func.call @axpy(%xs2, %ys2) : (!rc_vec, !rc_vec) -> !rc_vec
    func.call @check_iota3(%zs2) : (!rc_vec) -> ()
    // sources intact: xs2[1] == 1, ys2[1] == 2
    %bx = reussir.rc.borrow (%xs2 : !rc_vec) : !reussir.ref<!vec>
    %vx = reussir.array.view(%bx : !reussir.ref<!vec>) : memref<64xi32>
    %x1 = memref.load %vx[%c1_idx] : memref<64xi32>
    %xbad = arith.cmpi ne, %x1, %ci1 : i32
    scf.if %xbad {
      reussir.panic "shared source xs was clobbered by the kernel"
    }
    %by = reussir.rc.borrow (%ys2 : !rc_vec) : !reussir.ref<!vec>
    %vy = reussir.array.view(%by : !reussir.ref<!vec>) : memref<64xi32>
    %y1 = memref.load %vy[%c1_idx] : memref<64xi32>
    %ybad = arith.cmpi ne, %y1, %ci2 : i32
    scf.if %ybad {
      reussir.panic "shared source ys was clobbered by the kernel"
    }
    reussir.rc.dec (%xs2 : !rc_vec)
    reussir.rc.dec (%ys2 : !rc_vec)
    reussir.rc.dec (%zs2 : !rc_vec)
    return %ret : i32
  }
}
