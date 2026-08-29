// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=LEAK
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,func.func(promote-buffers-to-stack),canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=STACK
// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-closure-outlining,reussir-lowering-region-patterns,func.func(reussir-inc-dec-cancellation),reussir-rc-decrement-expansion,func.func(reussir-infer-variant-tag),reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-inc-dec-cancellation),reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,func.func(reussir-rc-create-sink),func.func(reussir-rc-create-fusion),reussir-trmc-recursion-analysis,reussir-compile-polymorphic-ffi,canonicalize,cse,one-shot-bufferize{allow-unknown-ops},canonicalize,cse,convert-linalg-to-loops,convert-bufferization-to-memref,func.func(ownership-based-buffer-deallocation),canonicalize,func.func(buffer-deallocation-simplification),bufferization-lower-deallocations,canonicalize,cse)' \
// RUN:   | %FileCheck %s --check-prefix=DEALLOC

// A read-only reduction (dot product): the kernel's 0-d accumulator is a
// genuine bufferization temporary, invisible to the rc/token machinery. This
// test pins the composition hazard and both remedies. With the bare
// bufferization sequence (first run) the accumulator is a memref.alloc with
// no dealloc anywhere — a leak, so any pipeline integration must include a
// cleanup stage. promote-buffers-to-stack (second run) turns the bounded
// temporary into an alloca, preferred for small accumulators; the
// ownership-based deallocation pipeline (third run) pairs the alloc with a
// dealloc when it stays on the heap. The dying inputs' tokens have no
// acceptor here; the decs degenerate to plain frees on the unique path.

!vec = !reussir.array<64 x i32>
!rc_vec = !reussir.rc<!vec>
!vtoken = !reussir.token<align: 4, size: 260>

module {
  // LEAK-LABEL: func.func @dot
  // LEAK: memref.alloc() {{.*}} : memref<i32>
  // LEAK-NOT: memref.dealloc
  // LEAK: return

  // STACK-LABEL: func.func @dot
  // STACK-NOT: memref.alloc()
  // STACK: memref.alloca() {{.*}} : memref<i32>
  // STACK-NOT: memref.alloc()
  // STACK: return

  // DEALLOC-LABEL: func.func @dot
  // DEALLOC: memref.alloc() {{.*}} : memref<i32>
  // DEALLOC: memref.dealloc
  // DEALLOC: return
  func.func @dot(%xs: !rc_vec, %ys: !rc_vec) -> i32 {
    %bx = reussir.rc.borrow (%xs : !rc_vec) : !reussir.ref<!vec>
    %vx = reussir.array.view(%bx : !reussir.ref<!vec>) : tensor<64xi32>
    %by = reussir.rc.borrow (%ys : !rc_vec) : !reussir.ref<!vec>
    %vy = reussir.array.view(%by : !reussir.ref<!vec>) : tensor<64xi32>
    %zero = arith.constant 0 : i32
    %init = tensor.empty() : tensor<i32>
    %acc0 = linalg.fill ins(%zero : i32) outs(%init : tensor<i32>) -> tensor<i32>
    %acc = linalg.dot ins(%vx, %vy : tensor<64xi32>, tensor<64xi32>)
                      outs(%acc0 : tensor<i32>) -> tensor<i32>
    %r = tensor.extract %acc[] : tensor<i32>
    %t1 = reussir.rc.dec (%xs : !rc_vec) : !reussir.nullable<!vtoken>
    %t2 = reussir.rc.dec (%ys : !rc_vec) : !reussir.nullable<!vtoken>
    return %r : i32
  }
}
