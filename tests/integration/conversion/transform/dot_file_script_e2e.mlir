// `rrc --transform-script` end-to-end (issue #349): the integrated lowering
// pipeline preloads the file-based schedule, runs the transform interpreter
// at the `kernel` anchor, and continues through the ordinary LLVM descent —
// reading MLIR and emitting the LLVM dialect, then a linked binary that
// EXECUTES and checks the scheduled kernel still computes
// dot(0..15, 1..16) = 1360.
//
// The schedule (Inputs/dot_unroll.transform.mlir) unrolls the dot loop by 4;
// only the `reussir.kernel`-tagged function is matched, so `iota` keeps its
// rolled loop. In the LLVM-dialect output that shows up as the four
// multiplies of the unrolled body (the kernel is otherwise the only source
// of integer multiplication).
//
// Both runs pin `-O none`: with optimization enabled the inliner runs long
// before the `kernel` anchor, inlines the private `@dot` into `@main` and
// erases it, so a schedule scoped to the tagged function finds no match and
// (correctly) fails the build. Keeping scheduled kernels alive across the
// optimization prologue is frontend work (`#[kernel]` in issue #349 Phase 2).
//
// RUN: %rrc %s --emit mlir-llvm -O none \
// RUN:   --transform-script %S/Inputs/dot_unroll.transform.mlir@kernel -o - \
// RUN:   | %FileCheck %s
//
// The anchor tag is optional and defaults to `kernel`.
// RUN: %rrc %s -O none --transform-script %S/Inputs/dot_unroll.transform.mlir \
// RUN:   -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs
// RUN: %t.exe
//
// CHECK: llvm.func
// CHECK-COUNT-4: llvm.mul
// CHECK-NOT: llvm.mul

!arr = !reussir.array<16 x i64>
!rc = !reussir.rc<!arr>

module {
  // dot(a, b) = sum a[i] * b[i], reading both arrays through borrowed views.
  func.func private @dot(%a: !rc, %b: !rc) -> i64
      attributes {llvm.linkage = #llvm.linkage<internal>, reussir.kernel} {
    %ab = reussir.rc.borrow (%a : !rc) : !reussir.ref<!arr>
    %av = reussir.array.view(%ab : !reussir.ref<!arr>) : memref<16xi64>
    %bb = reussir.rc.borrow (%b : !rc) : !reussir.ref<!arr>
    %bv = reussir.array.view(%bb : !reussir.ref<!arr>) : memref<16xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 16 : index
    %zero = arith.constant 0 : i64
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (i64) {
      %ai = memref.load %av[%i] : memref<16xi64>
      %bi = memref.load %bv[%i] : memref<16xi64>
      %p = arith.muli %ai, %bi : i64
      %s = arith.addi %acc, %p : i64
      scf.yield %s : i64
    }
    reussir.rc.dec (%a : !rc)
    reussir.rc.dec (%b : !rc)
    return %r : i64
  }

  // iota(s)[i] = i + s.
  func.func private @iota(%s: i64) -> !rc
      attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !arr
    %xs = reussir.rc.create value(%poison : !arr) : !rc
    %filled = reussir.array.with_unique_view (%xs : !rc) -> !rc {
      ^bb0(%v: memref<16xi64>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %n = arith.constant 16 : index
        scf.for %i = %c0 to %n step %c1 {
          %ii = arith.index_cast %i : index to i64
          %val = arith.addi %ii, %s : i64
          memref.store %val, %v[%i] : memref<16xi64>
        }
        reussir.scf.yield
    }
    return %filled : !rc
  }

  func.func @main() -> i32 {
    %z = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %a = func.call @iota(%z) : (i64) -> !rc
    %b = func.call @iota(%one) : (i64) -> !rc
    %d = func.call @dot(%a, %b) : (!rc, !rc) -> i64
    // sum_{i=0}^{15} i*(i+1) = 1360
    %expected = arith.constant 1360 : i64
    %fail = arith.cmpi ne, %d, %expected : i64
    scf.if %fail {
      reussir.panic "scheduled dot kernel produced a wrong result"
    }
    %c0i32 = arith.constant 0 : i32
    return %c0i32 : i32
  }
}
