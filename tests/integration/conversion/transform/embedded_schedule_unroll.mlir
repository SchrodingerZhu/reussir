// Payload-embedded transform schedule (issue #349): the schedule lives in the
// same module as the kernel — a `transform.named_sequence` discovered through
// the module's `transform.with_named_sequence` attribute and the interpreter's
// default `__transform_main` entry point. This is the shape the planned
// frontend `transform <name> for <fn> { ... }` items codegen to, and it pins
// the LLVM transform-op spellings (`transform.structured.match` with an
// `attributes{...}` filter, `transform.loop.unroll`) the pipeline anchors
// rely on.
//
// RUN: %reussir-opt %s --transform-interpreter -o - | %FileCheck %s

!arr = !reussir.array<1024 x f64>
!rc = !reussir.rc<!arr>

module attributes {transform.with_named_sequence} {
  // saxpy over refcounted arrays: y[i] = a * x[i] + y[i], reading x through a
  // borrowed view and updating y in place through a unique view.
  func.func @saxpy(%y: !rc, %x: !rc, %a: f64) -> !rc
      attributes {reussir.kernel} {
    %xb = reussir.rc.borrow (%x : !rc) : !reussir.ref<!arr>
    %xv = reussir.array.view(%xb : !reussir.ref<!arr>) : memref<1024xf64>
    %r = reussir.array.with_unique_view (%y : !rc) -> !rc {
      ^bb0(%v: memref<1024xf64>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %n = arith.constant 1024 : index
        scf.for %i = %c0 to %n step %c1 {
          %xi = memref.load %xv[%i] : memref<1024xf64>
          %yi = memref.load %v[%i] : memref<1024xf64>
          %axi = arith.mulf %a, %xi : f64
          %s = arith.addf %axi, %yi : f64
          memref.store %s, %v[%i] : memref<1024xf64>
        }
        reussir.scf.yield
    }
    reussir.rc.dec (%x : !rc)
    return %r : !rc
  }

  // The user-authored schedule, parameterized over the target function.
  transform.named_sequence @saxpy_schedule(
      %target: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %target
        : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %loops { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }

  // The entry point scopes matching to `reussir.kernel`-tagged functions and
  // dispatches to the schedule, so untagged functions stay untouched.
  transform.named_sequence @__transform_main(
      %m: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]}
        attributes{reussir.kernel} in %m
        : (!transform.any_op) -> !transform.any_op
    transform.include @saxpy_schedule failures(propagate) (%f)
        : (!transform.any_op) -> ()
    transform.yield
  }
}

// After unroll by 4: one loop stepping by 4 whose body carries four
// multiply-add pairs.
// CHECK: func.func @saxpy
// CHECK: scf.for
// CHECK-COUNT-4: arith.mulf
// CHECK-NOT: arith.mulf
