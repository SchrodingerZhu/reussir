// File-based schedule consumed by `rrc --transform-script <this file>@kernel`
// (issue #349). The driver preloads it into the context's transform library;
// the interpreter then runs the anchor entry point
// `@__reussir_anchor_kernel` over the payload module.
//
// Matching contract: schedules scope themselves to `reussir.kernel`-tagged
// functions, keeping the blast radius limited to code that opted in.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__reussir_anchor_kernel(
      %m: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]}
        attributes{reussir.kernel} in %m
        : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %f
        : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %loops { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}
