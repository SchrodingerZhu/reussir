// A well-formed transform library that never defines any
// `__reussir_anchor_<name>` entry point; the interpreter must fail with a
// diagnostic naming the missing sequence rather than silently doing nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @some_other_sequence(
      %m: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
