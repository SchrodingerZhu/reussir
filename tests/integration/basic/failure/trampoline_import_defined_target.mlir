// RUN: %reussir-opt %s -verify-diagnostics

// An import trampoline materializes its target's definition, so the target
// must be a body-less declaration.
module {
  func.func private @already_defined(%x: i64) -> i64 {
    return %x : i64
  }

  // expected-error @below {{import target must be a body-less function declaration: @already_defined}}
  reussir.trampoline import "C" @already_defined_c = @already_defined
}
