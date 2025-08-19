// RUN: %reussir-opt %s -verify-diagnostics
module @test {
  func.func private @foo() -> !reussir.nullable<index>
  // expected-error @-1 {{failed to verify 'ptrTy': Reussir non-null pointer type}}
}
