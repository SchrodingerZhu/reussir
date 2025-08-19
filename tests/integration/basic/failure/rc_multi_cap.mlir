// RUN: %reussir-opt %s -verify-diagnostics
module @test {
  // expected-error @+1 {{Capability is already specified}}
  func.func private @foo() -> !reussir.rc<index shared flex>
}
