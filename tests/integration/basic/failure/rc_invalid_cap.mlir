// RUN: %reussir-opt %s -verify-diagnostics
module @test {
  // expected-error @+1 {{Capability must be shared, flex or rigid for RcType}}
  func.func private @foo() -> !reussir.rc<index field>
}
