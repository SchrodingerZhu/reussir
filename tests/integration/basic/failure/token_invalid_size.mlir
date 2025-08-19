// RUN: %reussir-opt %s -verify-diagnostics
module @test {
  func.func private @bar() -> !reussir.token<align: 8, size: 63>
  // expected-error @-1 {{Token size must be a multiple of alignment}}
}
