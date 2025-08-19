// RUN: %reussir-opt %s -verify-diagnostics
// expected-error @+1 {{Token alignment must be a power of two}}
!foo = !reussir.token<align: 7, size: 64>
module @test {
  func.func private @bar() -> !reussir.token<align: 3, size: 8>
}
