// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
!foo = !reussir.token<align: 7, size: 64>
module @test {
  // CHECK: error: Token alignment must be a power of two
  func.func private @foo() -> !foo
}
