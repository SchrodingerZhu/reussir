// RUN: %reussir-opt %s | %FileCheck %s
!foo = !reussir.token<align: 8, size: 64>
module @test {
  // CHECK: func.func private @foo() -> !reussir.token<align : 8, size : 64>
  func.func private @foo() -> !foo
}
