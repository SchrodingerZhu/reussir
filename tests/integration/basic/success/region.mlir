// RUN: %reussir-opt %s | %FileCheck %s
module @test {
  // CHECK: func.func private @foo() -> !reussir.region
  func.func private @foo() -> !reussir.region
}
