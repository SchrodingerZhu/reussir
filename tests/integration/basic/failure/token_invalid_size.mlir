// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
  // CHECK: error: Token size must be a multiple of alignment
  func.func private @bar() -> !reussir.token<align: 8, size: 63>
}
