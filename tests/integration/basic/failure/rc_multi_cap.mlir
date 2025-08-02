// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
  // CHECK: error: Capability is already specified
  func.func private @foo() -> !reussir.rc<index rigid shared normal>
}
