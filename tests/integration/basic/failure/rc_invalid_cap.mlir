// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
  // CHECK: error: Capability must not be Field or Value for RCType
  func.func private @foo() -> !reussir.rc<index field>
}
