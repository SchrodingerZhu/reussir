// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
  // CHECK: Capability must be shared, flex or rigid for RcType
  func.func private @foo() -> !reussir.rc<index field>
}
