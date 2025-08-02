// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
  // CHECK: error: failed to verify 'ptrTy': Reussir non-null pointer type
  func.func private @foo() -> !reussir.nullable<index>
}
