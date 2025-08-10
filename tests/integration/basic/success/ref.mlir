// Pass twice to make sure the output is stable
// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s
module @test {
  // CHECK: func.func private @foo() -> !reussir.ref<index>
  func.func private @foo() -> !reussir.ref<index unspecified normal>

  // CHECK: func.func private @bar() -> !reussir.ref<index rigid>
  func.func private @bar() -> !reussir.ref<index rigid normal>

  // CHECK: func.func private @baz() -> !reussir.ref<index shared atomic>
  func.func private @baz() -> !reussir.ref<index shared atomic>

  // CHECK: func.func private @qux() -> !reussir.ref<index shared atomic>
  func.func private @qux() -> !reussir.ref<index atomic shared>
}
