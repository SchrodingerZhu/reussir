// Pass twice to make sure the output is stable
// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s
module @test {
  // CHECK: func.func private @foo() -> !reussir.rc<index>
  func.func private @foo() -> !reussir.rc<index shared normal>

  // CHECK: func.func private @bar() -> !reussir.rc<index rigid>
  func.func private @bar() -> !reussir.rc<index rigid normal>

  // CHECK: func.func private @baz() -> !reussir.rc<index atomic>
  func.func private @baz() -> !reussir.rc<index shared atomic>

  // CHECK: func.func private @qux() -> !reussir.rc<index atomic>
  func.func private @qux() -> !reussir.rc<index atomic shared>
}
