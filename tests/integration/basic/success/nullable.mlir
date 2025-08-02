// RUN: %reussir-opt %s | %FileCheck %s
module @test {
  // CHECK: func.func private @foo() -> !reussir.nullable<!reussir.token<align : 8, size : 64>
  func.func private @foo() 
    -> !reussir.nullable<!reussir.token<align: 8, size: 64>>

}
