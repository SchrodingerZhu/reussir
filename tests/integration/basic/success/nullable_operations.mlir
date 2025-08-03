// RUN: %reussir-opt %s | %reussir-opt
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @nullable_check(%nullable: !reussir.nullable<!reussir.rc<i64>>) 
    -> i1 {
      %flag = reussir.nullable.check
        (%nullable : !reussir.nullable<!reussir.rc<i64>>) : i1
      return %flag : i1
  }
}
