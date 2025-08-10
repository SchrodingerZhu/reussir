// RUN: %reussir-opt %s | %reussir-opt
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @nullable_check(%nullable: !reussir.nullable<!reussir.rc<i64>>) 
    -> i1 {
      %flag = reussir.nullable.check
        (%nullable : !reussir.nullable<!reussir.rc<i64>>) : i1
      return %flag : i1
  }
  func.func private @nullable_create_nonnull(%rc: !reussir.rc<i64>) 
    -> !reussir.nullable<!reussir.rc<i64>> {
      %created = reussir.nullable.create (%rc : !reussir.rc<i64>) : !reussir.nullable<!reussir.rc<i64>>
      return %created : !reussir.nullable<!reussir.rc<i64>>
  }
  func.func private @nullable_create_null() 
    -> !reussir.nullable<!reussir.rc<i64>> {
      %created = reussir.nullable.create : !reussir.nullable<!reussir.rc<i64>>
      return %created : !reussir.nullable<!reussir.rc<i64>>
  }
}
