// RUN: %reussir-opt %s | %reussir-opt
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @inc_rc(%rc: !reussir.rc<i64>) {
      reussir.rc.inc (%rc : !reussir.rc<i64>)
      return 
  }
  func.func private @dec_rc(%rc: !reussir.rc<i64>) 
    -> !reussir.nullable<!reussir.token<align: 8, size: 8>> {
      %tk = reussir.rc.dec (%rc : !reussir.rc<i64>) 
        : !reussir.nullable<!reussir.token<align: 8, size: 8>>
      return %tk : !reussir.nullable<!reussir.token<align: 8, size: 8>>
  }
}
