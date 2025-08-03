// RUN: %reussir-opt %s | %reussir-opt
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func private @token_alloc() 
    -> !reussir.token<align: 8, size: 64> {
      %token = reussir.token.alloc : !reussir.token<align: 8, size: 64>
      return %token : !reussir.token<align: 8, size: 64>
  }
  func.func private @token_free(%token: !reussir.token<align: 8, size: 64>) {
      reussir.token.free (%token : !reussir.token<align: 8, size: 64>)
      return
  }
  func.func private @token_reinterpret(%token: !reussir.token<align: 8, size: 8>) 
    -> !reussir.ref<i64> {
      %reinterpreted = reussir.token.reinterpret 
        (%token : !reussir.token<align: 8, size: 8>) 
        : !reussir.ref<i64>
      return %reinterpreted : !reussir.ref<i64>
  }
}
