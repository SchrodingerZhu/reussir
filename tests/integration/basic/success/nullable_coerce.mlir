// RUN: %reussir-opt %s | %reussir-opt

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @nullable_coerce_token() {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 64>
    
    %nullable = reussir.nullable.create(%token : !reussir.token<align: 8, size: 64>) : !reussir.nullable<!reussir.token<align: 8, size: 64>>
    
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 64>>) : !reussir.token<align: 8, size: 64>
    
    func.return
  }

  func.func @nullable_coerce_rc() {
    // Create an RC value properly
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 16>
    %val = arith.constant 42 : i64
    %rc = reussir.rc.create value(%val : i64) token(%token : !reussir.token<align: 8, size: 16>) : !reussir.rc<i64>
    
    %nullable = reussir.nullable.create(%rc : !reussir.rc<i64>) : !reussir.nullable<!reussir.rc<i64>>
    
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.rc<i64>>) : !reussir.rc<i64>
    
    func.return
  }

  func.func @nullable_coerce_ref() {
    // Create a ref value properly
    %val = arith.constant 42 : i64
    %ref = reussir.ref.spilled(%val : i64) : !reussir.ref<i64>
    
    %nullable = reussir.nullable.create(%ref : !reussir.ref<i64>) : !reussir.nullable<!reussir.ref<i64>>
    
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.ref<i64>>) : !reussir.ref<i64>
    
    func.return
  }
}
