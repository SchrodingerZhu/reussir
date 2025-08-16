module {
  func.func @test() {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 8>
    %nullable = reussir.nullable.create(%token : !reussir.token<align: 8, size: 8>) : !reussir.nullable<!reussir.token<align: 8, size: 8>>
    %coerced = reussir.nullable.coerce(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 8>>) : !reussir.token<align: 8, size: 16>
    func.return
  }
}
