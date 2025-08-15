// RUN: %reussir-opt %s | %reussir-opt

module {
  func.func @test_nullable_dispatch_with_yield(%nullable : !reussir.nullable<!reussir.ref<i32>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.ref<i32>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.ref<i32>):
          %value = reussir.ref.load(%nonnull_ptr : !reussir.ref<i32>) : i32
          reussir.scf.yield %value : i32
      }
      null -> {
        ^bb0:
          %default = arith.constant 0 : i32
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }

  func.func @test_nullable_dispatch_void(%nullable : !reussir.nullable<!reussir.ref<i32>>) {
    reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.ref<i32>>) {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.ref<i32>):
          reussir.scf.yield
      }
      null -> {
        ^bb0:
          reussir.scf.yield
      }
    }
    func.return
  }
}
