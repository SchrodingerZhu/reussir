// RUN: %reussir-opt %s | %reussir-opt

module {
  // Test nullable dispatch with return value
  func.func @test_nullable_dispatch_with_return(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i64>):
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
      null -> {
        ^bb0:
          %c0 = arith.constant 0 : i32
          reussir.scf.yield %c0 : i32
      }
    }
    func.return %result : i32
  }

  // Test nullable dispatch without return value
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

  // Test nullable dispatch with different inner types
  func.func @test_nullable_dispatch_different_types(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 16>>) -> i1 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 16>>) -> i1 {
      nonnull -> {
        ^bb0(%nonnull_token : !reussir.token<align: 8, size: 16>):
          %true = arith.constant 1 : i1
          reussir.scf.yield %true : i1
      }
      null -> {
        ^bb0:
          %false = arith.constant 0 : i1
          reussir.scf.yield %false : i1
      }
    }
    func.return %result : i1
  }
}
