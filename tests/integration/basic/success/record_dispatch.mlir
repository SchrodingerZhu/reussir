// RUN: %reussir-opt %s | %reussir-opt

// Define variant record types for testing dispatch
!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

!result_ok = !reussir.record<compound "Result::Ok" {i32}>
!result_err = !reussir.record<compound "Result::Err" {i32}>
!result = !reussir.record<variant "Result" {!result_ok, !result_err}>

module {
  // Test dispatch with single tag
  func.func @test_single_tag_dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%arg : !reussir.ref<!option_some>):
          %value = reussir.ref.load(%opt_ref : !reussir.ref<!option>) : !option
          %extracted = reussir.record.tag(%opt_ref : !reussir.ref<!option>) : index
          %cast = arith.index_cast %extracted : index to i32
          reussir.scf.yield %cast : i32
      }
      [1] -> {
        ^bb0(%arg : !reussir.ref<!option_none>):
          %c42 = arith.constant 42 : i32
          reussir.scf.yield %c42 : i32
      }
    }
    func.return %result : i32
  }

  // Test dispatch with multiple tags in a set
  func.func @test_multi_tag_dispatch(%result_ref : !reussir.ref<!result>) -> i32 {
    %result = reussir.record.dispatch(%result_ref : !reussir.ref<!result>) -> i32 {
      [0, 1] -> {
        %c100 = arith.constant 100 : i32
        reussir.scf.yield %c100 : i32
      }
    }
    func.return %result : i32
  }

  // Test dispatch without return value
  func.func @test_void_dispatch(%opt_ref : !reussir.ref<!option>) {
    reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) {
      [0] -> {
        ^bb0(%arg : !reussir.ref<!option_some>):
          reussir.scf.yield
      }
      [1] -> {
        ^bb0(%arg : !reussir.ref<!option_none>):
          reussir.scf.yield
      }
    }
    func.return
  }
}
