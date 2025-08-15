// RUN: %reussir-opt %s --reussir-lowering-scf-ops --reussir-lowering-basic-ops --convert-scf-to-cf --convert-to-llvm | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

!result_ok = !reussir.record<compound "Result::Ok" {i32}>
!result_err = !reussir.record<compound "Result::Err" {i32}>
!result = !reussir.record<variant "Result" {!result_ok, !result_err}>

module {
  // Test option-like behavior: extract value or return default
  func.func @test_option_unwrap_or_default(%opt_ref : !reussir.ref<!option>) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none>):
          %default = arith.constant -1 : i32
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }

  // Test result-like behavior: extract ok value or return error code
  func.func @test_result_unwrap_or_error_code(%result_ref : !reussir.ref<!result>) -> i32 {
    %result = reussir.record.dispatch(%result_ref : !reussir.ref<!result>) -> i32 {
      [0] -> {
        ^bb0(%ok_ref : !reussir.ref<!result_ok>):
          %field_ref = reussir.ref.project(%ok_ref : !reussir.ref<!result_ok>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%err_ref : !reussir.ref<!result_err>):
          %error_code = arith.constant 1 : i32
          reussir.scf.yield %error_code : i32
      }
    }
    func.return %result : i32
  }

  // Test void dispatch with side effects
  func.func @test_void_dispatch_with_side_effects(%opt_ref : !reussir.ref<!option>) {
    reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          %doubled = arith.muli %extracted, %extracted : i32
          reussir.scf.yield
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none>):
          reussir.scf.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: define i32 @test_option_unwrap_or_default(ptr %0)
// CHECK: getelementptr %Option, ptr %0, i32 0, i32 0
// CHECK: load i64, ptr
// CHECK: trunc i64
// CHECK: switch i32
// CHECK: i32 0, label
// CHECK: i32 1, label
// CHECK: getelementptr %"Option::Some", ptr
// CHECK: load i32, ptr
// CHECK: phi i32 [ poison
// CHECK: ret i32

// CHECK-LABEL: define i32 @test_result_unwrap_or_error_code(ptr %0)
// CHECK: getelementptr %Result, ptr %0, i32 0, i32 0
// CHECK: load i64, ptr
// CHECK: trunc i64
// CHECK: switch i32
// CHECK: i32 0, label
// CHECK: i32 1, label
// CHECK: getelementptr %"Result::Ok", ptr
// CHECK: load i32, ptr
// CHECK: phi i32 [ poison
// CHECK: ret i32

// CHECK-LABEL: define void @test_void_dispatch_with_side_effects(ptr %0)
// CHECK: getelementptr %Option, ptr %0, i32 0, i32 0
// CHECK: load i64, ptr
// CHECK: trunc i64
// CHECK: switch i32
// CHECK: i32 0, label
// CHECK: i32 1, label
// CHECK: getelementptr %"Option::Some", ptr
// CHECK: load i32, ptr
// CHECK: mul i32
// CHECK: ret void
