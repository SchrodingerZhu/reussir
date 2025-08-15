// RUN: %reussir-opt %s --reussir-lowering-scf-ops --reussir-lowering-basic-ops --convert-scf-to-cf --convert-to-llvm | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

!option_some = !reussir.record<compound "Option::Some" {i32}>
!option_none = !reussir.record<compound "Option::None" {}>
!option = !reussir.record<variant "Option" {!option_some, !option_none}>

!result_ok = !reussir.record<compound "Result::Ok" {i32}>
!result_err = !reussir.record<compound "Result::Err" {i32}>
!result = !reussir.record<variant "Result" {!result_ok, !result_err}>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // Option-like API using record dispatch: unwrap or default
  func.func @option_unwrap_or_default(%opt_ref : !reussir.ref<!option>, %default : i32) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none>):
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }

  // Option-like API using record dispatch: unwrap or compute default
  func.func @option_unwrap_or_compute(%opt_ref : !reussir.ref<!option>) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none>):
          %computed = arith.constant 42 : i32
          %doubled = arith.muli %computed, %computed : i32
          reussir.scf.yield %doubled : i32
      }
    }
    func.return %result : i32
  }

  // Result-like API using record dispatch: unwrap ok or return error
  func.func @result_unwrap_or_error(%result_ref : !reussir.ref<!result>) -> i32 {
    %result = reussir.record.dispatch(%result_ref : !reussir.ref<!result>) -> i32 {
      [0] -> {
        ^bb0(%ok_ref : !reussir.ref<!result_ok>):
          %field_ref = reussir.ref.project(%ok_ref : !reussir.ref<!result_ok>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%err_ref : !reussir.ref<!result_err>):
          %field_ref = reussir.ref.project(%err_ref : !reussir.ref<!result_err>) [0] : !reussir.ref<i32>
          %error_code = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          %negated = arith.subi %error_code, %error_code : i32
          reussir.scf.yield %negated : i32
      }
    }
    func.return %result : i32
  }

  // Nullable option-like API: unwrap or default
  func.func @nullable_unwrap_or_default(%nullable : !reussir.nullable<!reussir.rc<i32>>, %default : i32) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i32>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i32>):
          %borrowed = reussir.rc.borrow(%nonnull_ptr : !reussir.rc<i32>) : !reussir.ref<i32 shared>
          %value = reussir.ref.load(%borrowed : !reussir.ref<i32 shared>) : i32
          reussir.scf.yield %value : i32
      }
      null -> {
        ^bb0:
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }

  // Nullable option-like API: unwrap or compute
  func.func @nullable_unwrap_or_compute(%nullable : !reussir.nullable<!reussir.ref<i64>>) -> i64 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.ref<i64>>) -> i64 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.ref<i64>):
          %value = reussir.ref.load(%nonnull_ptr : !reussir.ref<i64>) : i64
          reussir.scf.yield %value : i64
      }
      null -> {
        ^bb0:
          %computed = arith.constant 100 : i64
          %squared = arith.muli %computed, %computed : i64
          reussir.scf.yield %squared : i64
      }
    }
    func.return %result : i64
  }

  // Option-like API with side effects: map or execute default
  func.func @option_map_or_execute(%opt_ref : !reussir.ref<!option>) -> i32 {
    %res = reussir.record.dispatch(%opt_ref : !reussir.ref<!option>) -> i32 {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some>) [0] : !reussir.ref<i32>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32>) : i32
          %doubled = arith.muli %extracted, %extracted : i32
          reussir.scf.yield %doubled : i32
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none>):
          %default_action = arith.constant 0 : i32
          reussir.scf.yield %default_action : i32
      }
    }
    func.return %res : i32
  }
}

// CHECK-LABEL: define i32 @option_unwrap_or_default(ptr %0, i32 %1)
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

// CHECK-LABEL: define i32 @option_unwrap_or_compute(ptr %0)
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

// CHECK-LABEL: define i32 @result_unwrap_or_error(ptr %0)
// CHECK: getelementptr %Result, ptr %0, i32 0, i32 0
// CHECK: load i64, ptr
// CHECK: trunc i64
// CHECK: switch i32
// CHECK: i32 0, label
// CHECK: i32 1, label
// CHECK: getelementptr %"Result::Ok", ptr
// CHECK: load i32, ptr
// CHECK: getelementptr %"Result::Err", ptr
// CHECK: load i32, ptr
// CHECK: phi i32 [ poison
// CHECK: ret i32

// CHECK-LABEL: define i32 @nullable_unwrap_or_default(ptr %0, i32 %1)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: getelementptr { i64, i32 }, ptr %0, i32 0, i32 1
// CHECK: load i32, ptr
// CHECK: phi i32 [ %1
// CHECK: ret i32

// CHECK-LABEL: define i64 @nullable_unwrap_or_compute(ptr %0)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: load i64, ptr
// CHECK: phi i64 [
// CHECK: ret i64
