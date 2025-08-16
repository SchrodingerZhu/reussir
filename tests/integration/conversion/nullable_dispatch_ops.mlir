// RUN: %reussir-opt %s --reussir-lowering-scf-ops --reussir-lowering-basic-ops --convert-scf-to-cf --convert-to-llvm | \
// RUN: %mlir-translate --mlir-to-llvmir | %FileCheck %s

module {
  // Test nullable dispatch with value extraction or default
  func.func @test_nullable_unwrap_or_default(%nullable : !reussir.nullable<!reussir.rc<i32>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i32>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i32>):
          %borrowed = reussir.rc.borrow(%nonnull_ptr : !reussir.rc<i32>) : !reussir.ref<i32 shared>
          %value = reussir.ref.load(%borrowed : !reussir.ref<i32 shared>) : i32
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

  // Test nullable dispatch with different return types
  func.func @test_nullable_unwrap_or_false(%nullable : !reussir.nullable<!reussir.ref<i64>>) -> i1 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.ref<i64>>) -> i1 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.ref<i64>):
          %value = reussir.ref.load(%nonnull_ptr : !reussir.ref<i64>) : i64
          %is_positive = arith.cmpi sgt, %value, %value : i64
          %result = arith.cmpi sgt, %value, %value : i64
          reussir.scf.yield %result : i1
      }
      null -> {
        ^bb0:
          %false = arith.constant 0 : i1
          reussir.scf.yield %false : i1
      }
    }
    func.return %result : i1
  }

  // Test nullable dispatch with token types
  func.func @test_nullable_token_dispatch(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 16>>) -> i32 {
    %result = reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.token<align: 8, size: 16>>) -> i32 {
      nonnull -> {
        ^bb0(%nonnull_token : !reussir.token<align: 8, size: 16>):
          %success = arith.constant 1 : i32
          reussir.scf.yield %success : i32
      }
      null -> {
        ^bb0:
          %failure = arith.constant -1 : i32
          reussir.scf.yield %failure : i32
      }
    }
    func.return %result : i32
  }

  // Test void nullable dispatch with side effects
  func.func @test_void_nullable_dispatch(%nullable : !reussir.nullable<!reussir.rc<i32>>) {
    reussir.nullable.dispatch(%nullable : !reussir.nullable<!reussir.rc<i32>>) {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i32>):
          %borrowed = reussir.rc.borrow(%nonnull_ptr : !reussir.rc<i32>) : !reussir.ref<i32 shared>
          %value = reussir.ref.load(%borrowed : !reussir.ref<i32 shared>) : i32
          %doubled = arith.muli %value, %value : i32
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

// CHECK-LABEL: define i32 @test_nullable_unwrap_or_default(ptr %0)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: getelementptr { i64, i32 }, ptr %0, i32 0, i32 1
// CHECK: load i32, ptr
// CHECK: phi i32 [ 0
// CHECK: ret i32

// CHECK-LABEL: define i1 @test_nullable_unwrap_or_false(ptr %0)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: load i64, ptr
// CHECK: phi i1 [ false
// CHECK: ret i1

// CHECK-LABEL: define i32 @test_nullable_token_dispatch(ptr %0)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: phi i32 [ -1
// CHECK: ret i32

// CHECK-LABEL: define void @test_void_nullable_dispatch(ptr %0)
// CHECK: icmp ne ptr %0, null
// CHECK: br i1
// CHECK: getelementptr { i64, i32 }, ptr %0, i32 0, i32 1
// CHECK: load i32, ptr
// CHECK: mul i32
// CHECK: ret void
