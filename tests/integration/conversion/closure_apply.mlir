// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

// Test closure apply operation lowering pattern
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  
  // Test function that accepts a closure and applies an argument to it
  // CHECK-LABEL: define ptr @apply_to_closure(ptr %0, i32 %1) {
  // CHECK: %3 = getelementptr { i32, { ptr, ptr } }, ptr %0, i32 0, i32 1, i32 1
  // CHECK: %4 = load ptr, ptr %3, align 8
  // CHECK: %5 = ptrtoint ptr %4 to i64
  // CHECK: %6 = sub i64 0, %5
  // CHECK: %7 = and i64 %6, 3
  // CHECK: %8 = add nuw i64 %5, %7
  // CHECK: %9 = inttoptr i64 %8 to ptr
  // CHECK: store i32 %1, ptr %9, align 4
  // CHECK: %10 = ptrtoint ptr %9 to i64
  // CHECK: %11 = add nuw i64 %10, 4
  // CHECK: %12 = inttoptr i64 %11 to ptr
  // CHECK: store ptr %12, ptr %3, align 8
  // CHECK: ret ptr %0
  func.func @apply_to_closure(%closure: !reussir.rc<!reussir.closure<(i32) -> i32>>, %arg: i32) -> !reussir.rc<!reussir.closure<() -> i32>> {
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    return %applied : !reussir.rc<!reussir.closure<() -> i32>>
  }

  // Test function that applies multiple arguments to a closure
  // CHECK-LABEL: define ptr @apply_multiple_args(ptr %0, i32 %1, i64 %2) {
  // CHECK: %4 = getelementptr { i32, { ptr, ptr } }, ptr %0, i32 0, i32 1, i32 1
  // CHECK: %5 = load ptr, ptr %4, align 8
  // CHECK: %6 = ptrtoint ptr %5 to i64
  // CHECK: %7 = sub i64 0, %6
  // CHECK: %8 = and i64 %7, 3
  // CHECK: %9 = add nuw i64 %6, %8
  // CHECK: %10 = inttoptr i64 %9 to ptr
  // CHECK: store i32 %1, ptr %10, align 4
  // CHECK: %11 = ptrtoint ptr %10 to i64
  // CHECK: %12 = add nuw i64 %11, 4
  // CHECK: %13 = inttoptr i64 %12 to ptr
  // CHECK: store ptr %13, ptr %4, align 8
  // CHECK: %14 = getelementptr { i32, { ptr, ptr } }, ptr %0, i32 0, i32 1, i32 1
  // CHECK: %15 = load ptr, ptr %14, align 8
  // CHECK: %16 = ptrtoint ptr %15 to i64
  // CHECK: %17 = sub i64 0, %16
  // CHECK: %18 = and i64 %17, 7
  // CHECK: %19 = add nuw i64 %16, %18
  // CHECK: %20 = inttoptr i64 %19 to ptr
  // CHECK: store i64 %2, ptr %20, align 8
  // CHECK: %21 = ptrtoint ptr %20 to i64
  // CHECK: %22 = add nuw i64 %21, 8
  // CHECK: %23 = inttoptr i64 %22 to ptr
  // CHECK: store ptr %23, ptr %14, align 8
  // CHECK: ret ptr %0
  func.func @apply_multiple_args(%closure: !reussir.rc<!reussir.closure<(i32, i64) -> i32>>, %arg1: i32, %arg2: i64) -> !reussir.rc<!reussir.closure<() -> i32>> {
    %applied1 = reussir.closure.apply (%arg1 : i32) to (%closure : !reussir.rc<!reussir.closure<(i32, i64) -> i32>>) : !reussir.rc<!reussir.closure<(i64) -> i32>>
    %applied2 = reussir.closure.apply (%arg2 : i64) to (%applied1 : !reussir.rc<!reussir.closure<(i64) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    return %applied2 : !reussir.rc<!reussir.closure<() -> i32>>
  }

  // Test function that applies an argument to a closure with no return value
  // CHECK-LABEL: define ptr @apply_void_closure(ptr %0, i32 %1) {
  // CHECK: %3 = getelementptr { i32, { ptr, ptr } }, ptr %0, i32 0, i32 1, i32 1
  // CHECK: %4 = load ptr, ptr %3, align 8
  // CHECK: %5 = ptrtoint ptr %4 to i64
  // CHECK: %6 = sub i64 0, %5
  // CHECK: %7 = and i64 %6, 3
  // CHECK: %8 = add nuw i64 %5, %7
  // CHECK: %9 = inttoptr i64 %8 to ptr
  // CHECK: store i32 %1, ptr %9, align 4
  // CHECK: %10 = ptrtoint ptr %9 to i64
  // CHECK: %11 = add nuw i64 %10, 4
  // CHECK: %12 = inttoptr i64 %11 to ptr
  // CHECK: store ptr %12, ptr %3, align 8
  // CHECK: ret ptr %0
  func.func @apply_void_closure(%closure: !reussir.rc<!reussir.closure<(i32)>>, %arg: i32) -> !reussir.rc<!reussir.closure<()>> {
    %applied = reussir.closure.apply (%arg : i32) to (%closure : !reussir.rc<!reussir.closure<(i32)>>) : !reussir.rc<!reussir.closure<()>>
    return %applied : !reussir.rc<!reussir.closure<()>>
  }

  // Test function that applies a small type (i8) to verify alignment handling
  // CHECK-LABEL: define ptr @apply_small_type(ptr %0, i8 %1) {
  // CHECK: %3 = getelementptr { i32, { ptr, ptr } }, ptr %0, i32 0, i32 1, i32 1
  // CHECK: %4 = load ptr, ptr %3, align 8
  // CHECK: store i8 %1, ptr %4, align 1
  // CHECK: %5 = ptrtoint ptr %4 to i64
  // CHECK: %6 = add nuw i64 %5, 1
  // CHECK: %7 = inttoptr i64 %6 to ptr
  // CHECK: store ptr %7, ptr %3, align 8
  // CHECK: ret ptr %0
  func.func @apply_small_type(%closure: !reussir.rc<!reussir.closure<(i8) -> i8>>, %arg: i8) -> !reussir.rc<!reussir.closure<() -> i8>> {
    %applied = reussir.closure.apply (%arg : i8) to (%closure : !reussir.rc<!reussir.closure<(i8) -> i8>>) : !reussir.rc<!reussir.closure<() -> i8>>
    return %applied : !reussir.rc<!reussir.closure<() -> i8>>
  }
}
