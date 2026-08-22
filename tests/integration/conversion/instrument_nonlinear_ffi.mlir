// RUN: %reussir-opt %s --reussir-instrument-nonlinear-ffi | %FileCheck %s

// `reussir-instrument-nonlinear-ffi` guards every FFI-import call that
// consumes an rc'd ffi_object/array argument with a reference-count check:
// a count other than one reports the call's source location to
// `__reussir_report_nonlinear_usage`. Scalar arguments and calls to
// functions no import trampoline names stay untouched.

!vec = !reussir.rc<!reussir.ffi_object<"::reussir_rt::collections::vec::Vec<f64>", @vec_drop>>
!arr = !reussir.rc<!reussir.array<4 x f64>>

module {
  func.func private @vec_drop(!vec)
  func.func private @vec_push(!vec, f64) -> !vec
  func.func private @arr_sum(!arr) -> f64
  func.func private @scalar_id(i64) -> i64
  func.func private @native_use(!vec) -> !vec
  reussir.trampoline import "C" @vec_push_ffi = @vec_push
  reussir.trampoline import "C" @arr_sum_ffi = @arr_sum
  reussir.trampoline import "C" @scalar_id_ffi = @scalar_id

  func.func @use(%v: !vec, %x: f64, %a: !arr, %n: i64) -> f64 {
    %r = func.call @vec_push(%v, %x) : (!vec, f64) -> !vec loc("demo.rr":10:9)
    %s = func.call @arr_sum(%a) : (!arr) -> f64 loc("demo.rr":11:9)
    %m = func.call @scalar_id(%n) : (i64) -> i64
    %r2 = func.call @native_use(%r) : (!vec) -> !vec
    return %s : f64
  }
}

// The runtime declaration and the NUL-terminated file-name global:
// CHECK-DAG: llvm.func @__reussir_report_nonlinear_usage(!llvm.ptr, i32, i32)
// CHECK-DAG: llvm.mlir.global linkonce_odr constant @{{.*}}("demo.rr\00")

// CHECK-LABEL: func.func @use

// The ffi_object argument of the imported `vec_push`:
// CHECK: %[[COUNT:.+]] = reussir.rc.fetch
// CHECK: %[[SHARED:.+]] = arith.cmpi ne, %[[COUNT]]
// CHECK: %[[COLD:.+]] = reussir.expect(%[[SHARED]]
// CHECK: scf.if %[[COLD]]
// CHECK: %[[FILE:.+]] = llvm.mlir.addressof
// CHECK-DAG: %[[LINE:.+]] = llvm.mlir.constant(10 : i32)
// CHECK-DAG: %[[COL:.+]] = llvm.mlir.constant(9 : i32)
// CHECK: llvm.call @__reussir_report_nonlinear_usage(%[[FILE]], %[[LINE]], %[[COL]])
// CHECK: call @vec_push

// The array argument of the imported `arr_sum`:
// CHECK: reussir.rc.fetch
// CHECK: scf.if
// CHECK: llvm.mlir.constant(11 : i32)
// CHECK: llvm.call @__reussir_report_nonlinear_usage
// CHECK: call @arr_sum

// A scalar argument and a call without an import trampoline are untouched:
// CHECK-NOT: reussir.rc.fetch
// CHECK: call @scalar_id
// CHECK-NEXT: call @native_use
