// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// A statically small, naturally aligned token takes the sized fast path
// (`__reussir_allocate_small(size)` = mi_malloc_small behind the OOM check);
// an over-aligned or over-sized layout keeps the generic entry with its
// runtime alignment handling.

// CHECK-LABEL: llvm.func @small_natural
// CHECK: %[[SZ:.+]] = llvm.mlir.constant(40 : i64)
// CHECK: llvm.call @__reussir_allocate_small(%[[SZ]])
func.func @small_natural() -> !reussir.token<align: 8, size: 40> {
  %t = reussir.token.alloc : !reussir.token<align: 8, size: 40>
  return %t : !reussir.token<align: 8, size: 40>
}

// CHECK-LABEL: llvm.func @large_natural
// CHECK-NOT: allocate_small
// CHECK: llvm.call @__reussir_allocate(
func.func @large_natural() -> !reussir.token<align: 8, size: 4096> {
  %t = reussir.token.alloc : !reussir.token<align: 8, size: 4096>
  return %t : !reussir.token<align: 8, size: 4096>
}

// CHECK-LABEL: llvm.func @over_aligned
// CHECK-NOT: allocate_small
// CHECK: llvm.call @__reussir_allocate(
func.func @over_aligned() -> !reussir.token<align: 64, size: 128> {
  %t = reussir.token.alloc : !reussir.token<align: 64, size: 128>
  return %t : !reussir.token<align: 64, size: 128>
}
