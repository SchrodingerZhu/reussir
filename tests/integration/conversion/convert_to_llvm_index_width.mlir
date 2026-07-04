// RUN: %reussir-opt %s --split-input-file --reussir-convert-to-llvm | %FileCheck %s

// `reussir-convert-to-llvm` builds its type converter from the module's
// stamped `llvm.data_layout`, index width included — the upstream
// `convert-to-llvm` freezes a default converter (64-bit index) before ever
// seeing a module, so on a 32-bit-pointer target its output disagrees with
// every converter Reussir derives from the layout (most visibly the runtime
// declarations: `__reussir_allocate` at `i32` vs call sites at `i64`).

// A 32-bit-pointer (wasm32) module: index lowers to i32.
module @thirty_two attributes {
  llvm.data_layout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20",
  llvm.target_triple = "wasm32-unknown-unknown"
} {
  func.func @f() -> index {
    %0 = arith.constant 7 : index
    return %0 : index
  }
}

// CHECK-LABEL: module @thirty_two
// CHECK: llvm.func @f() -> i32
// CHECK: llvm.mlir.constant(7 : index) : i32

// -----

// Without a stamped layout the converter keeps MLIR's 64-bit default —
// unstamped modules behave exactly as under the upstream pass.
module @unstamped {
  func.func @g() -> index {
    %0 = arith.constant 7 : index
    return %0 : index
  }
}

// CHECK-LABEL: module @unstamped
// CHECK: llvm.func @g() -> i64
