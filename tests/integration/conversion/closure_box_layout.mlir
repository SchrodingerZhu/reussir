// RUN: %reussir-opt %s --reussir-token-instantiation | \
// RUN:   %FileCheck %s --check-prefix=TOKEN
// RUN: %reussir-opt %s --reussir-token-instantiation \
// RUN:   --reussir-closure-outlining --reussir-convert-to-llvm | \
// RUN:   %FileCheck %s --check-prefix=LLVM

// A wasm32 closure box remains pointer-aligned even when its payload requires
// stronger alignment. The explicit padding keeps the cursor and payload
// correctly positioned without allowing the payload to move the header.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<32> : vector<4xi64>,
    i64 = dense<64> : vector<2xi64>,
    i32 = dense<32> : vector<2xi64>,
    i8 = dense<8> : vector<2xi64>>,
  llvm.data_layout = "e-m:e-p:32:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "wasm32-unknown-wasip1"
} {
  // TOKEN-LABEL: func.func @make
  // TOKEN: %[[TOKEN:.*]] = reussir.token.alloc : <align : 4, size : 24>
  // TOKEN: reussir.closure.create
  // TOKEN: token (%[[TOKEN]] : <align : 4, size : 24>)
  func.func @make() -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
    %closure = reussir.closure.create
        -> !reussir.rc<!reussir.closure<(i64) -> i64>> {
      body {
      ^bb0(%value: i64):
        reussir.closure.yield %value : i64
      }
    }
    return %closure : !reussir.rc<!reussir.closure<(i64) -> i64>>
  }

  // LLVM-LABEL: llvm.func @make
  // LLVM: llvm.getelementptr {{.*}}[0, 1, 1] : {{.*}}, !llvm.struct<(i32, struct<(ptr, ptr, struct<packed (array<4 x i8>, struct<(i64)>)>)>)>
  // LLVM: llvm.getelementptr {{.*}}[0, 1, 2, 1] : {{.*}}, !llvm.struct<(i32, struct<(ptr, ptr, struct<packed (array<4 x i8>, struct<(i64)>)>)>)>
}
