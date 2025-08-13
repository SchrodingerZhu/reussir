// RUN: %reussir-opt %s --reussir-lowering-basic-ops
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // CHECK-LABEL: llvm.func @token_alloc() -> !llvm.ptr attributes {sym_visibility = "private"} {
  // CHECK: %0 = llvm.mlir.constant(8 : index) : i64
  // CHECK: %1 = llvm.mlir.constant(64 : index) : i64
  // CHECK: %2 = llvm.call @__reussir_allocate(%0, %1) : (i64, i64) -> !llvm.ptr
  // CHECK: llvm.return %2 : !llvm.ptr
  // CHECK: }
  func.func private @token_alloc() 
    -> !reussir.token<align: 8, size: 64> {
      %token = reussir.token.alloc : !reussir.token<align: 8, size: 64>
      return %token : !reussir.token<align: 8, size: 64>
  }
  
  // CHECK-LABEL: llvm.func @token_free(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
  // CHECK: %0 = llvm.mlir.constant(8 : index) : i64
  // CHECK: %1 = llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.call @__reussir_deallocate(%arg0, %0, %1) : (!llvm.ptr, i64, i64) -> ()
  // CHECK: llvm.return
  // CHECK: }
  func.func private @token_free(%token: !reussir.token<align: 8, size: 64>) {
      reussir.token.free (%token : !reussir.token<align: 8, size: 64>)
      return
  }
 
  // CHECK-LABEL: llvm.func @token_reinterpret(%arg0: !llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"} {
  // CHECK: llvm.return %arg0 : !llvm.ptr
  // CHECK: }
  func.func private @token_reinterpret(%token: !reussir.token<align: 8, size: 8>) 
    -> !reussir.ref<i64> {
      %reinterpreted = reussir.token.reinterpret 
        (%token : !reussir.token<align: 8, size: 8>) 
        : !reussir.ref<i64>
      return %reinterpreted : !reussir.ref<i64>
  }
  
  // CHECK-LABEL: llvm.func @token_realloc(%arg0: !llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"} {
  // CHECK: %0 = llvm.mlir.constant(8 : index) : i64
  // CHECK: %1 = llvm.mlir.constant(8 : index) : i64
  // CHECK: %2 = llvm.mlir.constant(8 : index) : i64
  // CHECK: %3 = llvm.mlir.constant(16 : index) : i64
  // CHECK: %4 = llvm.call @__reussir_reallocate(%arg0, %0, %1, %2, %3) : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr
  // CHECK: llvm.return %4 : !llvm.ptr
  // CHECK: }
  func.func private @token_realloc(%token: !reussir.token<align: 8, size: 8>) -> 
    !reussir.token<align: 8, size: 16> {
      %reallocated = reussir.token.realloc 
        (%token : !reussir.token<align: 8, size: 8>) 
        : !reussir.token<align: 8, size: 16>
      return %reallocated : !reussir.token<align: 8, size: 16>
  }

}
