// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  // CHECK-LABEL: llvm.func @token_alloc() -> !llvm.ptr attributes {sym_visibility = "private"} {
  // CHECK: %0 = llvm.mlir.constant(64 : [[INDEX_T:i[0-9]+]]) : [[INDEX_T]]
  // CHECK: %1 = llvm.call @__reussir_allocate_small(%0) : ([[INDEX_T]]) -> !llvm.ptr
  // CHECK: llvm.return %1 : !llvm.ptr
  // CHECK: }
  func.func private @token_alloc() 
    -> !reussir.token<align: 8, size: 64> {
      %token = reussir.token.alloc : !reussir.token<align: 8, size: 64>
      return %token : !reussir.token<align: 8, size: 64>
  }
  
  // CHECK-LABEL: llvm.func @token_free(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
  // CHECK: %0 = llvm.mlir.constant(8 : [[INDEX_T]]) : [[INDEX_T]]
  // CHECK: %1 = llvm.mlir.constant(64 : [[INDEX_T]]) : [[INDEX_T]]
  // CHECK: llvm.call @__reussir_deallocate(%arg0, %0, %1) : (!llvm.ptr, [[INDEX_T]], [[INDEX_T]]) -> ()
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
  
  // A realloc lowers to `__reussir_reallocate` followed by an invariant-group
  // launder of the result: an LTO'd allocator fast path may return the
  // identical pointer, whose stale invariant-group metadata must be stripped;
  // the ptr-eq assume keeps value propagation across the barrier.
  // CHECK-LABEL: llvm.func @token_realloc(%arg0: !llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"} {
  // CHECK: %[[R:.+]] = llvm.call @__reussir_reallocate(%arg0, {{.*}}) : (!llvm.ptr, [[INDEX_T]], [[INDEX_T]], [[INDEX_T]], [[INDEX_T]]) -> !llvm.ptr
  // CHECK: %[[L:.+]] = llvm.intr.launder.invariant.group %[[R]] : !llvm.ptr
  // CHECK: %[[EQ:.+]] = llvm.icmp "eq" %[[L]], %[[R]] : !llvm.ptr
  // CHECK: llvm.intr.assume %[[EQ]] : i1
  // CHECK: llvm.return %[[L]] : !llvm.ptr
  // CHECK: }
  func.func private @token_realloc(%token: !reussir.token<align: 8, size: 8>) -> 
    !reussir.token<align: 8, size: 16> {
      %reallocated = reussir.token.realloc 
        (%token : !reussir.token<align: 8, size: 8>) 
        : !reussir.token<align: 8, size: 16>
      return %reallocated : !reussir.token<align: 8, size: 16>
  }

}
