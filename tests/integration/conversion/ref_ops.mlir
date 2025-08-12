// RUN: %reussir-opt %s --reussir-lowering-basic-ops
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {

  func.func @reference_load(%ref: !reussir.ref<i64>) 
    -> i64 {
      %value = reussir.ref.load (%ref : !reussir.ref<i64>) : i64
      return %value : i64
  }
  // CHECK-LABEL: llvm.func @reference_load(%arg0: !llvm.ptr) -> i64
  // CHECK: %0 = llvm.load %arg0 : !llvm.ptr -> i64
  // CHECK: llvm.return %0 : i64
  // CHECK: }
  
  func.func @reference_store(%ref: !reussir.ref<i64 field>, %value: i64) {
      reussir.ref.store (%ref : !reussir.ref<i64 field>) (%value : i64)
      return
  }
  // CHECK-LABEL: llvm.func @reference_store(%arg0: !llvm.ptr, %arg1: i64)
  // CHECK: llvm.store %arg1, %arg0 : i64, !llvm.ptr
  // CHECK: llvm.return
  // CHECK: }

  func.func @reference_spill(%value: i64) -> i64 {
      %spilled = reussir.ref.spilled (%value : i64) : !reussir.ref<i64>
      %load = reussir.ref.load (%spilled : !reussir.ref<i64>) : i64
      return %load : i64
  }
  // CHECK-LABEL: llvm.func @reference_spill(%arg0: i64) -> i64
  // CHECK: %0 = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK: llvm.store %arg0, %1 : i64, !llvm.ptr
  // CHECK: %2 = llvm.load %1 : !llvm.ptr -> i64
  // CHECK: llvm.return %2 : i64
  // CHECK: }

}
