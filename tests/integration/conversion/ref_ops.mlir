// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s
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
  // CHECK: %2 = llvm.intr.invariant.start 8, %1 : !llvm.ptr
  // CHECK: %3 = llvm.load %1 : !llvm.ptr -> i64
  // CHECK: llvm.return %3 : i64
  // CHECK: }

  func.func @reference_project(%struct_ref: !reussir.ref<!reussir.record<compound "TestStruct" {i64, i64}>>) -> !reussir.ref<i64> {
      %field_ref = reussir.ref.project (%struct_ref : !reussir.ref<!reussir.record<compound "TestStruct" {i64, i64}>>) [0] : !reussir.ref<i64>
      return %field_ref : !reussir.ref<i64>
  }
  // CHECK-LABEL: llvm.func @reference_project(%arg0: !llvm.ptr) -> !llvm.ptr
  // CHECK: %[[result:[0-9]+]] = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.return %[[result]] : !llvm.ptr
  // CHECK: }

  func.func @reference_diff(%base: !reussir.ref<i64>, %target: !reussir.ref<i64>) -> index {
      %diff = reussir.ref.diff %base, %target : (!reussir.ref<i64>, !reussir.ref<i64>) -> index
      return %diff : index
  }
  // CHECK-LABEL: llvm.func @reference_diff(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i64
  // CHECK: %[[base:[0-9]+]] = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  // CHECK: %[[target:[0-9]+]] = llvm.ptrtoint %arg1 : !llvm.ptr to i64
  // CHECK: %[[diff:[0-9]+]] = llvm.sub %[[target]], %[[base]] overflow<nsw> : i64
  // CHECK: llvm.return %[[diff]] : i64
  // CHECK: }

  func.func @reference_cmp(%lhs: !reussir.ref<i64>, %rhs: !reussir.ref<i64>) -> i1 {
      %flag = reussir.ref.cmp eq %lhs, %rhs : (!reussir.ref<i64>, !reussir.ref<i64>) -> i1
      return %flag : i1
  }
  // CHECK-LABEL: llvm.func @reference_cmp(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i1
  // CHECK: %[[lhs:[0-9]+]] = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  // CHECK: %[[rhs:[0-9]+]] = llvm.ptrtoint %arg1 : !llvm.ptr to i64
  // CHECK: %[[cmp:[0-9]+]] = llvm.icmp "eq" %[[lhs]], %[[rhs]] : i64
  // CHECK: llvm.return %[[cmp]] : i1
  // CHECK: }

  func.func @reference_memcpy(%src: !reussir.ref<i64>, %dst: !reussir.ref<i64>) {
      reussir.ref.memcpy %src to %dst : !reussir.ref<i64> to !reussir.ref<i64>
      return
  }
  // CHECK-LABEL: llvm.func @reference_memcpy(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
  // CHECK: %[[LEN:.+]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: "llvm.intr.memcpy"(%arg1, %arg0, %[[LEN]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // CHECK: llvm.return
  // CHECK: }

}
