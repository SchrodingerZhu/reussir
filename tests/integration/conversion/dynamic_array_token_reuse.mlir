// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-token-reuse))' -o %t.mlir
// RUN: %FileCheck %s < %t.mlir
// RUN: %reussir-opt %t.mlir --pass-pipeline='builtin.module(reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,reussir-convert-to-llvm,reconcile-unrealized-casts)' -o %t.llvm.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll

!array = !reussir.array<? x i32>
!rc = !reussir.rc<!array>
!wide = !reussir.rc<!reussir.array<? x i128>>

module {
  // Equal token types do not imply equal runtime sizes. This used to select
  // token.ensure, which also lost the required size on its null branch.
  // CHECK-LABEL: func.func @dynamic_to_dynamic
  // CHECK: %[[BYTES:.+]] = arith.addi
  // CHECK: %[[ALLOC:.+]] = reussir.token.alloc(%[[BYTES]] : index)
  // CHECK: reussir.rc.create
  // CHECK-SAME: token(%[[ALLOC]]
  // CHECK-SAME: extents(%arg1)
  // CHECK: return
  func.func @dynamic_to_dynamic(%old: !rc, %n: index) -> !rc {
    reussir.rc.dec (%old : !rc)
    %poison = ub.poison : !array
    %new = reussir.rc.create value(%poison : !array) extents(%n) : !rc
    return %new : !rc
  }

  // A dynamic donor with a different alignment used to select realloc with
  // the dynamic size sentinel as the requested byte count.
  // CHECK-LABEL: func.func @different_alignment
  // CHECK: %[[BYTES:.+]] = arith.addi
  // CHECK: %[[ALLOC:.+]] = reussir.token.alloc(%[[BYTES]] : index)
  // CHECK: reussir.rc.create
  // CHECK-SAME: token(%[[ALLOC]]
  // CHECK-SAME: extents(%arg1)
  // CHECK: return
  func.func @different_alignment(%old: !wide, %n: index) -> !rc {
    reussir.rc.dec (%old : !wide)
    %poison = ub.poison : !array
    %new = reussir.rc.create value(%poison : !array) extents(%n) : !rc
    return %new : !rc
  }

  // Fixed-size donors must not consume a dynamic recipient's allocation.
  // CHECK-LABEL: func.func @static_to_dynamic
  // CHECK: %[[BYTES:.+]] = arith.addi
  // CHECK: %[[ALLOC:.+]] = reussir.token.alloc(%[[BYTES]] : index)
  // CHECK: reussir.rc.create
  // CHECK-SAME: token(%[[ALLOC]]
  // CHECK-SAME: extents(%arg1)
  // CHECK: return
  func.func @static_to_dynamic(%old: !reussir.rc<i32>, %n: index) -> !rc {
    reussir.rc.dec (%old : !reussir.rc<i32>)
    %poison = ub.poison : !array
    %new = reussir.rc.create value(%poison : !array) extents(%n) : !rc
    return %new : !rc
  }

  // Preserve the supported direction: a dynamic box can still donate to a
  // statically sized recipient through the unsized realloc ABI.
  // CHECK-LABEL: func.func @dynamic_to_static
  // CHECK: %[[REUSED:.+]] = reussir.token.realloc
  // CHECK: reussir.rc.create
  // CHECK-SAME: token(%[[REUSED]]
  // CHECK: return
  func.func @dynamic_to_static(%old: !rc, %value: i32) -> !reussir.rc<i32> {
    reussir.rc.dec (%old : !rc)
    %new = reussir.rc.create value(%value : i32) : !reussir.rc<i32>
    return %new : !reussir.rc<i32>
  }
}
