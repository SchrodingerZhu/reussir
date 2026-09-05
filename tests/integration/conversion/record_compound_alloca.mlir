// RUN: %reussir-opt %s --convert-to-llvm | %reussir-translate --mlir-to-llvmir | %FileCheck %s

// A padded compound built on a conditional path must still use a static
// entry-block slot. Non-entry allocas block LLVM's tail-recursion pass even
// when their size is constant; padding loads can prevent their early removal.
!pair = !reussir.record<compound "PaddedPair" [value] {i64, i32}>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @conditional_pair(%make : i1, %first : i64, %second : i32,
                              %fallback : !pair) -> !pair {
    cf.cond_br %make, ^construct, ^other
  ^construct:
    %pair = reussir.record.compound(%first, %second : i64, i32) : !pair
    return %pair : !pair
  ^other:
    return %fallback : !pair
  }
}

// CHECK: %PaddedPair = type { i64, i32 }
// CHECK-LABEL: define %PaddedPair @conditional_pair(
// CHECK: %[[SLOT:[0-9]+]] = alloca %PaddedPair, i64 1, align 8
// CHECK-NOT: store
// CHECK-NOT: lifetime.start
// CHECK: br i1 %0, label %[[CONSTRUCT:[0-9]+]], label %[[OTHER:[0-9]+]]
// CHECK: [[CONSTRUCT]]:
// CHECK-NOT: alloca
// CHECK: call void @llvm.lifetime.start.p0(ptr %[[SLOT]])
// CHECK: store i64 %1,
// CHECK: store i32 %2,
// CHECK: %[[VALUE:[0-9]+]] = load %PaddedPair, ptr %[[SLOT]], align 8
// CHECK: call void @llvm.lifetime.end.p0(ptr %[[SLOT]])
// CHECK: ret %PaddedPair %[[VALUE]]
// CHECK: [[OTHER]]:
// CHECK-NOT: alloca
// CHECK-NOT: store
// CHECK-NOT: lifetime.start
// CHECK: ret %PaddedPair %3
