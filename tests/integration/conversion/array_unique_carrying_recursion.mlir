// RUN: %reussir-opt %s --reussir-unique-carrying-recursion-analysis | %FileCheck %s
// RUN: %reussir-opt %s \
// RUN:   --reussir-unique-carrying-recursion-analysis \
// RUN:   --reussir-convert-to-std \
// RUN:   --reussir-rc-decrement-expansion \
// RUN:   --reussir-acquire-drop-expansion \
// RUN:   --reussir-acquire-drop-expansion='expand-decrement=1 outline-record=1' \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-to-llvm | \
// RUN:   %reussir-translate --mlir-to-llvmir | \
// RUN:   %opt -passes='default<O3>' -pass-remarks=loop-vectorize -disable-output 2>&1 | \
// RUN:   %FileCheck %s --check-prefix=VEC

!arr32768 = !reussir.array<32768 x i8>
!rc_arr32768 = !reussir.rc<!arr32768>

module {
  // CHECK-LABEL: func.func private @inc_1(
  // CHECK: attributes {llvm.linkage = #llvm.linkage<internal>, reussir.carrying_uniqueness}
  // CHECK: %[[UPDATED:.+]] = reussir.array.with_unique_view
  // CHECK: func.call @inc_1.unique(%[[UPDATED]], %{{.+}}) : (!reussir.rc<!reussir.array<32768 x i8>>, index) -> !reussir.rc<!reussir.array<32768 x i8>>
  func.func private @inc_1(%xs: !rc_arr32768, %i: index) -> !rc_arr32768 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c32768 = arith.constant 32768 : index
    %cond = arith.cmpi ult, %i, %c32768 : index
    %result = scf.if %cond -> (!rc_arr32768) {
      %updated = reussir.array.with_unique_view (%xs : !rc_arr32768) -> !rc_arr32768 {
        ^bb0(%view: memref<32768xi8>):
          %old = memref.load %view[%i] : memref<32768xi8>
          %one = arith.constant 1 : i8
          %new = arith.addi %old, %one : i8
          memref.store %new, %view[%i] : memref<32768xi8>
          reussir.scf.yield
      }
      %c1 = arith.constant 1 : index
      %next = arith.addi %i, %c1 : index
      %rec = func.call @inc_1(%updated, %next) : (!rc_arr32768, index) -> !rc_arr32768
      scf.yield %rec : !rc_arr32768
    } else {
      scf.yield %xs : !rc_arr32768
    }
    return %result : !rc_arr32768
  }

  // CHECK-LABEL: func.func @inc_1_export(
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[RES:.+]] = call @inc_1(%arg0, %[[C0]]) : (!reussir.rc<!reussir.array<32768 x i8>>, index) -> !reussir.rc<!reussir.array<32768 x i8>>
  // CHECK: return %[[RES]] : !reussir.rc<!reussir.array<32768 x i8>>
  func.func @inc_1_export(%xs: !rc_arr32768) -> !rc_arr32768 {
    %c0 = arith.constant 0 : index
    %res = func.call @inc_1(%xs, %c0) : (!rc_arr32768, index) -> !rc_arr32768
    return %res : !rc_arr32768
  }

  // CHECK-LABEL: func.func private @inc_1.unique(
  // CHECK: reussir.rc.assume_unique(%arg0 : !reussir.rc<!reussir.array<32768 x i8>>)
  // CHECK: %[[UPDATED2:.+]] = reussir.array.with_unique_view
  // CHECK: func.call @inc_1.unique(%[[UPDATED2]], %{{.+}}) : (!reussir.rc<!reussir.array<32768 x i8>>, index) -> !reussir.rc<!reussir.array<32768 x i8>>

  // VEC: remark: <unknown>:0:0: vectorized loop
}
