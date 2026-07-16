// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s

!arr4 = !reussir.array<4 x i8>
!rc_arr4 = !reussir.rc<!arr4>

module {
  func.func @borrow_tensor(%xs: !rc_arr4) -> i8 {
    %borrow = reussir.rc.borrow (%xs : !rc_arr4) : !reussir.ref<!arr4>
    %view = reussir.array.view(%borrow : !reussir.ref<!arr4>) : tensor<4xi8>
    %c0 = arith.constant 0 : index
    %value = tensor.extract %view[%c0] : tensor<4xi8>
    return %value : i8
  }

  func.func @update0_tensor(%xs: !rc_arr4) -> i8 {
    %res = reussir.array.with_unique_view (%xs : !rc_arr4) -> i8 {
      ^bb0(%view: tensor<4xi8>):
        %zero = arith.constant 0 : i8
        %filled = linalg.fill ins(%zero : i8) outs(%view : tensor<4xi8>) -> tensor<4xi8>
        %c0 = arith.constant 0 : index
        %loaded = tensor.extract %filled[%c0] : tensor<4xi8>
        reussir.scf.yield %loaded : i8
    }
    return %res : i8
  }
}

// CHECK-LABEL: func.func @borrow_tensor(
// CHECK: %[[BORROW:.+]] = reussir.rc.borrow
// CHECK: %[[MEMREF:.+]] = reussir.array.view(%[[BORROW]] : !reussir.ref<!reussir.array<4 x i8>>) : memref<4xi8>
// CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[MEMREF]] restrict : memref<4xi8> to tensor<4xi8>
// CHECK: %[[VALUE:.+]] = tensor.extract %[[TENSOR]]
// CHECK: return %[[VALUE]] : i8

// CHECK-LABEL: func.func @update0_tensor(
// CHECK: %[[IS_UNIQUE:.+]] = reussir.rc.is_unique
// CHECK: %[[RESULT:.+]] = scf.if %[[IS_UNIQUE]] -> (i8) {
// CHECK: %[[BORROWED:.+]] = reussir.rc.borrow
// CHECK: %[[VIEW_MEMREF:.+]] = reussir.array.view(%[[BORROWED]] : !reussir.ref<!reussir.array<4 x i8>>) : memref<4xi8>
// CHECK: %[[VIEW_TENSOR:.+]] = bufferization.to_tensor %[[VIEW_MEMREF]] restrict writable : memref<4xi8> to tensor<4xi8>
// CHECK: %[[FILLED:.+]] = linalg.fill ins(%{{.+}} : i8) outs(%[[VIEW_TENSOR]] : tensor<4xi8>) -> tensor<4xi8>
// CHECK: tensor.extract %[[FILLED]]
// CHECK: } else {
// CHECK: %[[CLONED_BORROW:.+]] = reussir.rc.borrow(%{{.+}} : !reussir.rc<!reussir.array<4 x i8>>) : !reussir.ref<!reussir.array<4 x i8>>
// CHECK: %[[CLONED_VIEW_MEMREF:.+]] = reussir.array.view(%{{.+}} : !reussir.ref<!reussir.array<4 x i8>>) : memref<4xi8>
// CHECK: %[[CLONED_VIEW_TENSOR:.+]] = bufferization.to_tensor %[[CLONED_VIEW_MEMREF]] restrict writable : memref<4xi8> to tensor<4xi8>
// CHECK: linalg.fill ins(%{{.+}} : i8) outs(%[[CLONED_VIEW_TENSOR]] : tensor<4xi8>) -> tensor<4xi8>
// CHECK: }
// CHECK: return %[[RESULT]] : i8
