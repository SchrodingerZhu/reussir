// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s

!arr4 = !reussir.array<4 x i8>
!rc_arr4 = !reussir.rc<!arr4>

module {
  func.func @update0(%xs: !rc_arr4) -> !rc_arr4 {
    %res = reussir.array.with_unique_view (%xs : !rc_arr4) -> !rc_arr4 {
      ^bb0(%view: memref<4xi8>):
        %c0 = arith.constant 0 : index
        %old = memref.load %view[%c0] : memref<4xi8>
        %one = arith.constant 1 : i8
        %new = arith.addi %old, %one : i8
        memref.store %new, %view[%c0] : memref<4xi8>
        reussir.scf.yield
    }
    return %res : !rc_arr4
  }
}

// CHECK-LABEL: func.func @update0(
// CHECK: %[[IS_UNIQUE:.+]] = reussir.rc.is_unique
// CHECK: %[[RESULT:.+]] = scf.if %[[IS_UNIQUE]] -> (!reussir.rc<!reussir.array<4 x i8>>) {
// CHECK: %[[BORROWED:.+]] = reussir.rc.borrow
// CHECK: %[[VIEW:.+]] = reussir.array.view(%[[BORROWED]] : !reussir.ref<!reussir.array<4 x i8>>) : memref<4xi8>
// CHECK: memref.load %[[VIEW]]
// CHECK: memref.store
// CHECK: scf.yield %arg0 : !reussir.rc<!reussir.array<4 x i8>>
// CHECK: } else {
// CHECK: %[[SRC_BORROW:.+]] = reussir.rc.borrow(%arg0 : !reussir.rc<!reussir.array<4 x i8>>) : !reussir.ref<!reussir.array<4 x i8>>
// CHECK: %[[TOKEN:.+]] = reussir.token.alloc
// CHECK: %[[POISON:.+]] = ub.poison : !reussir.array<4 x i8>
// CHECK: %[[CLONED:.+]] = reussir.rc.create
// CHECK: %[[CLONED_BORROW:.+]] = reussir.rc.borrow(%[[CLONED]] : !reussir.rc<!reussir.array<4 x i8>>) : !reussir.ref<!reussir.array<4 x i8>>
// CHECK: reussir.ref.memcpy %[[SRC_BORROW]] to %[[CLONED_BORROW]] : <!reussir.array<4 x i8>> to <!reussir.array<4 x i8>>
// CHECK: reussir.ref.acquire(%[[CLONED_BORROW]] : !reussir.ref<!reussir.array<4 x i8>>)
// CHECK: %[[COUNT:.+]] = reussir.rc.fetch(%arg0 : !reussir.rc<!reussir.array<4 x i8>>) : index
// CHECK: %[[DEC:.+]] = arith.subi %[[COUNT]], %{{.+}} : index
// CHECK: reussir.rc.set(%arg0 : !reussir.rc<!reussir.array<4 x i8>>, %[[DEC]] : index)
// CHECK: %[[CLONED_VIEW:.+]] = reussir.array.view(%[[CLONED_BORROW]] : !reussir.ref<!reussir.array<4 x i8>>) : memref<4xi8>
// CHECK: memref.load %[[CLONED_VIEW]]
// CHECK: memref.store
// CHECK: scf.yield %[[CLONED]] : !reussir.rc<!reussir.array<4 x i8>>
// CHECK: }
// CHECK: return %[[RESULT]] : !reussir.rc<!reussir.array<4 x i8>>
// CHECK-NOT: reussir.rc.dec
