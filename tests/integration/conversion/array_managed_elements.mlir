// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=ACQ2D --check-prefix=DROP32
// RUN: %reussir-opt %s --reussir-convert-to-std --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=CLONE
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion --convert-scf-to-cf | %FileCheck %s --check-prefix=CF

!elt = !reussir.rc<i64>
!arr2 = !reussir.array<2 x !elt>
!arr32 = !reussir.array<32 x !elt>
!arr2x2 = !reussir.array<2 x 2 x !elt>
!rc_arr2 = !reussir.rc<!arr2>

module {
  // ACQ2D-LABEL: func.func @acquire_2d(
  // ACQ2D: %[[VIEW:.+]] = reussir.array.view
  // ACQ2D: scf.for %[[ROW:.+]] =
  // ACQ2D: scf.for %[[COLUMN:.+]] =
  // ACQ2D: %[[ROW_VIEW:.+]] = reussir.array.project(%[[VIEW]]{{.*}}) [%[[ROW]] : index]
  // ACQ2D: %[[ELEMENT:.+]] = reussir.array.project(%[[ROW_VIEW]]{{.*}}) [%[[COLUMN]] : index]
  // ACQ2D: reussir.rc.inc
  func.func @acquire_2d(%xs: !reussir.ref<!arr2x2>) {
    reussir.ref.acquire (%xs : !reussir.ref<!arr2x2>)
    return
  }

  // DROP32-LABEL: func.func @drop_32(
  // DROP32: %[[DROP_VIEW:.+]] = reussir.array.view
  // DROP32: scf.for %[[INDEX:.+]] =
  // DROP32: reussir.array.project(%[[DROP_VIEW]]{{.*}}) [%[[INDEX]] : index]
  // DROP32: reussir.rc.dec
  // DROP32-NOT: reussir.rc.dec
  // DROP32: return
  // CF-LABEL: func.func @drop_32(
  // CF: cf.br ^[[HEADER:bb[0-9]+]](
  // CF: ^[[HEADER]](%[[IV:[0-9]+]]: index):
  // CF: cf.cond_br
  // CF: reussir.array.project{{.*}}[%[[IV]] : index]
  // CF: reussir.rc.dec
  // CF: cf.br ^[[HEADER]](
  func.func @drop_32(%xs: !reussir.ref<!arr32>) {
    reussir.ref.drop (%xs : !reussir.ref<!arr32>)
    return
  }

  // CLONE-LABEL: func.func @clone_managed(
  // CLONE: %[[POISON:.+]] = ub.poison : !reussir.array<2 x !reussir.rc<i64>>
  // CLONE: %[[IS_UNIQUE:.+]] = reussir.rc.is_unique
  // CLONE: scf.if %[[IS_UNIQUE]] -> (!reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>) {
  // CLONE: } else {
  // CLONE: %[[SRC_BORROW:.+]] = reussir.rc.borrow(%arg0 : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>) : !reussir.ref<!reussir.array<2 x !reussir.rc<i64>>>
  // CLONE: %[[TOKEN:.+]] = reussir.token.alloc
  // CLONE: %[[CLONED:.+]] = reussir.rc.create value(%[[POISON]] : !reussir.array<2 x !reussir.rc<i64>>) token(%[[TOKEN]] : !reussir.token<align : 8, size : 24>) : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>
  // CLONE: %[[DST_BORROW:.+]] = reussir.rc.borrow(%[[CLONED]] : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>) : !reussir.ref<!reussir.array<2 x !reussir.rc<i64>>>
  // CLONE: reussir.ref.memcpy %[[SRC_BORROW]] to %[[DST_BORROW]] : <!reussir.array<2 x !reussir.rc<i64>>> to <!reussir.array<2 x !reussir.rc<i64>>>
  // CLONE: %[[CLONED_VIEW:.+]] = reussir.array.view(%[[DST_BORROW]] : !reussir.ref<!reussir.array<2 x !reussir.rc<i64>>>) : memref<2x!reussir.rc<i64>>
  // CLONE: scf.for %[[CLONE_INDEX:.+]] =
  // CLONE: reussir.array.project(%[[CLONED_VIEW]]{{.*}}) [%[[CLONE_INDEX]] : index]
  // CLONE: reussir.rc.inc
  // CLONE: %[[COUNT:.+]] = reussir.rc.fetch(%arg0 : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>) : index
  // CLONE: reussir.rc.set(%arg0 : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>,
  // CLONE: scf.yield %[[CLONED]] : !reussir.rc<!reussir.array<2 x !reussir.rc<i64>>>
  func.func @clone_managed(%xs: !rc_arr2) -> !rc_arr2 {
    %res = reussir.array.with_unique_view (%xs : !rc_arr2) -> !rc_arr2 {
      ^bb0(%view: memref<2x!reussir.rc<i64>>):
        reussir.scf.yield
    }
    return %res : !rc_arr2
  }
}
