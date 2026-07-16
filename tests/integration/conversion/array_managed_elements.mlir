// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=ACQ2D --check-prefix=DROP1D
// RUN: %reussir-opt %s --reussir-convert-to-std --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=CLONE

!elt = !reussir.rc<i64>
!arr2 = !reussir.array<2 x !elt>
!arr2x2 = !reussir.array<2 x 2 x !elt>
!rc_arr2 = !reussir.rc<!arr2>

module {
  // ACQ2D-LABEL: func.func @acquire_2d(
  // ACQ2D: reussir.array.view
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.rc.inc
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.rc.inc
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.rc.inc
  // ACQ2D: reussir.array.project
  // ACQ2D: reussir.rc.inc
  func.func @acquire_2d(%xs: !reussir.ref<!arr2x2>) {
    reussir.ref.acquire (%xs : !reussir.ref<!arr2x2>)
    return
  }

  // DROP1D-LABEL: func.func @drop_1d(
  // DROP1D: reussir.array.view
  // DROP1D: reussir.array.project
  // DROP1D: reussir.rc.dec
  // DROP1D: reussir.array.project
  // DROP1D: reussir.rc.dec
  func.func @drop_1d(%xs: !reussir.ref<!arr2>) {
    reussir.ref.drop (%xs : !reussir.ref<!arr2>)
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
  // CLONE: reussir.array.view(%[[DST_BORROW]] : !reussir.ref<!reussir.array<2 x !reussir.rc<i64>>>) : memref<2x!reussir.rc<i64>>
  // CLONE: reussir.rc.inc
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
