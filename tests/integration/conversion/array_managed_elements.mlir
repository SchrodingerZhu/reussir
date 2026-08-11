// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=ACQ2D --check-prefix=ACQ2DLOOP --check-prefix=UNROLL3 --check-prefix=DROP32
// RUN: %reussir-opt %s --reussir-convert-to-std --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=CLONE
// RUN: %reussir-opt %s --reussir-acquire-drop-expansion --convert-scf-to-cf | %FileCheck %s --check-prefix=CF

!elt = !reussir.rc<i64>
!arr2 = !reussir.array<2 x !elt>
!arr3 = !reussir.array<3 x !elt>
!arr32 = !reussir.array<32 x !elt>
!arr2x2 = !reussir.array<2 x 2 x !elt>
!arr2x3 = !reussir.array<2 x 3 x !elt>
!rc_arr2 = !reussir.rc<!arr2>

module {
  // Four elements is exactly `kArrayOwnershipUnrollThreshold`, and the bound
  // is inclusive, so the largest nested shape that still unrolls: each row is
  // projected once and each of its elements once, with no loop anywhere.
  // ACQ2D-LABEL: func.func @acquire_2d(
  // ACQ2D: %[[VIEW:.+]] = reussir.array.view
  // ACQ2D-NOT: scf.for
  // ACQ2D: %[[ROW0:.+]] = reussir.array.project(%[[VIEW]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.array.project(%[[ROW0]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.rc.inc
  // ACQ2D: reussir.array.project(%[[ROW0]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.rc.inc
  // ACQ2D: %[[ROW1:.+]] = reussir.array.project(%[[VIEW]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.array.project(%[[ROW1]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.rc.inc
  // ACQ2D: reussir.array.project(%[[ROW1]]{{.*}}) [%{{.+}} : index]
  // ACQ2D: reussir.rc.inc
  // ACQ2D-NOT: reussir.rc.inc
  // ACQ2D-NOT: scf.for
  // ACQ2D: return
  func.func @acquire_2d(%xs: !reussir.ref<!arr2x2>) {
    reussir.ref.acquire (%xs : !reussir.ref<!arr2x2>)
    return
  }

  // Six elements is over the threshold, so the nested traversal becomes a
  // loop per dimension with one projection each and a single inc in the
  // innermost body. This is the multi-dimensional loop form; `acquire_2d`
  // above covers the unrolled one on the other side of the same bound.
  // ACQ2DLOOP-LABEL: func.func @acquire_2d_loop(
  // ACQ2DLOOP: %[[VIEW:.+]] = reussir.array.view
  // ACQ2DLOOP: scf.for %[[ROW:.+]] =
  // ACQ2DLOOP: scf.for %[[COLUMN:.+]] =
  // ACQ2DLOOP: %[[ROW_VIEW:.+]] = reussir.array.project(%[[VIEW]]{{.*}}) [%[[ROW]] : index]
  // ACQ2DLOOP: %[[ELEMENT:.+]] = reussir.array.project(%[[ROW_VIEW]]{{.*}}) [%[[COLUMN]] : index]
  // ACQ2DLOOP: reussir.rc.inc
  // ACQ2DLOOP-NOT: reussir.rc.inc
  // ACQ2DLOOP: return
  func.func @acquire_2d_loop(%xs: !reussir.ref<!arr2x3>) {
    reussir.ref.acquire (%xs : !reussir.ref<!arr2x3>)
    return
  }

  // UNROLL3-LABEL: func.func @drop_3(
  // UNROLL3-NOT: scf.for
  // UNROLL3: reussir.rc.dec
  // UNROLL3-NOT: scf.for
  // UNROLL3: reussir.rc.dec
  // UNROLL3-NOT: scf.for
  // UNROLL3: reussir.rc.dec
  // UNROLL3-NOT: scf.for
  // UNROLL3-NOT: reussir.rc.dec
  // UNROLL3: return
  func.func @drop_3(%xs: !reussir.ref<!arr3>) {
    reussir.ref.drop (%xs : !reussir.ref<!arr3>)
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
  // CLONE-NOT: scf.for
  // CLONE: reussir.array.project(%[[CLONED_VIEW]]
  // CLONE: reussir.rc.inc
  // CLONE: reussir.array.project(%[[CLONED_VIEW]]
  // CLONE: reussir.rc.inc
  // CLONE-NOT: scf.for
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
