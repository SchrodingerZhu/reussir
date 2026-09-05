// RUN: %reussir-opt %s --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=DROP
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=CLONE
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' | %FileCheck %s --check-prefix=LLVM

// Ownership and cloning of a dynamic-extent array
// (docs/design/dynamic-extent-arrays.md): the element traversal never
// unrolls a dynamic shape, takes its loop bounds from `memref.dim` on the
// view, and projects through the dynamic strided layout at every rank; the
// `with_unique_view` clone branch sizes its token at runtime, constructs the
// clone canonical from the source's sizes, and copies the payload flat when
// the source header is canonical or element-wise otherwise.

!elt = !reussir.rc<i64>
!dv = !reussir.array<? x 2 x !elt>
!rc_dv = !reussir.rc<!dv>
!view = memref<?x2x!elt, strided<[?, ?], offset: ?>>
!row = memref<2x!elt, strided<[?], offset: ?>>

module {
  // DROP-LABEL: func.func @drop_dynamic(
  // DROP: %[[VIEW:.+]] = reussir.array.view(%arg0 : !reussir.ref<!reussir.array<? x 2 x !reussir.rc<i64>>>) : memref<?x2x!reussir.rc<i64>, strided<[?, ?], offset: ?>>
  // DROP: %[[DIM:.+]] = memref.dim %[[VIEW]], %{{.+}} : memref<?x2x!reussir.rc<i64>, strided<[?, ?], offset: ?>>
  // DROP: scf.for %[[ROW:.+]] = %{{.+}} to %[[DIM]]
  // DROP: scf.for %[[COLUMN:.+]] = %{{.+}} to %{{.+}}
  // DROP: %[[ROW_VIEW:.+]] = reussir.array.project(%[[VIEW]] : memref<?x2x!reussir.rc<i64>, strided<[?, ?], offset: ?>>) [%[[ROW]] : index] : memref<2x!reussir.rc<i64>, strided<[?], offset: ?>>
  // DROP: reussir.array.project(%[[ROW_VIEW]] : memref<2x!reussir.rc<i64>, strided<[?], offset: ?>>) [%[[COLUMN]] : index] : !reussir.ref<!reussir.rc<i64>>
  // DROP: reussir.rc.dec
  // DROP-NOT: reussir.rc.dec
  // DROP: return
  func.func @drop_dynamic(%xs: !reussir.ref<!dv>) {
    reussir.ref.drop (%xs : !reussir.ref<!dv>)
    return
  }

  // CLONE-LABEL: func.func @clone_dynamic(
  // CLONE: } else {
  // CLONE: %[[SRC:.+]] = reussir.rc.borrow(%arg0 : !reussir.rc<!reussir.array<? x 2 x !reussir.rc<i64>>>)
  // CLONE: %[[SRC_VIEW:.+]] = reussir.array.view(%[[SRC]] : {{.*}}) : memref<?x2x!reussir.rc<i64>, strided<[?, ?], offset: ?>>
  // CLONE: %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZES:.+]]:2, %[[STRIDES:.+]]:2 = memref.extract_strided_metadata %[[SRC_VIEW]]
  // CLONE: %[[C1:.+]] = arith.constant 1 : index
  // CLONE: %[[STRIDE0:.+]] = arith.muli %[[C1]], %[[SIZES]]#1
  // CLONE: %[[COUNT:.+]] = arith.muli %[[STRIDE0]], %[[SIZES]]#0
  // CLONE: %[[C8:.+]] = arith.constant 8 : index
  // CLONE: %[[BYTES:.+]] = arith.muli %[[COUNT]], %[[C8]]
  // CLONE: %[[HEADER:.+]] = arith.constant 48 : index
  // CLONE: %[[BOX_BYTES:.+]] = arith.addi %[[BYTES]], %[[HEADER]]
  // CLONE: %[[TOKEN:.+]] = reussir.token.alloc(%[[BOX_BYTES]] : index) : <align : 8, size : ?>
  // CLONE: %[[CLONED:.+]] = reussir.rc.create value(%{{.+}} : !reussir.array<? x 2 x !reussir.rc<i64>>) token(%[[TOKEN]] : !reussir.token<align : 8, size : ?>) extents(%[[SIZES]]#0)
  // CLONE: %[[DST:.+]] = reussir.rc.borrow(%[[CLONED]]
  // CLONE: %[[C0:.+]] = arith.constant 0 : index
  // CLONE: %[[OFF_OK:.+]] = arith.cmpi eq, %[[OFFSET]], %[[C0]]
  // CLONE: %[[S0_OK:.+]] = arith.cmpi eq, %[[STRIDES]]#0, %[[STRIDE0]]
  // CLONE: %[[AND0:.+]] = arith.andi %[[OFF_OK]], %[[S0_OK]]
  // CLONE: %[[S1_OK:.+]] = arith.cmpi eq, %[[STRIDES]]#1, %[[C1]]
  // CLONE: %[[CANONICAL:.+]] = arith.andi %[[AND0]], %[[S1_OK]]
  // CLONE: scf.if %[[CANONICAL]] {
  // CLONE: reussir.ref.memcpy %[[SRC]] to %[[DST]] size(%[[BYTES]] : index)
  // CLONE: } else {
  // CLONE: %[[DST_VIEW:.+]] = reussir.array.view(%[[DST]]
  // CLONE: scf.for %[[I:.+]] = %{{.+}} to %[[SIZES]]#0
  // CLONE: scf.for %[[J:.+]] = %{{.+}} to %[[SIZES]]#1
  // CLONE: %[[SRC_ROW:.+]] = reussir.array.project(%[[SRC_VIEW]] : {{.*}}) [%[[I]] : index]
  // CLONE: %[[SRC_ELT:.+]] = reussir.array.project(%[[SRC_ROW]] : {{.*}}) [%[[J]] : index]
  // CLONE: %[[DST_ROW:.+]] = reussir.array.project(%[[DST_VIEW]] : {{.*}}) [%[[I]] : index]
  // CLONE: %[[DST_ELT:.+]] = reussir.array.project(%[[DST_ROW]] : {{.*}}) [%[[J]] : index]
  // CLONE: reussir.ref.memcpy %[[SRC_ELT]] to %[[DST_ELT]] : <!reussir.rc<i64>> to <!reussir.rc<i64>>
  // CLONE: }
  // CLONE: reussir.ref.acquire(%[[DST]]
  // CLONE: reussir.rc.fetch(%arg0
  // CLONE: reussir.rc.set(%arg0
  // CLONE: scf.yield %[[CLONED]]
  func.func @clone_dynamic(%xs: !rc_dv) -> !rc_dv {
    %res = reussir.array.with_unique_view (%xs : !rc_dv) -> !rc_dv {
      ^bb0(%view: !view):
        reussir.scf.yield
    }
    return %res : !rc_dv
  }

  // A strided projection is descriptor arithmetic: the leading extent for
  // the bounds assumption is the descriptor's size, the row shift lands in
  // the live offset field (`offset + i * stride[0]`), the aligned pointer is
  // untouched, and the trailing size/stride pass through.
  // The bare-argument descriptor: `%arg1` aligned pointer, `%arg2` offset,
  // `%arg3`/`%arg4` sizes, `%arg5`/`%arg6` strides, `%arg7` the index.
  // LLVM-LABEL: llvm.func @project_dynamic(
  // LLVM: %[[IN_BOUNDS:.+]] = llvm.icmp "ult" %arg7, %arg3
  // LLVM: llvm.intr.assume %[[IN_BOUNDS]]
  // LLVM: %[[DELTA:.+]] = llvm.mul %arg7, %arg5
  // LLVM: %[[SHIFT:.+]] = llvm.add %arg2, %[[DELTA]]
  // LLVM: llvm.insertvalue %arg1, %{{.+}}[1]
  // LLVM: llvm.insertvalue %[[SHIFT]], %{{.+}}[2]
  // LLVM: llvm.insertvalue %arg4, %{{.+}}[3, 0]
  // LLVM: llvm.insertvalue %arg6, %{{.+}}[4, 0]
  // LLVM-NOT: llvm.getelementptr
  // LLVM: llvm.return
  func.func @project_dynamic(%view: !view, %i: index) -> !row {
    %row = reussir.array.project(%view : !view) [%i : index] : !row
    return %row : !row
  }
}
