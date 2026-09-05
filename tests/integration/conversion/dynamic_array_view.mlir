// RUN: %reussir-opt %s \
// RUN:   --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),control-flow-sink,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,cse,canonicalize)' \
// RUN:   | %FileCheck %s

// A dynamic-extent array (docs/design/dynamic-extent-arrays.md) boxes as
// `{ i32 count | index offset | index size[r] | index stride[r] | tail }`;
// `array.view` is a straight header copy — recover the box from the payload
// ref by the static header offset, load the descriptor fields, address as
// `payload + offset + i * stride`.

!dv = !reussir.array<? x i32>
!rc_dv = !reussir.rc<!dv>

module {
  // CHECK-LABEL: llvm.func @read
  // CHECK: %[[PAYLOAD:.+]] = llvm.getelementptr %arg0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64, i64, i64, array<0 x i32>)>
  // CHECK: %[[BOX:.+]] = llvm.getelementptr %[[PAYLOAD]][-32] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[OFFP:.+]] = llvm.getelementptr %[[BOX]][0, 1]
  // CHECK: %[[OFF:.+]] = llvm.load %[[OFFP]] : !llvm.ptr -> i64
  // CHECK: %[[STRP:.+]] = llvm.getelementptr %[[BOX]][0, 3]
  // CHECK: %[[STR:.+]] = llvm.load %[[STRP]] : !llvm.ptr -> i64
  // CHECK: llvm.getelementptr %{{.+}}[%[[OFF]]]
  // CHECK: llvm.mul %arg1, %[[STR]]
  // CHECK: llvm.load %{{.+}} : !llvm.ptr -> i32
  func.func @read(%a: !rc_dv, %i: index) -> i32 {
    %ref = reussir.rc.borrow (%a : !rc_dv) : !reussir.ref<!dv>
    %v = reussir.array.view(%ref : !reussir.ref<!dv>) : memref<?xi32, strided<[?], offset: ?>>
    %x = memref.load %v[%i] : memref<?xi32, strided<[?], offset: ?>>
    return %x : i32
  }

  // Construction: the instantiated token computes `header + n * elemsize`,
  // allocation takes the generic entry point with the runtime size, and the
  // header stores are canonical (offset 0, size n, stride 1).
  // CHECK-LABEL: llvm.func @make
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : index) : i64
  // CHECK-DAG: %[[C32:.+]] = llvm.mlir.constant(32 : index) : i64
  // CHECK: %[[BYTES0:.+]] = llvm.mul %arg0, %[[C4]]
  // CHECK: %[[BYTES:.+]] = llvm.add %[[BYTES0]], %[[C32]]
  // CHECK: %[[TOK:.+]] = llvm.call @__reussir_allocate(%{{.+}}, %[[BYTES]])
  // CHECK-DAG: llvm.getelementptr %[[TOK]][0, 1]
  // CHECK-DAG: llvm.getelementptr %[[TOK]][0, 2]
  // CHECK-DAG: llvm.getelementptr %[[TOK]][0, 3]
  // CHECK-DAG: llvm.getelementptr %[[TOK]][0, 0]
  func.func @make(%n: index) -> !rc_dv {
    %poison = ub.poison : !dv
    %rc = reussir.rc.create value(%poison : !dv) extents(%n) : !rc_dv
    return %rc : !rc_dv
  }
}
