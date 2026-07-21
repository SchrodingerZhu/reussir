// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// Nontrivial import: the aggregate argument is packed into a struct passed
// by pointer and the aggregate result comes back through a leading return
// pointer — the exact mirror of the export boundary, independent of the
// target triple.
module attributes {llvm.target_triple = "x86_64-pc-linux-gnu"} {
  llvm.func @id_huge(!llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64)>)
      -> !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64)>

  reussir.trampoline import "C" @id_huge_c = @id_huge
}

// CHECK-DAG: llvm.func @id_huge_c(!llvm.ptr, !llvm.ptr)
// CHECK-LABEL: llvm.func @id_huge(
// CHECK-SAME: %[[A:.*]]: !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64)>
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[RET:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64)>
// CHECK: %[[ARGS:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(struct<(i64, i64, i64, i64, i64, i64, i64, i64)>)>
// CHECK: %[[F0:.*]] = llvm.getelementptr %[[ARGS]][0, 0]
// CHECK: llvm.store %[[A]], %[[F0]]
// CHECK: llvm.call @id_huge_c(%[[RET]], %[[ARGS]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK: %[[R:.*]] = llvm.load %[[RET]] : !llvm.ptr -> !llvm.struct<(i64, i64, i64, i64, i64, i64, i64, i64)>
// CHECK: llvm.return %[[R]]
