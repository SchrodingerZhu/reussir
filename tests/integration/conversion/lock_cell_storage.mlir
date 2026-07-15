// RUN: %reussir-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

module {
  // A lock-guarded cell is physically stored as its `sync` primitive: the cell
  // type converts to the sync rwlock layout — an rwlock header (two i32 words)
  // followed by the i64 payload — so the RC box lowers to
  // `{ refcount, { { i32, i32 }, i64 } }` and a borrow GEPs into that shape.
  // The `reussir.ref.as_memref` bridge then views the storage as a
  // `memref<!sync.rwlock<i64>>` for the sync operations.
  // ROUNDTRIP-LABEL: func.func @lock_storage
  // ROUNDTRIP: reussir.ref.as_memref(%{{.*}}) : memref<!sync.rwlock<i64>>

  // LLVM-LABEL: define {{.*}}@lock_storage
  // LLVM: getelementptr { i32, { { i32, i32 }, i64 } }
  func.func @lock_storage(%cell: !rwlock_cell)
      -> memref<!sync.rwlock<i64>> {
    %ref = reussir.rc.borrow(%cell : !rwlock_cell)
      : !reussir.ref<!reussir.cell<i64 rwlock> atomic>
    %m = reussir.ref.as_memref(
        %ref : !reussir.ref<!reussir.cell<i64 rwlock> atomic>)
      : memref<!sync.rwlock<i64>>
    return %m : memref<!sync.rwlock<i64>>
  }
}
