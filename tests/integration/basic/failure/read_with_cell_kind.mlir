// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// read_with holds a *read* lock, so it only applies to a reader-writer-lock
// cell; a mutex cell has no shared-read mode.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @read_mutex(%cell: !mutex_cell) {
  // expected-error @+1 {{read_with requires a reader-writer-lock cell, got a cell of kind 'mutex'}}
  reussir.cell.read_with(%cell : !mutex_cell) {
    ^bb0(%view: memref<i64>):
      reussir.scf.yield
  }
  return
}

// -----

// The body observes the payload through a zero-ranked memref view whose element
// type must match the cell element.
!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

func.func @read_wrong_view(%cell: !rwlock_cell) {
  // expected-error @+1 {{body argument must be a zero-ranked memref view of the cell element, expected 'memref<i64>', got 'memref<i32>'}}
  reussir.cell.read_with(%cell : !rwlock_cell) {
    ^bb0(%view: memref<i32>):
      reussir.scf.yield
  }
  return
}
