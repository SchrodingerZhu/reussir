// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// A read transaction shares the reader-writer lock; every other lock kind
// only has an exclusive critical section, so rdlock rejects them.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @rdlock_mutex(%cell: !mutex_cell) -> i64 {
  // expected-error @+1 {{read transaction requires an rwlock cell, got a cell of kind 'mutex'}}
  %v = reussir.cell.rdlock(%cell : !mutex_cell) -> i64 {
    ^bb0(%current: i64):
      reussir.scf.yield %current : i64
  }
  return %v : i64
}

// -----

// Plain cells have no lock at all.
!plain_cell = !reussir.rc<!reussir.cell<i64> shared>

func.func @rdlock_plain(%cell: !plain_cell) -> i64 {
  // expected-error @+1 {{read transaction requires an rwlock cell, got a cell of kind 'plain'}}
  %v = reussir.cell.rdlock(%cell : !plain_cell) -> i64 {
    ^bb0(%current: i64):
      reussir.scf.yield %current : i64
  }
  return %v : i64
}

// -----

// The body receives the cloned element, so its argument must match the cell
// element type.
!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

func.func @rdlock_bad_argument(%cell: !rwlock_cell) -> i64 {
  // expected-error @+1 {{body argument type must match cell element type, expected 'i64', got 'i32'}}
  %v = reussir.cell.rdlock(%cell : !rwlock_cell) -> i64 {
    ^bb0(%current: i32):
      %wide = arith.extsi %current : i32 to i64
      reussir.scf.yield %wide : i64
  }
  return %v : i64
}

// -----

// A transaction with a result must yield it.
!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

func.func @rdlock_missing_yield_value(%cell: !rwlock_cell) -> i64 {
  // expected-error @+1 {{body must yield exactly one optional output when the operation has a result}}
  %v = reussir.cell.rdlock(%cell : !rwlock_cell) -> i64 {
    ^bb0(%current: i64):
      reussir.scf.yield
  }
  return %v : i64
}

// -----

// The yielded value's type is the operation result type.
!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

func.func @rdlock_yield_type_mismatch(%cell: !rwlock_cell) -> i32 {
  // expected-error @+1 {{body output type must match operation result type, got 'i64' and 'i32'}}
  %v = reussir.cell.rdlock(%cell : !rwlock_cell) -> i32 {
    ^bb0(%current: i64):
      reussir.scf.yield %current : i64
  }
  return %v : i32
}
