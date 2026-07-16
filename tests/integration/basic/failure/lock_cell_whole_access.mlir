// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// A lock-guarded cell keeps its payload inside a `sync` primitive, not at the
// leading slot, so lock-free whole-element access is unsound: it must be
// reached through a critical-section region instead. Mutex creation is
// supported separately; the other lock constructors remain unavailable.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @get_mutex(%cell: !mutex_cell) -> i64 {
  // expected-error @+1 {{whole-element access is not supported on a cell of kind 'mutex'; access it through a critical-section region}}
  %v = reussir.cell.get(%cell : !mutex_cell) : i64
  return %v : i64
}

// -----

!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

func.func @set_rwlock(%cell: !rwlock_cell, %value: i64) {
  // expected-error @+1 {{whole-element access is not supported on a cell of kind 'rwlock'; access it through a critical-section region}}
  reussir.cell.set(%value : i64, %cell : !rwlock_cell)
  return
}

// -----

!flatlock_cell = !reussir.rc<!reussir.cell<i64 flatlock> atomic>

func.func @create_flatlock(%value: i64) -> !flatlock_cell {
  // expected-error @+1 {{whole-element access is not supported on a cell of kind 'flatlock'; access it through a critical-section region}}
  %cell = reussir.cell.create value(%value : i64) : !flatlock_cell
  return %cell : !flatlock_cell
}
