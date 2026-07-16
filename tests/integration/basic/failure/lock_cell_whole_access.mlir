// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// A lock-guarded cell keeps its payload inside a `sync` primitive, not at the
// leading slot, so an operation needs a dedicated lock-aware lowering. Mutex
// create/get/set/rmw are supported; the other lock operations remain
// unavailable.
!flatlock_cell = !reussir.rc<!reussir.cell<i64 flatlock> atomic>

func.func @get_flatlock(%cell: !flatlock_cell) -> i64 {
  // expected-error @+1 {{whole-element access is not supported on a cell of kind 'flatlock'; access it through a critical-section region}}
  %v = reussir.cell.get(%cell : !flatlock_cell) : i64
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

// -----

// The mutex exemption covers exactly create/get/set/rmw: a mutex cell has no
// in-use flag to observe, so `cell.in_use` stays rejected.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @in_use_mutex(%cell: !mutex_cell) -> i1 {
  // expected-error @+1 {{whole-element access is not supported on a cell of kind 'mutex'; access it through a critical-section region}}
  %flag = reussir.cell.in_use(%cell : !mutex_cell) : i1
  return %flag : i1
}

// -----

// The direct atomic RMW form is atomic-only: a mutex read-modify-write is the
// region form running inside the critical section.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @direct_rmw_mutex(%delta: i64, %cell: !mutex_cell) -> i64 {
  // expected-error @+1 {{direct atomic RMW form requires an atomic cell, got a mutex cell}}
  %old = reussir.cell.rmw addi(%delta : i64, %cell : !mutex_cell) -> i64
  return %old : i64
}

// -----

// An ordering annotation is meaningless under a lock: it stays atomic-only.
!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>

func.func @ordered_get_mutex(%cell: !mutex_cell) -> i64 {
  // expected-error @+1 {{atomic ordering is only valid for an atomic cell}}
  %v = reussir.cell.get(%cell : !mutex_cell) ordering(acquire) : i64
  return %v : i64
}
