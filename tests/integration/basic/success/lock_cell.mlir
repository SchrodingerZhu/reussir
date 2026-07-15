// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

// Lock-guarded cells (mutex / flatlock / rwlock) wrap an arbitrary payload in a
// `sync` synchronization primitive and, like atomic cells, are always managed
// by an atomic shared RC box. This test pins down the type syntax roundtrip.

!mutex_cell = !reussir.rc<!reussir.cell<i64 mutex> atomic>
!flatlock_cell = !reussir.rc<!reussir.cell<!reussir.rc<i32> flatlock> atomic>
!rwlock_cell = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

module {
  // CHECK-LABEL: func.func @lock_cell_types
  // CHECK-SAME: !reussir.rc<!reussir.cell<i64 mutex> atomic>
  // CHECK-SAME: !reussir.rc<!reussir.cell<!reussir.rc<i32> flatlock> atomic>
  // CHECK-SAME: !reussir.rc<!reussir.cell<i64 rwlock> atomic>
  func.func @lock_cell_types(%m: !mutex_cell, %f: !flatlock_cell,
                             %w: !rwlock_cell) {
    return
  }
}
