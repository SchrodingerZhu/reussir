// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// A mutex cell, like an atomic cell, only exists to be shared across threads,
// so the box managing it must be a shared RC pointer with an atomic refcount.
// expected-error @+1 {{a cell of kind 'mutex' must be managed by an atomic shared RC pointer, got capability shared and atomic kind normal}}
!bad_mutex_box = !reussir.rc<!reussir.cell<i64 mutex>>

module {
}

// -----

// A flatlock (flat-combining lock) cell carries the same requirement.
// expected-error @+1 {{a cell of kind 'flatlock' must be managed by an atomic shared RC pointer, got capability shared and atomic kind normal}}
!bad_flatlock_box = !reussir.rc<!reussir.cell<i64 flatlock>>

module {
}

// -----

// The shared-capability half of the invariant holds for rwlock cells too:
// even with an atomic refcount, a regional (flex/rigid) box is rejected.
// expected-error @+1 {{a cell of kind 'rwlock' must be managed by an atomic shared RC pointer, got capability flex and atomic kind atomic}}
!bad_rwlock_box = !reussir.rc<!reussir.cell<i64 rwlock> flex atomic>

module {
}
