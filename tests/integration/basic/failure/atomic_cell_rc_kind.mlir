// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// An atomic cell only exists to be shared across threads, so the box managing
// it must be a shared RC pointer with an atomic refcount.
// expected-error @+1 {{a cell of kind 'atomic' must be managed by an atomic shared RC pointer, got capability shared and atomic kind normal}}
!bad_nonatomic_box = !reussir.rc<!reussir.cell<i64 atomic>>

module {
}

// -----

// The shared-capability half of the invariant: even with an atomic refcount,
// a regional (flex/rigid) box cannot manage an atomic cell.
// expected-error @+1 {{a cell of kind 'atomic' must be managed by an atomic shared RC pointer, got capability flex and atomic kind atomic}}
!bad_flex_box = !reussir.rc<!reussir.cell<i64 atomic> flex atomic>

module {
}
