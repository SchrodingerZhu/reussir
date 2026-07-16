// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

// A lock-guarded cell's payload is viewed through a zero-ranked memref inside
// the critical-section region, so the element must be a valid memref element
// type. A nullable is not one: without this restriction the lowering would
// construct an invalid `memref<!reussir.nullable<...>>`.
// expected-error @+1 {{a cell of kind 'mutex' requires an element that is a valid memref element type}}
!bad_mutex_nullable = !reussir.cell<!reussir.nullable<!reussir.rc<i64>> mutex>

module {
}

// -----

// The same restriction applies to every lock kind, records included.
// expected-error @+1 {{a cell of kind 'rwlock' requires an element that is a valid memref element type}}
!bad_rwlock_record = !reussir.cell<!reussir.record<compound "pair" {i64, i64}> rwlock>

module {
}
