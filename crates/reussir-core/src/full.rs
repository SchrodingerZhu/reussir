//! The *Full* phase: the monomorphized, mangled representation.
//!
//! Where [`crate::semi`] keeps generics open and may carry inference holes, the
//! Full phase is **ground**: every type argument is a concrete [`semi::ty::Ty`]
//! with no [`semi::ty::TyKind::Generic`] or [`semi::ty::TyKind::Hole`] left. That
//! invariant is what lets each item instance have a single, stable linker symbol
//! (see [`mangle`]) and what the backend lowering relies on.
//!
//! Full reuses the Semi type interner ([`semi::ty::TyCtxt`]) rather than
//! introducing a parallel type representation: a Full type *is* a Semi `Ty` that
//! happens to be ground, so monomorphization is a substitution that re-interns
//! into the same arena and downstream code keeps pointer-identity equality.
//!
//! [`semi`]: crate::semi

pub mod mangle;
