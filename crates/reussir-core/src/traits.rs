//! Trait references, the obligations the solver discharges, and the evidence it
//! produces.
//!
//! Resolution is nominal and coherent: an [`Obligation::Trait`] is discharged by
//! finding the unique impl (Phase 0 handles only ground self types), and an
//! [`Obligation::Cap`] by the capability lattice. The [`Evidence`] returned is a
//! proof tree consumed statically at monomorphization; its shape is also what a
//! future `dyn` vtable would carry.

pub mod builtins;
pub mod db;
pub mod def;

pub use db::{SelectError, TraitDb};
pub use def::{AssocTyDef, ImplDef, MethodSig, TraitDef};

use crate::ty::Ty;

/// Identifies a trait declaration.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TraitId(pub u32);

/// Identifies an impl.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImplId(pub u32);

/// `Trait<args>`. By convention `args[0]` is the `Self` type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TraitRef<'tcx> {
    pub trait_id: TraitId,
    pub args: Vec<Ty<'tcx>>,
}

impl<'tcx> TraitRef<'tcx> {
    /// The `Self` type this reference is about. By convention `args[0]` always
    /// exists for a well-formed reference.
    pub fn self_ty(&self) -> Ty<'tcx> {
        debug_assert!(
            !self.args.is_empty(),
            "a TraitRef must have a Self argument"
        );
        self.args[0]
    }
}

/// A predicate the solver must discharge. (Capability-as-a-bound will join this
/// once its semantics are settled — see [`crate::ty::Capability`].)
#[derive(Clone, Debug)]
pub enum Obligation<'tcx> {
    /// `τ : Trait<…>`, discharged by impl search.
    Trait(TraitRef<'tcx>),
}

/// A proof that an obligation holds.
#[derive(Clone, Debug)]
pub enum Evidence<'tcx> {
    /// A concrete impl was chosen; `sub` proves the impl's own where-clauses.
    Impl {
        impl_id: ImplId,
        args: Vec<Ty<'tcx>>,
        sub: Vec<Evidence<'tcx>>,
    },
    /// Discharged by an in-scope assumption (a `T: Trait` bound on the enclosing
    /// definition). Phase 1.
    Param(usize),
    /// Projected from a super-trait of other evidence; `index` is the position
    /// in the sub-trait's super-trait list.
    Super {
        of: Box<Evidence<'tcx>>,
        index: usize,
    },
}
