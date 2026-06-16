//! The collected trait and impl declarations.
//!
//! These are the program's trait items after name resolution. Method bodies are
//! deliberately absent in Phase 0 — only the shapes the resolver needs.

use crate::ty::{GenericId, Ty};

use super::{ImplId, Obligation, TraitId, TraitRef};

/// A method declared by a trait. Bodies live on the impl side (Phase 1+).
#[derive(Clone, Debug)]
pub struct MethodSig<'tcx> {
    pub name: String,
    pub params: Vec<Ty<'tcx>>,
    pub ret: Ty<'tcx>,
}

/// An associated type declaration. Reserved so the IR can grow into projections
/// without churn; unused in Phase 0.
#[derive(Clone, Debug)]
pub struct AssocTyDef {
    pub name: String,
}

/// A trait declaration.
#[derive(Clone, Debug)]
pub struct TraitDef<'tcx> {
    pub id: TraitId,
    pub name: String,
    /// Type parameters; `params[0]` is the implicit `Self`.
    pub params: Vec<GenericId>,
    /// Super-traits (`trait Sub: Super`). These subsume the old hard-coded class
    /// DAG: `Integral: Num` is just a super-trait edge.
    pub supertraits: Vec<TraitRef<'tcx>>,
    pub methods: Vec<MethodSig<'tcx>>,
    pub assoc_tys: Vec<AssocTyDef>,
}

/// An impl declaration.
#[derive(Clone, Debug)]
pub struct ImplDef<'tcx> {
    pub id: ImplId,
    /// The impl's own generics (`impl<…>`).
    pub generics: Vec<GenericId>,
    /// The trait being implemented; `trait_ref.args[0]` is `self_ty`.
    pub trait_ref: TraitRef<'tcx>,
    pub self_ty: Ty<'tcx>,
    /// Obligations the impl assumes (`where …`).
    pub where_clauses: Vec<Obligation<'tcx>>,
}
