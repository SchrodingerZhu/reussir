//! The collected trait and impl declarations.
//!
//! These are the program's trait items after name resolution. Method bodies are
//! deliberately absent in Phase 0 — only the shapes the resolver needs.

use reussir_syntax::kind::TokenKey;
use reussir_syntax::source::FileId;

use crate::semi::ty::{DefId, GenericId, Ty};
use crate::surface::{self, Span};

use super::{ImplId, Obligation, TraitId, TraitRef};

/// How a trait method takes its receiver. Tracked out of band because the
/// flex coloring is dropped on the generic `Self`, and `Arc<Self>` peels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiverForm {
    /// `self: Self`.
    Value,
    /// `self: Arc<Self>`.
    Arc,
    /// `self: [flex] Self` (implies `regional fn`).
    Flex,
}

/// A method declared by a trait. Bodies live on the impl side (Phase 1+).
/// Parameters are positional (`params[0]` is the receiver): conformance is
/// positional and no bodies exist, so names would serialize dead weight.
#[derive(Clone, Debug)]
pub struct MethodSig<'tcx> {
    pub name: TokenKey,
    /// The method's own generics, following the trait's in the binder.
    pub generics: Vec<(TokenKey, GenericId)>,
    pub receiver: ReceiverForm,
    pub params: Vec<Ty<'tcx>>,
    pub ret: Ty<'tcx>,
    pub is_regional: bool,
    pub span: Option<Span>,
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
    /// The trait's path-keyed identity in the [`DefTable`] — the single
    /// source of its name and module for display, resolution, and
    /// serialization.
    ///
    /// [`DefTable`]: crate::semi::resolve::DefTable
    pub def: DefId,
    pub visibility: surface::Visibility,
    /// A sealed trait cannot be implemented by user code: the builtin
    /// value-class traits (whose operations lower by ground type) and the
    /// structural `Sync`.
    pub sealed: bool,
    /// The implicit `Self` parameter. `GenericId(0)`-unallocated for the
    /// builtins (their defs predate the elaborator's generics table);
    /// `fresh_generic`-allocated for user traits.
    pub self_param: GenericId,
    /// The trait's declared (non-`Self`) type parameters.
    pub params: Vec<(TokenKey, GenericId)>,
    /// Super-traits (`trait Sub: Super`). These subsume the old hard-coded class
    /// DAG: `Integral: Num` is just a super-trait edge.
    pub supertraits: Vec<TraitRef<'tcx>>,
    pub methods: Vec<MethodSig<'tcx>>,
    pub assoc_tys: Vec<AssocTyDef>,
    pub span: Option<Span>,
    pub file: FileId,
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
    /// Obligations the impl assumes (`where …`; populated from the block
    /// generics' inline bounds this cut).
    pub where_clauses: Vec<Obligation<'tcx>>,
    /// The declared method bodies, as function defs in *trait-method order*
    /// — the future `TraitCall { method index }` targets. Empty for the
    /// builtin (method-less) impls.
    pub methods: Vec<DefId>,
    pub span: Option<Span>,
    pub file: FileId,
}
