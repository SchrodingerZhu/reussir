//! The fulfillment context: trait obligations that are collected during
//! checking and discharged once enough holes are solved.
//!
//! This replaces the inherited "bounded hole" mechanism. A numeric literal or a
//! generic instantiation registers an [`Obligation`]; after a function body is
//! checked, [`Elaborator::resolve_obligations`] runs a fixpoint that discharges
//! each obligation against the (now solved) self type — via impl search for
//! concrete types and super-trait subsumption for in-scope generics.

use crate::semi::traits::{Obligation, TraitRef};
use crate::semi::ty::TyKind;
use crate::surface::Span;

use super::ctxt::Elaborator;

/// An obligation plus the span to blame if it fails.
#[derive(Clone, Debug)]
pub struct PendingObligation<'tcx> {
    pub obligation: Obligation<'tcx>,
    pub span: Option<Span>,
}

#[derive(Default)]
pub struct FulfillCtxt<'tcx> {
    pending: Vec<PendingObligation<'tcx>>,
}

impl<'tcx> FulfillCtxt<'tcx> {
    pub fn register(&mut self, obligation: Obligation<'tcx>, span: Option<Span>) {
        self.pending.push(PendingObligation { obligation, span });
    }
}

enum Discharge {
    Solved,
    /// Self type is still a hole; try again after more solving.
    Defer,
    Failed(String),
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Discharge all pending obligations to a fixpoint, reporting any that fail
    /// or remain ambiguous.
    pub fn resolve_obligations(&mut self) {
        let mut pending = std::mem::take(&mut self.fulfill).into_pending();
        loop {
            let mut progress = false;
            let mut deferred = Vec::new();
            for p in std::mem::take(&mut pending) {
                match self.try_discharge(&p.obligation) {
                    Discharge::Solved => progress = true,
                    Discharge::Failed(msg) => {
                        self.error(p.span, msg);
                        progress = true;
                    }
                    Discharge::Defer => deferred.push(p),
                }
            }
            pending = deferred;
            if !progress || pending.is_empty() {
                break;
            }
        }
        for p in pending {
            self.error(p.span, "type annotations needed: cannot resolve a bound");
        }
    }

    fn try_discharge(&mut self, obligation: &Obligation<'tcx>) -> Discharge {
        // Only trait obligations exist today; capability-as-a-bound was removed
        // until its (non-total-order) semantics are settled.
        let Obligation::Trait(tref) = obligation;
        self.discharge_trait(tref)
    }

    fn discharge_trait(&mut self, tref: &TraitRef<'tcx>) -> Discharge {
        let self_ty = self.infer.shallow_resolve(tref.self_ty());
        match self_ty.kind() {
            TyKind::Hole(_) => Discharge::Defer,
            // In-scope generic: satisfied iff one of its declared bounds implies
            // the required trait (super-trait subsumption).
            TyKind::Generic(g) => {
                let want = tref.trait_id;
                if self
                    .generic_bounds(*g)
                    .iter()
                    .any(|&have| self.traits.implies(have, want))
                {
                    Discharge::Solved
                } else {
                    let name = &self.traits.trait_def(want).name;
                    Discharge::Failed(format!(
                        "the bound `{name}` is not satisfied by this generic"
                    ))
                }
            }
            _ => {
                let goal = Obligation::Trait(TraitRef {
                    trait_id: tref.trait_id,
                    args: vec![self_ty],
                });
                match self.traits.select(&goal) {
                    Ok(_) => Discharge::Solved,
                    Err(_) => {
                        let name = &self.traits.trait_def(tref.trait_id).name;
                        Discharge::Failed(format!("`{self_ty:?}` does not implement `{name}`"))
                    }
                }
            }
        }
    }
}

impl<'tcx> FulfillCtxt<'tcx> {
    fn into_pending(self) -> Vec<PendingObligation<'tcx>> {
        self.pending
    }
}
