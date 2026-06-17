//! The fulfillment context: trait obligations that are collected during
//! checking and discharged once enough holes are solved.
//!
//! *Discharge* is the proof-theory term we borrow: to **discharge** an
//! obligation is to remove it from the pending set by *deciding* it — either
//! proving it holds (finding the impl, or the super-trait subsumption, that
//! satisfies it) or rejecting it as unsatisfiable. An obligation that cannot yet
//! be decided — its self type is still an inference hole — is neither proved nor
//! rejected but **deferred**, and retried once more of the substitution is
//! known. (The dual of `register`: checking *registers* obligations, the
//! fulfillment pass *discharges* them.)
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
    ///
    /// Operational semantics. A configuration is ⟨P, E⟩ — P the pending
    /// obligation set, E the accumulated diagnostics. Each obligation decides to
    /// ✓ (solved), ✗ (failed) or ? (deferred) per the rules on `discharge_trait`.
    /// One round drops the decided obligations and blames the failures:
    ///
    /// ```text
    ///   P? = { o ∈ P | o ⇝ ? }       P✗ = { o ∈ P | o ⇝ ✗ }
    /// ──────────────────────────────────────────────────────── (round)
    ///            ⟨P, E⟩ ⟶ ⟨P?, E ∪ errors(P✗)⟩
    /// ```
    ///
    /// The loop applies (round) while it makes progress (P? ⊊ P) and P? ≠ ∅, then
    /// reports every obligation still pending at the fixpoint as ambiguous:
    ///
    /// ```text
    ///   ⟨P, E⟩ is a fixpoint (no progress)      o ∈ P
    /// ───────────────────────────────────────────────────── (ambiguous)
    ///       E ∪= ("type annotations needed" @ span(o))
    /// ```
    ///
    /// Termination: a productive (round) strictly shrinks P and an unproductive
    /// one ends the loop, so it runs at most |P| rounds. In fact σ (the inference
    /// substitution) is *read-only* during discharge — it reads the table but
    /// never solves a hole — so the deferred set is already stable after the
    /// first round. The loop is effectively single-pass today; the iteration is
    /// scaffolding for a future discharge that can itself solve holes (e.g. a
    /// sole-candidate impl driving its self type).
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

    /// Decide one obligation `o = (trait(o) : self(o))` against the current
    /// state.
    ///
    /// Operational semantics. With ⟦·⟧σ = `shallow_resolve`, Γ(g) the declared
    /// bounds of generic `g`, `D ⊢ b ⊵ t` for `TraitDb::implies` (t lies in b's
    /// super-trait closure), and select_D for impl search, the outcome
    /// ρ ∈ {✓, ✗, ?} is:
    ///
    /// ```text
    ///   ⟦self(o)⟧σ = α               (an inference hole)
    /// ──────────────────────────────────────────────────── (defer)
    ///                  o ⇝ ?
    ///
    ///   ⟦self(o)⟧σ = g    ∃ b ∈ Γ(g). D ⊢ b ⊵ trait(o)
    /// ──────────────────────────────────────────────────── (param)
    ///                  o ⇝ ✓
    ///
    ///   ⟦self(o)⟧σ = g    ∀ b ∈ Γ(g). ¬(D ⊢ b ⊵ trait(o))
    /// ──────────────────────────────────────────────────── (param-fail)
    ///                  o ⇝ ✗
    ///
    ///   ⟦self(o)⟧σ = τ (concrete)    select_D(trait(o), τ) = Ok
    /// ────────────────────────────────────────────────────────── (impl)
    ///                  o ⇝ ✓
    ///
    ///   ⟦self(o)⟧σ = τ (concrete)    select_D(trait(o), τ) = Err
    /// ────────────────────────────────────────────────────────── (no-impl)
    ///                  o ⇝ ✗
    /// ```
    ///
    /// Caveat: (defer) fires only on a *head* hole. A self type with holes nested
    /// under a constructor (`List<?h>`) is read as concrete, so (no-impl) can
    /// fire spuriously instead of deferring — latent until parameterized impls
    /// exist; the fix is to defer when the *deeply* resolved self type still
    /// holds any hole.
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
