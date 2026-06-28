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
    /// Operational semantics (substructural). Read the pending obligations as a
    /// *linear* context Δ — a multiset of resources, each to be discharged
    /// *exactly once*. The machine state is ⟨Δ | σ | E⟩ with σ the inference
    /// substitution and E the diagnostics. A decidable obligation is *consumed*
    /// (`o ⇝ ρ` is the per-obligation judgment on `discharge_trait`; `Δ, o` is
    /// multiset extension; `m @ span(o)` blames message `m` at `o`'s span). A
    /// deferred obligation matches no rule and stays in Δ:
    ///
    /// ```text
    ///         o ⇝ ✓                            o ⇝ ✗(m)
    /// ────────────────────────────    ────────────────────────────────────
    ///   ⟨Δ, o | σ | E⟩ ⟶ ⟨Δ | σ | E⟩    ⟨Δ, o | σ | E⟩ ⟶ ⟨Δ | σ | E, m@span(o)⟩
    /// ```
    ///
    /// Reduce to a normal form ⟨Δ | σ | E⟩ in which every remaining `o` is
    /// deferred. Those are *leaked* resources — obligations that could not be
    /// consumed — and each is reported:
    ///
    /// ```text
    ///   ⟨Δ | σ | E⟩ normal      o ∈ Δ
    /// ────────────────────────────────────── (leak)
    ///   E ∪= ("type annotations needed" @ span(o))
    /// ```
    ///
    /// Linearity is the content: an obligation is discharged exactly once — no
    /// contraction (a bound is not proved twice), no weakening (it cannot be
    /// silently dropped) — so a leftover resource is *precisely* an ambiguity.
    /// Termination is immediate: every ⟶ removes one obligation from Δ.
    ///
    /// Today no step advances σ — discharge only reads the table — so a deferred
    /// obligation is permanently stuck and the normal form is one sweep away.
    /// That holds only because every obligation is single-argument (`Self`). With
    /// type-carrying traits — multi-parameter (`T: Add<Rhs>`) or an associated-
    /// type output — the (✓) step would also step σ ⟶ σ′ (unifying the trait's
    /// non-`Self` arguments, which may be holes), which can make a previously
    /// stuck obligation reducible: the cascade, where discharge and unification
    /// interleave (rustc's model). The code realizes this rewrite with a batched
    /// schedule (`std::mem::take` the bag, sweep, keep the deferred) — one fair
    /// ordering of the steps above.
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
    ///
    /// The (impl) rule rebuilds the goal with only the resolved `self` argument;
    /// a trait's other arguments — a multi-parameter `Rhs`, an associated-type
    /// output — are dropped here. Supporting them means matching *all* arguments
    /// in `select` and unifying the chosen impl's against the obligation's, which
    /// is where discharge starts solving holes (see `resolve_obligations`).
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
                // Deeply resolve before deciding: a self type whose *head* is a
                // constructor may still hold holes under it (`List<?h>`). Reading
                // such a type as ground would let (no-impl) fire spuriously
                // instead of deferring. Defer while any hole remains so the
                // obligation is retried once more of the substitution is known.
                let self_ty = self.infer.resolve(self_ty);
                if ty_has_hole(self_ty) {
                    return Discharge::Defer;
                }
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

/// Does `ty` contain an unsolved inference hole anywhere in its structure?
/// `ty` is expected to be deeply resolved (`InferCtxt::resolve`), so a remaining
/// `Hole` is an unsolved variable, not an indirection.
fn ty_has_hole(ty: crate::semi::ty::Ty<'_>) -> bool {
    match ty.kind() {
        TyKind::Hole(_) => true,
        TyKind::Record { args, .. } => args.iter().any(|a| ty_has_hole(*a)),
        TyKind::Closure { params, ret } => {
            params.iter().any(|p| ty_has_hole(*p)) || ty_has_hole(*ret)
        }
        TyKind::Nullable(inner) => ty_has_hole(*inner),
        _ => false,
    }
}

impl<'tcx> FulfillCtxt<'tcx> {
    fn into_pending(self) -> Vec<PendingObligation<'tcx>> {
        self.pending
    }
}

#[cfg(test)]
mod tests {
    use super::ty_has_hole;
    use crate::semi::infer::InferCtxt;
    use crate::semi::ty::{Flexivity, DefId, IntTy};
    use crate::with_tcx;

    #[test]
    fn ty_has_hole_sees_holes_under_constructors() {
        with_tcx(|tcx| {
            let mut infer = InferCtxt::new(tcx);
            let hole = infer.new_hole_ty();
            let ground = tcx.mk_int(IntTy::Signed(32));

            // A bare hole and a hole nested under a constructor both count; a
            // ground type and a ground constructor do not. This is the invariant
            // the concrete-arm deep resolve in `discharge_trait` relies on: a
            // `Record<?h>` must defer, not be read as ground.
            assert!(ty_has_hole(hole));
            assert!(!ty_has_hole(ground));
            assert!(ty_has_hole(tcx.mk_record(
                DefId(0),
                &[hole],
                Flexivity::Irrelevant
            )));
            assert!(!ty_has_hole(tcx.mk_record(
                DefId(0),
                &[ground],
                Flexivity::Irrelevant
            )));
            assert!(ty_has_hole(tcx.mk_nullable(hole)));
        });
    }
}
