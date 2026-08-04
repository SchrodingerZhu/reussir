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

use crate::semi::traits::{Obligation, SelectCtxt, SelectError, TraitRef};
use crate::semi::ty::{FpTy, HoleId, IntTy, TyKind};
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
            // The unresolved bound is the root cause; suppress the zonk-time
            // "cannot infer" report for the holes it is stuck on, so each
            // hole yields exactly one diagnostic.
            let Obligation::Trait(tref) = &p.obligation;
            let self_ty = self.infer.resolve(tref.self_ty());
            let mut holes = Vec::new();
            collect_holes(self_ty, &mut holes);
            self.reported_holes.extend(holes);
        }
    }

    /// REPL defaulting: solve numeric holes that only literal/arithmetic
    /// bounds constrain. A hole pending a `FloatingPoint` obligation defaults
    /// to `f64`; a hole pending `Integral` or `Num` defaults to `i64` (the
    /// generalization of GHCi-style defaulting the old REPL applied to the
    /// root type only). Obligations stay registered — the subsequent
    /// [`resolve_obligations`](Elaborator::resolve_obligations) discharges
    /// them against the now-solved types, and still errors on genuinely
    /// conflicting bounds.
    ///
    /// Order matters: a hole can carry both `Num` (from arithmetic) and
    /// `FloatingPoint` (from a float literal), e.g. `1.5 * 2`; the float
    /// default must win, so `FloatingPoint` holes are solved first.
    pub(super) fn default_numeric_holes(&mut self) {
        let f64_ty = self.tcx.mk_fp(FpTy::Ieee(64));
        let i64_ty = self.tcx.mk_int(IntTy::Signed(64));
        let passes = [
            (self.builtins.floating_point, f64_ty),
            (self.builtins.integral, i64_ty),
            (self.builtins.num, i64_ty),
        ];
        for (want, default) in passes {
            self.fulfill
                .pending
                .iter()
                .filter_map(|p| {
                    let Obligation::Trait(tref) = &p.obligation;
                    (tref.trait_id == want).then(|| tref.self_ty())
                })
                .for_each(|ty| {
                    let ty = self.infer.shallow_resolve(ty);
                    if matches!(ty.kind(), TyKind::Hole(_)) {
                        // Unifying a hole with a ground type cannot fail.
                        let _ = self.infer.unify(ty, default);
                    }
                });
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
    /// The (impl) and (param) rules are realized by [`SelectCtxt`]: the goal
    /// carries *all* of the obligation's arguments (deeply resolved), impl
    /// search one-way-matches generic impls, residual where-clauses recurse,
    /// and a bare-generic self is proved from the enclosing definition's
    /// declared bounds — so `Box<G> : Show` holds under `impl<T: Num> Show
    /// for Box<T>` when `G : Num` is assumed. A goal whose arguments still
    /// hold holes (head or nested, `List<?h>`) defers instead of failing
    /// spuriously.
    fn discharge_trait(&mut self, tref: &TraitRef<'tcx>) -> Discharge {
        let self_ty = self.infer.shallow_resolve(tref.self_ty());
        match self_ty.kind() {
            TyKind::Hole(_) => Discharge::Defer,
            // In-scope generic: satisfied iff a declared bound proves the
            // goal (directly, or through the super-trait closure).
            TyKind::Generic(_) => {
                let args: Vec<_> = tref.args.iter().map(|&a| self.infer.resolve(a)).collect();
                if args.iter().any(|&a| ty_has_hole(a)) {
                    return Discharge::Defer;
                }
                let goal = Obligation::Trait(TraitRef {
                    trait_id: tref.trait_id,
                    args,
                });
                let mut select = SelectCtxt::new(&self.traits, self.tcx, &self.generics);
                match select.select(&goal) {
                    Ok(_) => Discharge::Solved,
                    Err(err) => {
                        let name = self.trait_display(tref.trait_id);
                        Discharge::Failed(match err {
                            SelectError::NoImpl(_) => {
                                format!("the bound `{name}` is not satisfied by this generic")
                            }
                            SelectError::DepthLimit(g) => self.depth_limit_message(&g),
                        })
                    }
                }
            }
            _ => {
                // Deeply resolve before deciding: a type whose *head* is a
                // constructor may still hold holes under it (`List<?h>`).
                // Defer while any hole remains so the obligation is retried
                // once more of the substitution is known.
                let args: Vec<_> = tref.args.iter().map(|&a| self.infer.resolve(a)).collect();
                if args.iter().any(|&a| ty_has_hole(a)) {
                    return Discharge::Defer;
                }
                let self_ty = args[0];
                // `Sync` is the structural thread-safety auto trait: answered
                // by the checker over the type's shape, never by impl search
                // (docs/design/thread-safety.md §2.2).
                if tref.trait_id == self.builtins.sync {
                    use crate::semi::traits::sync::{SyncVerdict, sync_verdict};
                    use crate::semi::ty_eval::ElabSyncEnv;
                    return match sync_verdict(&ElabSyncEnv { el: self }, self_ty) {
                        SyncVerdict::Sync => Discharge::Solved,
                        SyncVerdict::NotSync(w) => Discharge::Failed(format!(
                            "`{}` is not `Sync`: {}",
                            self.ty_display(self_ty),
                            w.describe()
                        )),
                        // The verdict depends on a generic nested under a
                        // constructor: everything grounds at monomorphization
                        // (where the `Arc` wf backstop re-checks), so be
                        // optimistic here rather than reporting a spurious
                        // ambiguity.
                        SyncVerdict::Blocked => Discharge::Solved,
                    };
                }
                let goal = Obligation::Trait(TraitRef {
                    trait_id: tref.trait_id,
                    args,
                });
                let mut select = SelectCtxt::new(&self.traits, self.tcx, &self.generics);
                match select.select(&goal) {
                    Ok(_) => Discharge::Solved,
                    Err(SelectError::NoImpl(inner)) => {
                        Discharge::Failed(self.no_impl_message(self_ty, tref.trait_id, &inner))
                    }
                    Err(SelectError::DepthLimit(g)) => {
                        Discharge::Failed(self.depth_limit_message(&g))
                    }
                }
            }
        }
    }

    /// "`τ` does not implement `Trait`", with the bubbled inner goal named
    /// as the root cause when a where-clause on the chosen impl failed.
    /// Shared by obligation discharge and trait-method dispatch.
    pub(super) fn no_impl_message(
        &self,
        self_ty: crate::semi::ty::Ty<'tcx>,
        trait_id: crate::semi::traits::TraitId,
        inner: &TraitRef<'tcx>,
    ) -> String {
        let name = self.trait_display(trait_id);
        let mut msg = format!("`{}` does not implement `{name}`", self.ty_display(self_ty));
        // A where-clause failure on the chosen impl bubbles the inner goal;
        // name the root cause.
        if inner.trait_id != trait_id || inner.self_ty() != self_ty {
            msg.push_str(&format!(
                ": `{}` does not implement `{}`",
                self.ty_display(inner.self_ty()),
                self.trait_display(inner.trait_id)
            ));
        }
        msg
    }

    pub(super) fn depth_limit_message(&self, goal: &TraitRef<'tcx>) -> String {
        use crate::semi::traits::select::SELECT_DEPTH_LIMIT;
        format!(
            "overflow evaluating the bound `{}: {}` (depth limit {SELECT_DEPTH_LIMIT})",
            self.ty_display(goal.self_ty()),
            self.trait_display(goal.trait_id)
        )
    }
}

/// Does `ty` contain an unsolved inference hole anywhere in its structure?
/// `ty` is expected to be deeply resolved (`InferCtxt::resolve`), so a remaining
/// `Hole` is an unsolved variable, not an indirection.
pub(super) fn ty_has_hole(ty: crate::semi::ty::Ty<'_>) -> bool {
    match ty.kind() {
        TyKind::Hole(_) => true,
        TyKind::Record { args, .. } => args.iter().any(|a| ty_has_hole(*a)),
        TyKind::Closure { params, ret } => {
            params.iter().any(|p| ty_has_hole(*p)) || ty_has_hole(*ret)
        }
        TyKind::Nullable(inner) => ty_has_hole(*inner),
        TyKind::Cell { elem: inner, .. } => ty_has_hole(*inner),
        TyKind::Arc(inner) => ty_has_hole(*inner),
        TyKind::Array { elem, .. } => ty_has_hole(*elem),
        _ => false,
    }
}

/// Collect every hole in a deeply-resolved type, in structural order.
/// Companion to [`ty_has_hole`]; used to report each hole once (see
/// `Elaborator::zonk_ty` and the unresolved-bound path of
/// [`Elaborator::resolve_obligations`]).
pub(super) fn collect_holes(ty: crate::semi::ty::Ty<'_>, out: &mut Vec<HoleId>) {
    match ty.kind() {
        TyKind::Hole(h) => out.push(*h),
        TyKind::Record { args, .. } => args.iter().for_each(|a| collect_holes(*a, out)),
        TyKind::Closure { params, ret } => {
            params.iter().for_each(|p| collect_holes(*p, out));
            collect_holes(*ret, out);
        }
        TyKind::Nullable(inner) => collect_holes(*inner, out),
        TyKind::Cell { elem: inner, .. } => collect_holes(*inner, out),
        TyKind::Arc(inner) => collect_holes(*inner, out),
        TyKind::Array { elem, .. } => collect_holes(*elem, out),
        _ => {}
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
    use crate::semi::ty::{DefId, Flexivity, IntTy};
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
