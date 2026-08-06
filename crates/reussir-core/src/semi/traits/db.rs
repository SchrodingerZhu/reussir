//! The trait/impl registry: dense-id storage, the `(trait, head)` candidate
//! index, super-trait reachability, and REPL-checkpoint truncation.
//!
//! Resolution lives in [`super::select::SelectCtxt`], which reads this
//! registry; coherence (the overlap and orphan gates in the elaborator)
//! is what entitles selection to commit to the first matching candidate.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::semi::ty::DefId;

use super::coherence::{HeadKey, head_key};
use super::def::{ImplDef, TraitDef};
use super::{ImplId, TraitId, TraitRef};

/// The collected trait program plus resolution.
#[derive(Default)]
pub struct TraitDb<'tcx> {
    traits: Vec<TraitDef<'tcx>>,
    impls: Vec<ImplDef<'tcx>>,
    /// The dense [`TraitId`] of each path-keyed trait def.
    by_def: FxHashMap<DefId, TraitId>,
    /// Impls bucketed by `(trait, self-head)` — the candidate index for
    /// overlap checking (and, in the selection stack, for `select`).
    by_head: FxHashMap<(TraitId, HeadKey), SmallVec<[ImplId; 2]>>,
}

/// Why an obligation could not be discharged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError<'tcx> {
    /// No impl makes `τ : Trait` hold. Carries the *deepest* failing goal:
    /// a where-clause failure on the unique applicable impl surfaces the
    /// inner obligation, which is the root cause.
    NoImpl(TraitRef<'tcx>),
    /// Proof search exceeded [`super::select::SELECT_DEPTH_LIMIT`].
    DepthLimit(TraitRef<'tcx>),
}

impl<'tcx> TraitDb<'tcx> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_trait(&mut self, def: TraitDef<'tcx>) -> TraitId {
        // `trait_def` indexes `traits` by `TraitId`, so ids must be dense and
        // inserted in order.
        debug_assert_eq!(
            def.id.0 as usize,
            self.traits.len(),
            "trait ids must be dense"
        );
        let id = def.id;
        let displaced = self.by_def.insert(def.def, id);
        debug_assert!(displaced.is_none(), "one TraitDb entry per trait def");
        self.traits.push(def);
        id
    }

    /// The dense id of the trait declared under `def`, if any. Every
    /// trait-namespace def has an entry — the elaborator registers them
    /// together.
    pub fn trait_by_def(&self, def: DefId) -> Option<TraitId> {
        self.by_def.get(&def).copied()
    }

    /// Checkpoint counters for [`TraitDb::truncate`].
    pub fn traits_len(&self) -> usize {
        self.traits.len()
    }

    pub fn impls_len(&self) -> usize {
        self.impls.len()
    }

    /// Retract every trait past `traits` and every impl past `impls` — the
    /// REPL's atomic-rollback hook, mirroring `DefTable::truncate` (ids are
    /// dense, so popping the tails restores exactly the checkpointed state).
    pub fn truncate(&mut self, traits: usize, impls: usize) {
        while self.traits.len() > traits {
            let def = self.traits.pop().expect("len checked above");
            let removed = self.by_def.remove(&def.def);
            debug_assert_eq!(
                removed,
                Some(def.id),
                "by_def must point at the popped trait"
            );
        }
        while self.impls.len() > impls {
            let def = self.impls.pop().expect("len checked above");
            let head = head_key(def.self_ty).expect("registered with a head");
            let bucket = self
                .by_head
                .get_mut(&(def.trait_ref.trait_id, head))
                .expect("registered impls have a bucket");
            let popped = bucket.pop();
            debug_assert_eq!(popped, Some(def.id), "buckets append in id order");
            if bucket.is_empty() {
                self.by_head.remove(&(def.trait_ref.trait_id, head));
            }
        }
    }

    pub fn add_impl(&mut self, def: ImplDef<'tcx>) -> ImplId {
        // Keep `ImplId`s dense and in storage order so they can index `impls`.
        debug_assert_eq!(
            def.id.0 as usize,
            self.impls.len(),
            "impl ids must be dense"
        );
        let id = def.id;
        let head = head_key(def.self_ty).expect("blanket impls are rejected before registration");
        self.by_head
            .entry((def.trait_ref.trait_id, head))
            .or_default()
            .push(id);
        self.impls.push(def);
        id
    }

    /// The impls of `trait_id` whose self head is `head`.
    pub fn impls_for(&self, trait_id: TraitId, head: HeadKey) -> &[ImplId] {
        self.by_head
            .get(&(trait_id, head))
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    pub fn trait_def(&self, id: TraitId) -> &TraitDef<'tcx> {
        &self.traits[id.0 as usize]
    }

    pub fn impl_def(&self, id: ImplId) -> &ImplDef<'tcx> {
        &self.impls[id.0 as usize]
    }

    /// Mutable access for the elaborator's two-phase populate: trait stubs
    /// register first (so bounds resolve during the record/function scans),
    /// then supertraits and members fill in.
    pub(crate) fn trait_def_mut(&mut self, id: TraitId) -> &mut TraitDef<'tcx> {
        &mut self.traits[id.0 as usize]
    }

    /// Does holding `have` imply holding `want`? True when they are equal or
    /// `want` is in `have`'s super-trait closure.
    pub fn implies(&self, have: TraitId, want: TraitId) -> bool {
        have == want || self.reaches(self.trait_def(have), want)
    }

    /// Is `target` in `def`'s transitive super-trait closure?
    fn reaches(&self, def: &TraitDef<'tcx>, target: TraitId) -> bool {
        def.supertraits
            .iter()
            .any(|s| s.trait_id == target || self.reaches(self.trait_def(s.trait_id), target))
    }

    /// All impls in registration (dense-id) order — the deterministic scan
    /// order for super-trait projection in selection.
    pub fn impls(&self) -> impl Iterator<Item = &ImplDef<'tcx>> {
        self.impls.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::traits::builtins::Builtins;
    use crate::semi::ty::TyCtxt;
    use crate::with_tcx;

    #[test]
    fn truncate_retracts_traits_and_impls() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let mut defs = crate::semi::resolve::DefTable::new();
            let mut interner = lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new();
            let b = Builtins::register(&mut db, &mut defs, &mut interner, tcx);
            let (traits, impls) = (db.traits_len(), db.impls_len());

            // Register a user trait + impl past the builtins, then retract.
            let key = reussir_syntax::Interner::get_or_intern(&mut interner, "Show");
            let def = defs.declare_trait(key).expect("fresh");
            let id = TraitId(traits as u32);
            db.add_trait(TraitDef {
                id,
                def,
                visibility: crate::surface::Visibility::Public,
                sealed: false,
                self_param: crate::semi::ty::GenericId(0),
                params: vec![],
                supertraits: vec![],
                methods: vec![],
                assoc_tys: vec![],
                span: None,
                file: reussir_syntax::source::FileId::ROOT,
            });
            let unit = tcx.mk_unit();
            db.add_impl(ImplDef {
                id: ImplId(impls as u32),
                generics: vec![],
                trait_ref: TraitRef {
                    trait_id: id,
                    args: vec![unit],
                },
                self_ty: unit,
                where_clauses: vec![],
                methods: vec![],
                span: None,
                file: reussir_syntax::source::FileId::ROOT,
            });
            assert_eq!(db.trait_by_def(def), Some(id));

            db.truncate(traits, impls);
            assert_eq!(db.traits_len(), traits);
            assert_eq!(db.impls_len(), impls);
            assert_eq!(db.trait_by_def(def), None);
            // The builtins survive and re-adding at the same dense ids works.
            assert_eq!(db.trait_by_def(db.trait_def(b.num).def), Some(b.num));
            db.add_trait(TraitDef {
                id,
                def,
                visibility: crate::surface::Visibility::Public,
                sealed: false,
                self_param: crate::semi::ty::GenericId(0),
                params: vec![],
                supertraits: vec![],
                methods: vec![],
                assoc_tys: vec![],
                span: None,
                file: reussir_syntax::source::FileId::ROOT,
            });
            assert_eq!(db.trait_by_def(def), Some(id));
        });
    }

    #[test]
    fn trait_by_def_round_trips_builtins() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let mut defs = crate::semi::resolve::DefTable::new();
            let mut interner = lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new();
            let b = Builtins::register(&mut db, &mut defs, &mut interner, tcx);
            for id in [b.num, b.integral, b.floating_point, b.ptr_like, b.sync] {
                assert_eq!(db.trait_by_def(db.trait_def(id).def), Some(id));
            }
        });
    }
}
