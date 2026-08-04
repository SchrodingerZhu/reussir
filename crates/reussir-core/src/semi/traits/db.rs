//! The trait/impl registry and a trivial resolver.
//!
//! Phase 0 resolves only *ground* obligations: a concrete self type is matched
//! exactly against impls (a pointer comparison, thanks to interning), with
//! super-trait projection, plus the capability lattice. Candidate gathering from
//! in-scope assumptions, one-way matching of generic impls, where-clause
//! discharge against an environment, and suspension of obligations whose self
//! type is still a hole all arrive in Phase 1 — slotting into this same
//! [`TraitDb::select`] entry point.

use rustc_hash::FxHashMap;

use crate::semi::ty::DefId;

use super::def::{ImplDef, TraitDef};
use super::{Evidence, ImplId, Obligation, TraitId, TraitRef};

/// The collected trait program plus resolution.
#[derive(Default)]
pub struct TraitDb<'tcx> {
    traits: Vec<TraitDef<'tcx>>,
    impls: Vec<ImplDef<'tcx>>,
    /// The dense [`TraitId`] of each path-keyed trait def.
    by_def: FxHashMap<DefId, TraitId>,
}

/// Why an obligation could not be discharged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError<'tcx> {
    /// No impl makes `τ : Trait` hold.
    NoImpl(TraitRef<'tcx>),
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
        self.impls.truncate(impls);
    }

    pub fn add_impl(&mut self, def: ImplDef<'tcx>) -> ImplId {
        // Keep `ImplId`s dense and in storage order so they can index `impls`.
        debug_assert_eq!(
            def.id.0 as usize,
            self.impls.len(),
            "impl ids must be dense"
        );
        let id = def.id;
        self.impls.push(def);
        id
    }

    pub fn trait_def(&self, id: TraitId) -> &TraitDef<'tcx> {
        &self.traits[id.0 as usize]
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

    /// Discharge an obligation, producing evidence.
    pub fn select(
        &self,
        obligation: &Obligation<'tcx>,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        let Obligation::Trait(goal) = obligation;
        self.select_trait(goal)
    }

    fn select_trait(&self, goal: &TraitRef<'tcx>) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        // A direct impl of the goal trait for this exact self type. With
        // interned types this self-type check is a pointer comparison.
        for imp in &self.impls {
            if imp.trait_ref.trait_id == goal.trait_id && imp.self_ty == goal.self_ty() {
                let mut sub = Vec::with_capacity(imp.where_clauses.len());
                for clause in &imp.where_clauses {
                    sub.push(self.select(clause)?);
                }
                return Ok(Evidence::Impl {
                    impl_id: imp.id,
                    args: imp.trait_ref.args.clone(),
                    sub,
                });
            }
        }

        // Otherwise, reach the goal through one or more super-trait hops: some
        // impl for this self type implements a sub-trait whose super-trait
        // closure contains the goal. (This is what the old class DAG did by
        // hand.) The evidence projects the impl up the full hop chain.
        for imp in &self.impls {
            if imp.self_ty != goal.self_ty() || imp.trait_ref.trait_id == goal.trait_id {
                continue;
            }
            let Ok(base) = self.select_trait(&imp.trait_ref) else {
                continue;
            };
            let sub_def = self.trait_def(imp.trait_ref.trait_id);
            if let Some(ev) = self.project_super(sub_def, base, goal.trait_id) {
                return Ok(ev);
            }
        }

        Err(SelectError::NoImpl(goal.clone()))
    }

    /// Project `of` — evidence that the self type implements `def` — up to
    /// `target` through super-traits, building one [`Evidence::Super`] hop per
    /// edge on the path. `None` if `target` is not in `def`'s super-trait
    /// closure.
    fn project_super(
        &self,
        def: &TraitDef<'tcx>,
        of: Evidence<'tcx>,
        target: TraitId,
    ) -> Option<Evidence<'tcx>> {
        for (i, sup) in def.supertraits.iter().enumerate() {
            let step = Evidence::Super {
                of: Box::new(of.clone()),
                index: i,
            };
            if sup.trait_id == target {
                return Some(step);
            }
            if let Some(ev) = self.project_super(self.trait_def(sup.trait_id), step, target) {
                return Some(ev);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::traits::builtins::Builtins;
    use crate::semi::ty::{FpTy, IntTy, Ty, TyCtxt};
    use crate::with_tcx;

    fn needs(trait_id: TraitId, self_ty: Ty<'_>) -> Obligation<'_> {
        Obligation::Trait(TraitRef {
            trait_id,
            args: vec![self_ty],
        })
    }

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

    #[test]
    fn primitive_membership_is_data_not_hardcoded() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let mut defs = crate::semi::resolve::DefTable::new();
            let mut interner = lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new();
            let b = Builtins::register(&mut db, &mut defs, &mut interner, tcx);

            let i32 = tcx.mk_int(IntTy::Signed(32));
            let f32 = tcx.mk_fp(FpTy::Ieee(32));

            // i32 is Integral directly, and Num through the super-trait edge.
            assert!(db.select(&needs(b.integral, i32)).is_ok());
            assert!(db.select(&needs(b.num, i32)).is_ok());

            // f32 is FloatingPoint and (super) Num, but not Integral.
            assert!(db.select(&needs(b.floating_point, f32)).is_ok());
            assert!(db.select(&needs(b.num, f32)).is_ok());
            assert!(db.select(&needs(b.integral, f32)).is_err());

            // bool belongs to no numeric trait.
            assert!(db.select(&needs(b.num, tcx.mk_bool())).is_err());
        });
    }

    // `Sync` has no impls: ground goals are answered structurally by
    // `traits::sync` (see its tests), so nothing here selects it.

    #[test]
    fn transitive_super_traits_build_a_full_evidence_chain() {
        use crate::semi::traits::def::{ImplDef, TraitDef};
        use crate::semi::traits::{Evidence, ImplId, TraitRef};

        with_tcx(|tcx: &TyCtxt| {
            // A 3-level chain: Sub : Mid : Top.
            let mut db = TraitDb::new();
            let self_g = crate::semi::ty::GenericId(0);
            let self_ref = |t: TraitId| TraitRef {
                trait_id: t,
                args: vec![tcx.mk_generic(self_g)],
            };
            let (top, mid, sub) = (TraitId(0), TraitId(1), TraitId(2));
            let def = |id: TraitId, def: u32, supers| TraitDef {
                id,
                def: crate::semi::ty::DefId(def),
                visibility: crate::surface::Visibility::Public,
                sealed: false,
                self_param: self_g,
                params: vec![],
                supertraits: supers,
                methods: vec![],
                assoc_tys: vec![],
                span: None,
                file: reussir_syntax::source::FileId::ROOT,
            };
            db.add_trait(def(top, 0, vec![]));
            db.add_trait(def(mid, 1, vec![self_ref(top)]));
            db.add_trait(def(sub, 2, vec![self_ref(mid)]));

            let unit = tcx.mk_unit();
            db.add_impl(ImplDef {
                id: ImplId(0),
                generics: vec![],
                trait_ref: TraitRef {
                    trait_id: sub,
                    args: vec![unit],
                },
                self_ty: unit,
                where_clauses: vec![],
            });

            // `unit: Top` is two hops away (Sub -> Mid -> Top): the evidence must
            // nest two `Super` projections over the `Sub` impl.
            let ev = db.select(&needs(top, unit)).expect("unit: Top should hold");
            match ev {
                Evidence::Super {
                    of: outer,
                    index: 0,
                } => match *outer {
                    Evidence::Super {
                        of: inner,
                        index: 0,
                    } => assert!(matches!(*inner, Evidence::Impl { .. })),
                    other => panic!("expected a nested Super, got {other:?}"),
                },
                other => panic!("expected a two-hop Super chain, got {other:?}"),
            }
        });
    }
}
