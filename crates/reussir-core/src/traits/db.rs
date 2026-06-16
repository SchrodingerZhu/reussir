//! The trait/impl registry and a trivial resolver.
//!
//! Phase 0 resolves only *ground* obligations: a concrete self type is matched
//! exactly against impls (a pointer comparison, thanks to interning), with
//! super-trait projection, plus the capability lattice. Candidate gathering from
//! in-scope assumptions, one-way matching of generic impls, where-clause
//! discharge against an environment, and suspension of obligations whose self
//! type is still a hole all arrive in Phase 1 — slotting into this same
//! [`TraitDb::select`] entry point.

use super::def::{ImplDef, TraitDef};
use super::{Evidence, ImplId, Obligation, TraitId, TraitRef};

/// The collected trait program plus resolution.
#[derive(Default)]
pub struct TraitDb<'tcx> {
    traits: Vec<TraitDef<'tcx>>,
    impls: Vec<ImplDef<'tcx>>,
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
        self.traits.push(def);
        id
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
    use crate::traits::builtins::Builtins;
    use crate::ty::{FpTy, IntTy, Ty, TyCtxt};
    use crate::with_tcx;

    fn needs(trait_id: TraitId, self_ty: Ty<'_>) -> Obligation<'_> {
        Obligation::Trait(TraitRef {
            trait_id,
            args: vec![self_ty],
        })
    }

    #[test]
    fn primitive_membership_is_data_not_hardcoded() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let b = Builtins::register(&mut db, tcx);

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

    #[test]
    fn primitive_scalars_are_send_and_sync() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let b = Builtins::register(&mut db, tcx);
            for ty in [tcx.mk_int(IntTy::Unsigned(64)), tcx.mk_bool(), tcx.mk_str()] {
                assert!(db.select(&needs(b.send, ty)).is_ok());
                assert!(db.select(&needs(b.sync, ty)).is_ok());
            }
        });
    }

    #[test]
    fn transitive_super_traits_build_a_full_evidence_chain() {
        use crate::traits::def::{ImplDef, TraitDef};
        use crate::traits::{Evidence, ImplId, TraitRef};

        with_tcx(|tcx: &TyCtxt| {
            // A 3-level chain: Sub : Mid : Top.
            let mut db = TraitDb::new();
            let self_g = crate::ty::GenericId(0);
            let self_ref = |t: TraitId| TraitRef {
                trait_id: t,
                args: vec![tcx.mk_generic(self_g)],
            };
            let (top, mid, sub) = (TraitId(0), TraitId(1), TraitId(2));
            let def = |id: TraitId, name: &str, supers| TraitDef {
                id,
                name: name.to_owned(),
                params: vec![self_g],
                supertraits: supers,
                methods: vec![],
                assoc_tys: vec![],
            };
            db.add_trait(def(top, "Top", vec![]));
            db.add_trait(def(mid, "Mid", vec![self_ref(top)]));
            db.add_trait(def(sub, "Sub", vec![self_ref(mid)]));

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
