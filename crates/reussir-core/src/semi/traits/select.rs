//! Impl selection: discharging a trait obligation into [`Evidence`].
//!
//! Selection is the run-time face of coherence: because the overlap gate
//! guarantees at most one impl per `(trait, self-head)` pair can apply, the
//! first candidate that matches *is* the unique one, and a failing
//! where-clause on it is a hard error rather than a backtrack point.
//!
//! The matcher is [`unify_args`] with only the *candidate impl's* generics
//! as variables — the goal's own generics (an enclosing function's `T`)
//! stay rigid, so `impl<T: Num> Show for Box<T>` matches the goal
//! `Box<G> : Show` by binding the impl's `T ↦ G`, and the residual clause
//! `G : Num` is discharged against the [`ParamEnv`] assumptions.
//!
//! Known cut: a *ground* `Sync` goal is answered structurally by the
//! fulfillment pass before selection runs, but a `Sync` clause reached
//! *through* where-clause recursion has no structural hook here and reports
//! no-impl; wiring the verdict into selection lands with dispatch.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::semi::ctxt::GenericInfo;
use crate::semi::traits::coherence::{head_key, resolve_binding, unify_args};
use crate::semi::traits::def::ImplDef;
use crate::semi::traits::subst::replace_generics;
use crate::semi::traits::{Evidence, Obligation, SelectError, TraitDb, TraitId, TraitRef};
use crate::semi::ty::{GenericId, Ty, TyCtxt, TyKind};

/// The proof-search depth bound. Where-clause instantiation descends
/// structurally into the self type, so honest programs terminate; the limit
/// converts a pathological or accidental cycle into a diagnostic. Distinct
/// from monomorphization's `RECURSION_LIMIT`, which bounds *instance*
/// expansion — this bounds *proof* depth at check time.
pub const SELECT_DEPTH_LIMIT: usize = 64;

/// The in-scope assumptions selection may discharge a goal against: the
/// enclosing definition's generics with their declared bounds, dense by
/// [`GenericId`]. Empty at monomorphization time, where every self type is
/// ground.
pub type ParamEnv<'a> = &'a [GenericInfo];

/// One selection session: a database view, an assumption environment, and a
/// memo table keyed by goal. Construct per obligation sweep (or per mono
/// instance); the memo pays for repeated sub-goals within one proof.
pub struct SelectCtxt<'a, 'tcx> {
    db: &'a TraitDb<'tcx>,
    tcx: &'a TyCtxt<'tcx>,
    env: ParamEnv<'a>,
    memo: FxHashMap<(TraitId, Vec<Ty<'tcx>>), Result<Evidence<'tcx>, SelectError<'tcx>>>,
}

impl<'a, 'tcx> SelectCtxt<'a, 'tcx> {
    pub fn new(db: &'a TraitDb<'tcx>, tcx: &'a TyCtxt<'tcx>, env: ParamEnv<'a>) -> Self {
        Self {
            db,
            tcx,
            env,
            memo: FxHashMap::default(),
        }
    }

    /// Discharge an obligation, producing evidence.
    pub fn select(
        &mut self,
        obligation: &Obligation<'tcx>,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        let Obligation::Trait(goal) = obligation;
        self.select_trait(goal, 0)
    }

    fn select_trait(
        &mut self,
        goal: &TraitRef<'tcx>,
        depth: usize,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        if depth > SELECT_DEPTH_LIMIT {
            return Err(SelectError::DepthLimit(goal.clone()));
        }
        let key = (goal.trait_id, goal.args.clone());
        if let Some(hit) = self.memo.get(&key) {
            return hit.clone();
        }
        let result = self.select_uncached(goal, depth);
        // An overflow is relative to the depth it was probed at, so caching
        // it would poison shallower queries of the same goal.
        if !matches!(result, Err(SelectError::DepthLimit(_))) {
            self.memo.insert(key, result.clone());
        }
        result
    }

    fn select_uncached(
        &mut self,
        goal: &TraitRef<'tcx>,
        depth: usize,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        let db = self.db;
        let self_ty = goal.self_ty();

        // A bare in-scope generic has no head, so no impl can apply (blanket
        // impls are rejected); only the environment's assumptions prove it.
        if let TyKind::Generic(g) = *self_ty.kind() {
            return self
                .prove_by_assumption(g, self_ty, goal)
                .ok_or_else(|| SelectError::NoImpl(goal.clone()));
        }

        let Some(head) = head_key(self_ty) else {
            // `Hole` is deferred by the caller; `Bottom` is already reported.
            return Err(SelectError::NoImpl(goal.clone()));
        };

        // The direct candidates: impls of the goal trait bucketed under this
        // head. Coherence makes at most one match; commit to it.
        for &id in db.impls_for(goal.trait_id, head) {
            let imp = db.impl_def(id);
            let Some(subst) = match_impl(&imp.trait_ref.args, goal.args.as_slice(), imp) else {
                continue;
            };
            let args: Vec<Ty<'tcx>> = imp
                .generics
                .iter()
                .map(|&g| resolve_binding(self.tcx.mk_generic(g), &subst))
                .collect();
            let subst: FxHashMap<GenericId, Ty<'tcx>> = imp
                .generics
                .iter()
                .copied()
                .zip(args.iter().copied())
                .collect();
            // A failing clause on the unique applicable impl is the root
            // cause; bubble the *inner* goal for the diagnostic.
            let sub = imp
                .where_clauses
                .iter()
                .map(|clause| {
                    let Obligation::Trait(c) = clause;
                    let inst = TraitRef {
                        trait_id: c.trait_id,
                        args: c
                            .args
                            .iter()
                            .map(|&t| replace_generics(self.tcx, t, &subst))
                            .collect(),
                    };
                    self.select_trait(&inst, depth + 1)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Evidence::Impl {
                impl_id: id,
                args,
                sub,
            });
        }

        // Otherwise reach the goal through super-trait hops: an impl of a
        // sub-trait for this head whose super-closure contains the goal.
        // (This subsumes the old hard-coded class DAG: `i32 : Num` holds via
        // `impl Integral for i32` and the `Integral: Num` edge.) Impls are
        // scanned in registration order, so a (legal) diamond resolves
        // deterministically to the earliest sub-impl.
        for imp in db.impls() {
            if imp.trait_ref.trait_id == goal.trait_id
                || head_key(imp.self_ty) != Some(head)
                || !db.implies(imp.trait_ref.trait_id, goal.trait_id)
            {
                continue;
            }
            let Some(binding) = match_impl(&imp.trait_ref.args[..1], &goal.args[..1], imp) else {
                continue;
            };
            let subst: FxHashMap<GenericId, Ty<'tcx>> = imp
                .generics
                .iter()
                .map(|&g| (g, resolve_binding(self.tcx.mk_generic(g), &binding)))
                .collect();
            let inst_ref = TraitRef {
                trait_id: imp.trait_ref.trait_id,
                args: imp
                    .trait_ref
                    .args
                    .iter()
                    .map(|&t| replace_generics(self.tcx, t, &subst))
                    .collect(),
            };
            let base = match self.select_trait(&inst_ref, depth + 1) {
                Ok(base) => base,
                // Overflow is a hard stop, not a candidate that failed to
                // apply — swallowing it would mislabel the result NoImpl
                // (and poison the memo with a depth-relative failure).
                Err(err @ SelectError::DepthLimit(_)) => return Err(err),
                Err(SelectError::NoImpl(_)) => continue,
            };
            if let Some(ev) = self.project_super(&inst_ref, base, goal) {
                return Ok(ev);
            }
        }

        Err(SelectError::NoImpl(goal.clone()))
    }

    /// Prove a bare-generic goal from the environment: a declared bound that
    /// *is* the goal yields [`Evidence::Param`]; a bound whose super-closure
    /// contains the goal projects up from it.
    fn prove_by_assumption(
        &mut self,
        g: GenericId,
        self_ty: Ty<'tcx>,
        goal: &TraitRef<'tcx>,
    ) -> Option<Evidence<'tcx>> {
        let env = self.env;
        env.get(g.0 as usize)?
            .bounds
            .iter()
            .enumerate()
            .find_map(|(index, &have)| {
                let base = Evidence::Param { generic: g, index };
                let base_ref = TraitRef {
                    trait_id: have,
                    args: vec![self_ty],
                };
                if have == goal.trait_id && base_ref.args == goal.args {
                    return Some(base);
                }
                self.project_super(&base_ref, base, goal)
            })
    }

    /// Project `base` — evidence that the fully-instantiated `base_ref`
    /// holds — up the super-trait closure to `goal`, one [`Evidence::Super`]
    /// hop per edge. Args are carried through each hop by instantiating the
    /// declaring trait's binder, so a multi-parameter super-reference stays
    /// faithful.
    fn project_super(
        &self,
        base_ref: &TraitRef<'tcx>,
        base: Evidence<'tcx>,
        goal: &TraitRef<'tcx>,
    ) -> Option<Evidence<'tcx>> {
        let def = self.db.trait_def(base_ref.trait_id);
        let mut map = FxHashMap::default();
        map.insert(def.self_param, base_ref.args[0]);
        for (&(_, p), &a) in def.params.iter().zip(base_ref.args.iter().skip(1)) {
            map.insert(p, a);
        }
        def.supertraits.iter().enumerate().find_map(|(index, sup)| {
            let sup_ref = TraitRef {
                trait_id: sup.trait_id,
                args: sup
                    .args
                    .iter()
                    .map(|&t| replace_generics(self.tcx, t, &map))
                    .collect(),
            };
            let step = Evidence::Super {
                of: Box::new(base.clone()),
                index,
            };
            if sup.trait_id == goal.trait_id && sup_ref.args == goal.args {
                return Some(step);
            }
            self.project_super(&sup_ref, step, goal)
        })
    }
}

/// One-way match of an impl's argument list against a goal's: only the
/// impl's own generics bind. `None` if the shapes disagree.
fn match_impl<'tcx>(
    imp_args: &[Ty<'tcx>],
    goal_args: &[Ty<'tcx>],
    imp: &ImplDef<'tcx>,
) -> Option<FxHashMap<GenericId, Ty<'tcx>>> {
    let vars: FxHashSet<GenericId> = imp.generics.iter().copied().collect();
    let mut binding = FxHashMap::default();
    unify_args(imp_args, goal_args, &vars, &mut binding).then_some(binding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::traits::ImplId;
    use crate::semi::traits::builtins::Builtins;
    use crate::semi::traits::def::{ImplDef, TraitDef};
    use crate::semi::ty::{DefId, Flexivity, FpTy, IntTy, TyCtxt};
    use crate::with_tcx;

    fn needs(trait_id: TraitId, self_ty: Ty<'_>) -> Obligation<'_> {
        Obligation::Trait(TraitRef {
            trait_id,
            args: vec![self_ty],
        })
    }

    fn trait_def<'tcx>(id: TraitId, def: u32, supers: Vec<TraitRef<'tcx>>) -> TraitDef<'tcx> {
        TraitDef {
            id,
            def: DefId(def),
            visibility: crate::surface::Visibility::Public,
            sealed: false,
            self_param: GenericId(0),
            params: vec![],
            supertraits: supers,
            methods: vec![],
            assoc_tys: vec![],
            span: None,
            file: reussir_syntax::source::FileId::ROOT,
        }
    }

    fn impl_def<'tcx>(
        id: u32,
        generics: Vec<GenericId>,
        trait_id: TraitId,
        self_ty: Ty<'tcx>,
        where_clauses: Vec<Obligation<'tcx>>,
    ) -> ImplDef<'tcx> {
        ImplDef {
            id: ImplId(id),
            generics,
            trait_ref: TraitRef {
                trait_id,
                args: vec![self_ty],
            },
            self_ty,
            where_clauses,
            methods: vec![],
            span: None,
            file: reussir_syntax::source::FileId::ROOT,
        }
    }

    #[test]
    fn primitive_membership_is_data_not_hardcoded() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let mut defs = crate::semi::resolve::DefTable::new();
            let mut interner = lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new();
            let b = Builtins::register(&mut db, &mut defs, &mut interner, tcx);
            let mut sel = SelectCtxt::new(&db, tcx, &[]);

            let i32 = tcx.mk_int(IntTy::Signed(32));
            let f32 = tcx.mk_fp(FpTy::Ieee(32));

            // i32 is Integral directly, and Num through the super-trait edge.
            assert!(sel.select(&needs(b.integral, i32)).is_ok());
            assert!(sel.select(&needs(b.num, i32)).is_ok());

            // f32 is FloatingPoint and (super) Num, but not Integral.
            assert!(sel.select(&needs(b.floating_point, f32)).is_ok());
            assert!(sel.select(&needs(b.num, f32)).is_ok());
            assert!(sel.select(&needs(b.integral, f32)).is_err());

            // bool belongs to no numeric trait.
            assert!(sel.select(&needs(b.num, tcx.mk_bool())).is_err());
        });
    }

    #[test]
    fn transitive_super_traits_build_a_full_evidence_chain() {
        with_tcx(|tcx: &TyCtxt| {
            // A 3-level chain: Sub : Mid : Top.
            let mut db = TraitDb::new();
            let self_ref = |t: TraitId| TraitRef {
                trait_id: t,
                args: vec![tcx.mk_generic(GenericId(0))],
            };
            let (top, mid, sub) = (TraitId(0), TraitId(1), TraitId(2));
            db.add_trait(trait_def(top, 0, vec![]));
            db.add_trait(trait_def(mid, 1, vec![self_ref(top)]));
            db.add_trait(trait_def(sub, 2, vec![self_ref(mid)]));

            let unit = tcx.mk_unit();
            db.add_impl(impl_def(0, vec![], sub, unit, vec![]));

            // `unit: Top` is two hops away (Sub -> Mid -> Top): the evidence must
            // nest two `Super` projections over the `Sub` impl.
            let mut sel = SelectCtxt::new(&db, tcx, &[]);
            let ev = sel
                .select(&needs(top, unit))
                .expect("unit: Top should hold");
            match ev {
                Evidence::Super {
                    of: outer,
                    index: 0,
                } => match *outer {
                    Evidence::Super {
                        of: inner,
                        index: 0,
                    } => {
                        assert!(matches!(*inner, Evidence::Impl { .. }))
                    }
                    other => panic!("expected a nested Super, got {other:?}"),
                },
                other => panic!("expected a two-hop Super chain, got {other:?}"),
            }
        });
    }

    #[test]
    fn generic_impl_matches_and_discharges_clauses() {
        with_tcx(|tcx: &TyCtxt| {
            // trait Num; trait Show; struct Box<T>;
            // impl Num for i32; impl<T: Num> Show for Box<T>;
            let mut db = TraitDb::new();
            let (num, show) = (TraitId(0), TraitId(1));
            db.add_trait(trait_def(num, 0, vec![]));
            db.add_trait(trait_def(show, 1, vec![]));
            let bx = DefId(10);
            let i32t = tcx.mk_int(IntTy::Signed(32));
            let t = GenericId(3);
            let box_t = tcx.mk_record(bx, &[tcx.mk_generic(t)], Flexivity::Irrelevant);
            db.add_impl(impl_def(0, vec![], num, i32t, vec![]));
            db.add_impl(impl_def(
                1,
                vec![t],
                show,
                box_t,
                vec![needs(num, tcx.mk_generic(t))],
            ));

            // Ground: Box<i32> : Show — the impl's T grounds to i32 and the
            // clause i32 : Num holds by the direct impl.
            let box_i32 = tcx.mk_record(bx, &[i32t], Flexivity::Irrelevant);
            let mut sel = SelectCtxt::new(&db, tcx, &[]);
            match sel.select(&needs(show, box_i32)).expect("Box<i32>: Show") {
                Evidence::Impl { impl_id, args, sub } => {
                    assert_eq!(impl_id, ImplId(1));
                    assert_eq!(args, vec![i32t], "T instantiated to i32");
                    assert!(matches!(
                        sub[0],
                        Evidence::Impl {
                            impl_id: ImplId(0),
                            ..
                        }
                    ));
                }
                other => panic!("expected the generic impl, got {other:?}"),
            }

            // The failing clause is the root cause: Box<bool> matches the
            // impl but bool : Num has no proof, and THAT goal bubbles.
            let box_bool = tcx.mk_record(bx, &[tcx.mk_bool()], Flexivity::Irrelevant);
            match sel.select(&needs(show, box_bool)) {
                Err(SelectError::NoImpl(inner)) => {
                    assert_eq!(inner.trait_id, num);
                    assert_eq!(inner.self_ty(), tcx.mk_bool());
                }
                other => panic!("expected the inner Num goal to fail, got {other:?}"),
            }
        });
    }

    #[test]
    fn assumptions_prove_generic_goals_with_projection() {
        with_tcx(|tcx: &TyCtxt| {
            // trait Num; trait Integral: Num; fn f<G: Integral> — inside f,
            // G : Integral holds by assumption and G : Num by projection.
            let mut db = TraitDb::new();
            let (num, integral) = (TraitId(0), TraitId(1));
            db.add_trait(trait_def(num, 0, vec![]));
            db.add_trait(trait_def(
                integral,
                1,
                vec![TraitRef {
                    trait_id: num,
                    args: vec![tcx.mk_generic(GenericId(0))],
                }],
            ));

            let g = GenericId(0);
            let key =
                lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new().get_or_intern_static("G");
            let env = vec![crate::semi::ctxt::GenericInfo {
                name: key,
                bounds: vec![integral],
            }];
            let g_ty = tcx.mk_generic(g);
            let mut sel = SelectCtxt::new(&db, tcx, &env);
            match sel.select(&needs(integral, g_ty)).expect("G: Integral") {
                Evidence::Param { generic, index } => {
                    assert_eq!(generic, g);
                    assert_eq!(index, 0);
                }
                other => panic!("expected the assumption itself, got {other:?}"),
            }
            match sel.select(&needs(num, g_ty)).expect("G: Num via super") {
                Evidence::Super { of, index: 0 } => {
                    assert!(matches!(*of, Evidence::Param { .. }))
                }
                other => panic!("expected projection over the assumption, got {other:?}"),
            }
            // An unrelated goal is not assumed.
            let other_trait = TraitId(0);
            let unbound = tcx.mk_generic(GenericId(7));
            assert!(sel.select(&needs(other_trait, unbound)).is_err());
        });
    }

    #[test]
    fn deep_recursion_hits_the_depth_limit() {
        with_tcx(|tcx: &TyCtxt| {
            // impl<T: Show> Show for Box<T> — proving Box^n<i32> : Show
            // descends n clause levels. Shallow nesting fails with the root
            // NoImpl (i32 : Show missing); past the limit it overflows.
            let mut db = TraitDb::new();
            let show = TraitId(0);
            db.add_trait(trait_def(show, 0, vec![]));
            let bx = DefId(10);
            let t = GenericId(0);
            let box_t = tcx.mk_record(bx, &[tcx.mk_generic(t)], Flexivity::Irrelevant);
            db.add_impl(impl_def(
                0,
                vec![t],
                show,
                box_t,
                vec![needs(show, tcx.mk_generic(t))],
            ));

            let nest = |depth: usize| {
                let mut ty = tcx.mk_int(IntTy::Signed(32));
                for _ in 0..depth {
                    ty = tcx.mk_record(bx, &[ty], Flexivity::Irrelevant);
                }
                ty
            };
            let mut sel = SelectCtxt::new(&db, tcx, &[]);
            match sel.select(&needs(show, nest(10))) {
                Err(SelectError::NoImpl(inner)) => {
                    assert_eq!(inner.self_ty(), tcx.mk_int(IntTy::Signed(32)));
                }
                other => panic!("expected the i32 root cause, got {other:?}"),
            }
            // A fresh session: the shallow probe above memoized its NoImpl
            // prefixes, which would truncate this descent below the limit.
            let mut sel = SelectCtxt::new(&db, tcx, &[]);
            assert!(matches!(
                sel.select(&needs(show, nest(SELECT_DEPTH_LIMIT + 6))),
                Err(SelectError::DepthLimit(_))
            ));
        });
    }
}
