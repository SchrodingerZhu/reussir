//! Pattern-match elaboration into decision trees.
//!
//! Phase 1 compiles a single scrutinee column: it type-checks every arm's
//! pattern and body, then switches on the top-level pattern (enum variant or
//! constant), binding nested sub-patterns by projection paths. Match guards and
//! deeply-nested refutable sub-patterns are simplified — see the notes inline.

use crate::infer::Instantiation;
use crate::surface::{self, Const, PatternKind, Span};
use crate::ty::{Ty, TyKind};
use crate::utils::string::StringToken;

use super::ctxt::{Elaborator, RecordFields};
use super::hir::{DecisionTree, Expr, ExprKind, PatVarRef, SwitchCases, VarId};

/// The top-level discriminant an arm tests.
#[derive(Clone, Debug)]
enum Test {
    Irrefutable,
    Variant(usize),
    Int(i128),
    Bool(bool),
    Str(StringToken),
}

struct Arm<'tcx> {
    test: Test,
    bindings: Vec<(VarId, PatVarRef)>,
    body: Expr<'tcx>,
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    pub(super) fn infer_match(
        &mut self,
        scrut: &surface::Expr,
        arms: &[(surface::Pattern, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let scrut = self.infer_expr(scrut);
        let scrut_ty = self.infer.shallow_resolve(scrut.ty);
        let result = self.infer.new_hole_ty();

        let mut compiled = Vec::new();
        for (pat, body) in arms {
            let mark = self.vars.mark();
            let mut bindings = Vec::new();
            let test = self.check_pattern(pat, scrut_ty, PatVarRef::default(), &mut bindings);
            if pat.guard.is_some() {
                self.error(span, "match guards are not yet supported");
            }
            let body = self.check_expr(body, result);
            self.vars.restore(mark);
            compiled.push(Arm {
                test,
                bindings,
                body,
            });
        }

        let tree = self.build_tree(&compiled, scrut_ty);
        self.mk_match(Box::new(scrut), tree, result, span)
    }

    fn mk_match(
        &mut self,
        scrut: Box<Expr<'tcx>>,
        tree: DecisionTree<'tcx>,
        ty: Ty<'tcx>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let id = self.fresh_expr_id();
        Expr {
            kind: ExprKind::Match(scrut, tree),
            ty,
            span,
            id,
        }
    }

    /// Type-check a pattern against `ty`, binding variables and returning the
    /// top-level discriminant. `base` is the projection path to the value the
    /// pattern matches.
    fn check_pattern(
        &mut self,
        pat: &surface::Pattern,
        ty: Ty<'tcx>,
        base: PatVarRef,
        bindings: &mut Vec<(VarId, PatVarRef)>,
    ) -> Test {
        match &pat.kind {
            PatternKind::Wildcard => Test::Irrefutable,
            PatternKind::Bind(name) => {
                let var = self.vars.fresh(name, ty, None);
                bindings.push((var, base));
                Test::Irrefutable
            }
            PatternKind::Const(c) => self.check_const_pattern(c, ty),
            PatternKind::Ctor(ctor) => self.check_ctor_pattern(ctor, ty, base, bindings),
        }
    }

    fn check_const_pattern(&mut self, c: &Const, ty: Ty<'tcx>) -> Test {
        match c {
            Const::ConstInt(i) => {
                let hole_ty = self.tcx.mk_int(crate::ty::IntTy::Signed(32));
                let _ = self.infer.unify(ty, hole_ty);
                Test::Int(*i as i128)
            }
            Const::ConstBool(b) => {
                let bool_ty = self.tcx.mk_bool();
                let _ = self.infer.unify(ty, bool_ty);
                Test::Bool(*b)
            }
            Const::ConstString(s) => {
                let str_ty = self.tcx.mk_str();
                let _ = self.infer.unify(ty, str_ty);
                Test::Str(self.strings.allocate(s))
            }
            Const::ConstDouble(_) => {
                self.error(None, "floating-point patterns are not supported");
                Test::Irrefutable
            }
        }
    }

    fn check_ctor_pattern(
        &mut self,
        ctor: &surface::CtorPat,
        ty: Ty<'tcx>,
        base: PatVarRef,
        bindings: &mut Vec<(VarId, PatVarRef)>,
    ) -> Test {
        let ty = self.infer.shallow_resolve(ty);
        let TyKind::Record { path, args, .. } = ty.kind() else {
            self.error(None, format!("cannot match constructor against `{ty:?}`"));
            return Test::Irrefutable;
        };
        // The enum is named by the path's qualifier (`Enum::Variant`); fall back
        // to the scrutinee's record name.
        let enum_name = ctor
            .path
            .segments
            .last()
            .map(String::as_str)
            .unwrap_or(path);
        let Some(record) = self.records.get(enum_name).cloned() else {
            self.error(None, format!("unknown enum `{enum_name}`"));
            return Test::Irrefutable;
        };
        let Some(RecordFields::Variants(variants)) = &record.fields else {
            self.error(None, format!("`{enum_name}` is not an enum"));
            return Test::Irrefutable;
        };
        let Some(vidx) = variants.iter().position(|v| v.name == ctor.path.basename) else {
            self.error(None, format!("no variant `{}`", ctor.path.basename));
            return Test::Irrefutable;
        };
        let payload = variants[vidx].fields.clone();
        let inst = Instantiation::from_pairs(
            record
                .ty_params
                .iter()
                .map(|(_, g)| *g)
                .zip(args.iter().copied()),
        );
        // Bind sub-patterns by projecting into the variant's fields.
        for (i, arg) in ctor.args.iter().enumerate() {
            let Some(decl) = payload.get(i) else { break };
            let field_ty = self.infer.instantiate_ty(*decl, &inst);
            let mut sub_base = base.0.clone();
            sub_base.push(i as u32);
            let sub = surface::Pattern {
                kind: arg.kind.clone(),
                guard: None,
            };
            let sub_test = self.check_pattern(&sub, field_ty, PatVarRef(sub_base), bindings);
            // Nested refutable sub-patterns are not switched on in Phase 1.
            if !matches!(sub_test, Test::Irrefutable) {
                self.error(None, "nested refutable patterns are not yet supported");
            }
        }
        Test::Variant(vidx)
    }

    /// Build a decision tree from the checked arms (first-match order).
    fn build_tree(&mut self, arms: &[Arm<'tcx>], scrut_ty: Ty<'tcx>) -> DecisionTree<'tcx> {
        let Some(first) = arms.first() else {
            return DecisionTree::Uncovered;
        };
        let leaf = |arm: &Arm<'tcx>| DecisionTree::Leaf {
            body: Box::new(arm.body.clone()),
            bindings: arm.bindings.clone(),
        };

        // A leading irrefutable arm matches everything; the rest are unreachable.
        if matches!(first.test, Test::Irrefutable) {
            return leaf(first);
        }

        let default = arms
            .iter()
            .find(|a| matches!(a.test, Test::Irrefutable))
            .map_or(DecisionTree::Uncovered, leaf);
        let scrutinee = PatVarRef::default();

        match &first.test {
            Test::Variant(_) => {
                let n_variants = self.variant_count(scrut_ty);
                let mut subtrees: Vec<DecisionTree<'tcx>> =
                    (0..n_variants).map(|_| default.clone()).collect();
                for arm in arms {
                    if let Test::Variant(v) = arm.test
                        && v < subtrees.len()
                        && matches!(subtrees[v], DecisionTree::Uncovered)
                    {
                        subtrees[v] = leaf(arm);
                    }
                }
                DecisionTree::Switch {
                    scrutinee,
                    cases: SwitchCases::Ctor(subtrees),
                }
            }
            Test::Int(_) => {
                let mut cases = Vec::new();
                for arm in arms {
                    if let Test::Int(i) = arm.test
                        && !cases.iter().any(|(k, _)| *k == i)
                    {
                        cases.push((i, leaf(arm)));
                    }
                }
                DecisionTree::Switch {
                    scrutinee,
                    cases: SwitchCases::Int {
                        cases,
                        default: Box::new(default),
                    },
                }
            }
            Test::Bool(_) => {
                let arm_for = |b: bool| {
                    arms.iter()
                        .find(|a| matches!(a.test, Test::Bool(x) if x == b))
                        .map_or(default.clone(), leaf)
                };
                DecisionTree::Switch {
                    scrutinee,
                    cases: SwitchCases::Bool {
                        if_true: Box::new(arm_for(true)),
                        if_false: Box::new(arm_for(false)),
                    },
                }
            }
            Test::Str(_) => {
                let mut cases = Vec::new();
                for arm in arms {
                    if let Test::Str(s) = arm.test {
                        cases.push((s, leaf(arm)));
                    }
                }
                DecisionTree::Switch {
                    scrutinee,
                    cases: SwitchCases::String {
                        cases,
                        default: Box::new(default),
                    },
                }
            }
            Test::Irrefutable => unreachable!("handled above"),
        }
    }

    fn variant_count(&mut self, ty: Ty<'tcx>) -> usize {
        let ty = self.infer.shallow_resolve(ty);
        if let TyKind::Record { path, .. } = ty.kind()
            && let Some(record) = self.records.get(*path)
            && let Some(RecordFields::Variants(vs)) = &record.fields
        {
            return vs.len();
        }
        0
    }

    // ----- zonking -----

    pub(super) fn zonk_tree(&mut self, tree: DecisionTree<'tcx>) -> DecisionTree<'tcx> {
        match tree {
            DecisionTree::Uncovered => DecisionTree::Uncovered,
            DecisionTree::Unreachable => DecisionTree::Unreachable,
            DecisionTree::Leaf { body, bindings } => DecisionTree::Leaf {
                body: Box::new(self.zonk_expr(*body)),
                bindings,
            },
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => DecisionTree::Guard {
                bindings,
                guard: Box::new(self.zonk_expr(*guard)),
                success: Box::new(self.zonk_tree(*success)),
                failure: Box::new(self.zonk_tree(*failure)),
            },
            DecisionTree::Switch { scrutinee, cases } => DecisionTree::Switch {
                scrutinee,
                cases: self.zonk_cases(cases),
            },
        }
    }

    fn zonk_cases(&mut self, cases: SwitchCases<'tcx>) -> SwitchCases<'tcx> {
        match cases {
            SwitchCases::Int { cases, default } => SwitchCases::Int {
                cases: cases
                    .into_iter()
                    .map(|(k, t)| (k, self.zonk_tree(t)))
                    .collect(),
                default: Box::new(self.zonk_tree(*default)),
            },
            SwitchCases::Bool { if_true, if_false } => SwitchCases::Bool {
                if_true: Box::new(self.zonk_tree(*if_true)),
                if_false: Box::new(self.zonk_tree(*if_false)),
            },
            SwitchCases::Ctor(ts) => {
                SwitchCases::Ctor(ts.into_iter().map(|t| self.zonk_tree(t)).collect())
            }
            SwitchCases::String { cases, default } => SwitchCases::String {
                cases: cases
                    .into_iter()
                    .map(|(k, t)| (k, self.zonk_tree(t)))
                    .collect(),
                default: Box::new(self.zonk_tree(*default)),
            },
            SwitchCases::Nullable { non_null, null } => SwitchCases::Nullable {
                non_null: Box::new(self.zonk_tree(*non_null)),
                null: Box::new(self.zonk_tree(*null)),
            },
        }
    }
}
