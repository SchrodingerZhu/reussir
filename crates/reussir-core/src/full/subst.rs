//! Type substitution: instantiate a Semi HIR tree at a ground type assignment.
//!
//! Monomorphization specializes a polymorphic function by mapping each of its
//! generic parameters ([`GenericId`]) to a concrete type and rewriting every
//! type in the body accordingly. Because Full types *are* Semi [`Ty`]s (just
//! ground), substitution re-interns through the same [`TyCtxt`], so the result
//! keeps pointer-identity equality and feeds straight into mangling and
//! lowering.
//!
//! The expression walk mirrors [`crate::semi::check`]'s zonking walk one-for-one
//! — same structure, but resolving against a generic→type map instead of the
//! inference table.

use rustc_hash::FxHashMap;

use crate::semi::hir::{ClosureExpr, DecisionTree, Expr, ExprKind, SwitchCases};
use crate::semi::ty::{GenericId, Ty, TyCtxt, TyKind};

/// A ground assignment for a function's generic parameters.
pub type Subst<'tcx> = FxHashMap<GenericId, Ty<'tcx>>;

/// Substitute generics in a type, re-interning into `tcx`. An empty `subst`
/// short-circuits (the type is already ground).
pub fn subst_ty<'tcx>(tcx: &TyCtxt<'tcx>, ty: Ty<'tcx>, subst: &Subst<'tcx>) -> Ty<'tcx> {
    if subst.is_empty() {
        return ty;
    }
    match *ty.kind() {
        TyKind::Generic(id) => *subst
            .get(&id)
            .unwrap_or_else(|| panic!("subst: generic {id:?} is not bound by this instance")),
        TyKind::Record { def, args, flex } => {
            let args: Vec<Ty<'tcx>> = args.iter().map(|&a| subst_ty(tcx, a, subst)).collect();
            tcx.mk_record(def, &args, flex)
        }
        TyKind::Closure { params, ret } => {
            let params: Vec<Ty<'tcx>> = params.iter().map(|&p| subst_ty(tcx, p, subst)).collect();
            tcx.mk_closure(&params, subst_ty(tcx, ret, subst))
        }
        TyKind::Nullable(inner) => tcx.mk_nullable(subst_ty(tcx, inner, subst)),
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Unit
        | TyKind::Bottom => ty,
        TyKind::Hole(_) => panic!("subst: an inference hole survived zonking"),
    }
}

/// Substitute generics throughout an expression tree.
pub fn subst_expr<'tcx>(tcx: &TyCtxt<'tcx>, e: &Expr<'tcx>, subst: &Subst<'tcx>) -> Expr<'tcx> {
    Expr {
        kind: subst_kind(tcx, &e.kind, subst),
        ty: subst_ty(tcx, e.ty, subst),
        span: e.span,
        id: e.id,
    }
}

fn subst_kind<'tcx>(
    tcx: &TyCtxt<'tcx>,
    kind: &ExprKind<'tcx>,
    subst: &Subst<'tcx>,
) -> ExprKind<'tcx> {
    use ExprKind::*;
    let sb = |e: &Expr<'tcx>| Box::new(subst_expr(tcx, e, subst));
    let sv = |es: &[Expr<'tcx>]| es.iter().map(|e| subst_expr(tcx, e, subst)).collect();
    match kind {
        // Leaves: nothing to substitute.
        GlobalStr(s) => GlobalStr(*s),
        ConstInt(n) => ConstInt(*n),
        ConstFloat(f) => ConstFloat(*f),
        ConstBool(b) => ConstBool(*b),
        Var(v) => Var(*v),
        Poison => Poison,

        Negate(e) => Negate(sb(e)),
        Not(e) => Not(sb(e)),
        Arith(l, op, r) => Arith(sb(l), *op, sb(r)),
        Cmp(l, op, r) => Cmp(sb(l), *op, sb(r)),
        Cast(e, t) => Cast(sb(e), subst_ty(tcx, *t, subst)),
        If(c, t, f) => If(sb(c), sb(t), sb(f)),
        RegionRun(e) => RegionRun(sb(e)),
        Proj(e, idx) => Proj(sb(e), idx.clone()),
        Assign(d, i, s) => Assign(sb(d), *i, sb(s)),
        Let {
            var,
            name,
            span,
            value,
        } => Let {
            var: *var,
            name: *name,
            span: *span,
            value: sb(value),
        },
        Seq(es) => Seq(sv(es)),
        FuncCall {
            target,
            ty_args,
            args,
            regional,
        } => FuncCall {
            target: *target,
            ty_args: ty_args.iter().map(|&t| subst_ty(tcx, t, subst)).collect(),
            args: sv(args),
            regional: *regional,
        },
        CompoundCall {
            target,
            ty_args,
            args,
        } => CompoundCall {
            target: *target,
            ty_args: ty_args.iter().map(|&t| subst_ty(tcx, t, subst)).collect(),
            args: sv(args),
        },
        VariantCall {
            target,
            ty_args,
            variant,
            args,
        } => VariantCall {
            target: *target,
            ty_args: ty_args.iter().map(|&t| subst_ty(tcx, t, subst)).collect(),
            variant: *variant,
            args: sv(args),
        },
        NullableCall(e) => NullableCall(e.as_ref().map(|e| sb(e))),
        ClosureCall { target, args } => ClosureCall {
            target: sb(target),
            args: sv(args),
        },
        Closure(c) => Closure(ClosureExpr {
            captures: c
                .captures
                .iter()
                .map(|(v, t)| (*v, subst_ty(tcx, *t, subst)))
                .collect(),
            params: c
                .params
                .iter()
                .map(|(v, t)| (*v, subst_ty(tcx, *t, subst)))
                .collect(),
            body: sb(&c.body),
        }),
        Match(scrut, tree) => Match(sb(scrut), subst_tree(tcx, tree, subst)),
    }
}

/// Substitute generics throughout a compiled decision tree.
fn subst_tree<'tcx>(
    tcx: &TyCtxt<'tcx>,
    tree: &DecisionTree<'tcx>,
    subst: &Subst<'tcx>,
) -> DecisionTree<'tcx> {
    use DecisionTree::*;
    match tree {
        Uncovered => Uncovered,
        Unreachable => Unreachable,
        Leaf { body, bindings } => Leaf {
            body: Box::new(subst_expr(tcx, body, subst)),
            bindings: bindings.clone(),
        },
        Guard {
            bindings,
            guard,
            success,
            failure,
        } => Guard {
            bindings: bindings.clone(),
            guard: Box::new(subst_expr(tcx, guard, subst)),
            success: Box::new(subst_tree(tcx, success, subst)),
            failure: Box::new(subst_tree(tcx, failure, subst)),
        },
        Switch { scrutinee, cases } => Switch {
            scrutinee: scrutinee.clone(),
            cases: subst_cases(tcx, cases, subst),
        },
    }
}

fn subst_cases<'tcx>(
    tcx: &TyCtxt<'tcx>,
    cases: &SwitchCases<'tcx>,
    subst: &Subst<'tcx>,
) -> SwitchCases<'tcx> {
    use SwitchCases::*;
    let st = |t: &DecisionTree<'tcx>| Box::new(subst_tree(tcx, t, subst));
    match cases {
        Int { cases, default } => Int {
            cases: cases
                .iter()
                .map(|(n, t)| (*n, subst_tree(tcx, t, subst)))
                .collect(),
            default: st(default),
        },
        Bool { if_true, if_false } => Bool {
            if_true: st(if_true),
            if_false: st(if_false),
        },
        Ctor(arms) => Ctor(arms.iter().map(|t| subst_tree(tcx, t, subst)).collect()),
        String { cases, default } => String {
            cases: cases
                .iter()
                .map(|(s, t)| (*s, subst_tree(tcx, t, subst)))
                .collect(),
            default: st(default),
        },
        Nullable { non_null, null } => Nullable {
            non_null: st(non_null),
            null: st(null),
        },
    }
}
