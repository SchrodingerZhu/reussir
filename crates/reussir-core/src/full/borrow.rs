//! Borrowed-parameter inference over the monomorphized MIR.
//!
//! A parameter is **borrowed** when the callee is a *pure reader*: every
//! occurrence of the parameter variable in the body is a match scrutinee or
//! the root of a projection chain — positions the ownership analysis treats
//! as borrows already — **and the function allocates nothing reference
//! counted**. Such a function (an `is_red`/`fold`-style predicate or
//! accessor) never consumes, returns, stores, or passes the value on, so
//! the owned convention only produces a caller-side `dup` paired with a
//! callee-side `drop`. Marking the parameter borrowed removes both: the
//! caller passes the value without a `dup`, keeps ownership, and settles it
//! after the call when the call was its last use; the callee never releases
//! it.
//!
//! The allocates-nothing restriction is what keeps this compatible with
//! reuse: an update-shaped function (rbtree's `ins`, nbe's `quote`) also
//! only *reads* its parameter syntactically, but consuming it is what feeds
//! every reuse token in its constructor chain — borrowing there trades
//! in-place spine updates for full path copies. A function that allocates
//! nothing has no reuse to lose, and the caller-side settle-dec it produces
//! is itself a reuse token in the caller.
//!
//! The verdict is a pure function of the function's own body — deliberately
//! **not** interprocedural — so every codegen unit, and every repl
//! re-monomorphization of the same instance, computes the same convention
//! for the same symbol. Passing a value to another function's parameter
//! (borrowed or not) therefore counts as a consuming use here; the fixpoint
//! refinement is future work.
//!
//! Exclusions: trampoline targets keep the owned convention (their C callers
//! transfer ownership) and regional functions are skipped wholesale (region
//! bookkeeping has its own discipline).

use rustc_hash::FxHashSet;

use crate::semi::hir::VarId;
use crate::semi::ty::TyCtxt;

use super::mir::{DecisionTree, Expr, ExprKind, Program, SwitchCases};
use super::ownership::{RecordTable, Rr};

/// Infer and record the borrowed convention for every eligible parameter of
/// every function in `program`. Runs once at the end of monomorphization.
pub fn infer_borrowed_params<'tcx>(tcx: &TyCtxt<'tcx>, program: &mut Program<'tcx>) {
    let excluded: FxHashSet<_> = program.trampolines.iter().map(|t| t.target).collect();
    let table = RecordTable::from_records(&program.records);
    let rr = Rr::new(tcx, &table);
    for func in &mut program.functions {
        if func.is_regional || excluded.contains(&func.symbol) {
            continue;
        }
        let Some(body) = func.body else { continue };
        if allocates(body) || calls(body, func.symbol) {
            continue;
        }
        for param in &mut func.params {
            // Only resource-relevant parameters carry reference counts; the
            // convention is meaningless (and noisy) for scalars.
            param.borrowed = rr.is_rr(param.ty) && only_read(body, param.var);
        }
    }
}

/// Whether the body contains a direct call to `symbol`. A self-recursive
/// reader is excluded from borrowing: with an owned parameter its walk
/// releases the structure incrementally (one node per level, a loop after
/// TRMC/tail lowering), while a borrowed parameter defers the whole release
/// to the caller's settle-dec — a single *recursive* drop whose depth is the
/// structure's depth, a stack overflow on long lists and app-chains.
fn calls(e: &Expr<'_>, symbol: super::mir::Symbol) -> bool {
    let mut found = false;
    walk(e, &mut |x| {
        found |= matches!(x.kind, ExprKind::Call { callee, .. } if callee == symbol);
    });
    found
}

/// Whether the body can allocate a reference-counted box: any constructor,
/// closure, or string literal. (Value-capability records still count — a
/// value record materialized here may be boxed by the caller's create chain
/// after inlining; staying conservative keeps every reuse token intact.)
fn allocates(e: &Expr<'_>) -> bool {
    let mut found = false;
    walk(e, &mut |x| {
        found |= matches!(
            x.kind,
            ExprKind::Ctor { .. }
                | ExprKind::Variant { .. }
                | ExprKind::Closure(_)
                | ExprKind::GlobalStr(_)
        );
    });
    found
}

/// Generic pre-order walk over an expression tree, including match arms.
fn walk(e: &Expr<'_>, f: &mut impl FnMut(&Expr<'_>)) {
    f(e);
    match e.kind {
        ExprKind::Var(_)
        | ExprKind::GlobalStr(_)
        | ExprKind::ConstChar(_)
        | ExprKind::ConstInt(_)
        | ExprKind::ConstFloat(_)
        | ExprKind::ConstBool(_)
        | ExprKind::Poison => {}
        ExprKind::Negate(a) | ExprKind::Not(a) | ExprKind::Cast(a, _) => walk(a, f),
        ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
            walk(l, f);
            walk(r, f);
        }
        ExprKind::Let { value, .. } => walk(value, f),
        ExprKind::Seq(es) => es.iter().for_each(|s| walk(s, f)),
        ExprKind::Call { args, .. }
        | ExprKind::Ctor { args, .. }
        | ExprKind::Variant { args, .. }
        | ExprKind::Intrinsic { args, .. } => args.iter().for_each(|a| walk(a, f)),
        ExprKind::NullableCall(opt) => {
            if let Some(a) = opt {
                walk(a, f)
            }
        }
        ExprKind::If(c, t, e2) => {
            walk(c, f);
            walk(t, f);
            walk(e2, f);
        }
        ExprKind::Match(scrut, ref tree) => {
            walk(scrut, f);
            walk_tree(tree, f);
        }
        ExprKind::Proj(base, _) => walk(base, f),
        ExprKind::Assign(dst, _, src) => {
            walk(dst, f);
            walk(src, f);
        }
        ExprKind::RegionRun(inner) => walk(inner, f),
        ExprKind::Closure(c) => walk(c.body, f),
        ExprKind::ClosureCall { target, args } => {
            walk(target, f);
            args.iter().for_each(|a| walk(a, f));
        }
    }
}

fn walk_tree(tree: &DecisionTree<'_>, f: &mut impl FnMut(&Expr<'_>)) {
    match *tree {
        DecisionTree::Uncovered | DecisionTree::Unreachable => {}
        DecisionTree::Leaf { body, .. } => walk(body, f),
        DecisionTree::Guard {
            guard,
            ref success,
            ref failure,
            ..
        } => {
            walk(guard, f);
            walk_tree(success, f);
            walk_tree(failure, f);
        }
        DecisionTree::Switch { ref cases, .. } => match *cases {
            SwitchCases::Int { cases, default } => {
                cases.iter().for_each(|(_, t)| walk_tree(t, f));
                walk_tree(default, f);
            }
            SwitchCases::Char { cases, default } => {
                cases.iter().for_each(|(_, t)| walk_tree(t, f));
                walk_tree(default, f);
            }
            SwitchCases::String { cases, default } => {
                cases.iter().for_each(|(_, t)| walk_tree(t, f));
                walk_tree(default, f);
            }
            SwitchCases::Bool { if_true, if_false } => {
                walk_tree(if_true, f);
                walk_tree(if_false, f);
            }
            SwitchCases::Nullable { non_null, null } => {
                walk_tree(non_null, f);
                walk_tree(null, f);
            }
            SwitchCases::Ctor(arms) => arms.iter().for_each(|t| walk_tree(t, f)),
        },
    }
}

/// Whether every occurrence of `x` in `e` is a read: a match scrutinee or a
/// projection-chain root. Any other occurrence (call/ctor argument, closure
/// capture, result position, assignment, …) makes the parameter owned.
fn only_read(e: &Expr<'_>, x: VarId) -> bool {
    match e.kind {
        ExprKind::Var(v) => v != x,
        ExprKind::GlobalStr(_)
        | ExprKind::ConstChar(_)
        | ExprKind::ConstInt(_)
        | ExprKind::ConstFloat(_)
        | ExprKind::ConstBool(_)
        | ExprKind::Poison => true,
        ExprKind::Negate(a) | ExprKind::Not(a) | ExprKind::Cast(a, _) => only_read(a, x),
        ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
            only_read(l, x) && only_read(r, x)
        }
        ExprKind::Let { value, .. } => only_read(value, x),
        ExprKind::Seq(es) => es.iter().all(|s| only_read(s, x)),
        ExprKind::Call { args, .. }
        | ExprKind::Ctor { args, .. }
        | ExprKind::Variant { args, .. }
        | ExprKind::Intrinsic { args, .. } => args.iter().all(|a| only_read(a, x)),
        ExprKind::NullableCall(opt) => opt.as_ref().is_none_or(|a| only_read(a, x)),
        ExprKind::If(c, t, e2) => only_read(c, x) && only_read(t, x) && only_read(e2, x),
        // The scrutinee position itself is a borrow; look through it when it
        // is the plain variable, but its subexpressions (if any) are ordinary
        // uses. Arm bodies and guards are walked for other occurrences.
        ExprKind::Match(scrut, ref tree) => {
            let scrut_ok = matches!(scrut.kind, ExprKind::Var(v) if v == x)
                || only_read(scrut, x);
            scrut_ok && tree_only_read(tree, x)
        }
        // A projection reads its base: accept `x` as the chain root.
        ExprKind::Proj(base, _) => proj_base_only_read(base, x),
        ExprKind::Assign(dst, _, src) => only_read(dst, x) && only_read(src, x),
        ExprKind::RegionRun(inner) => only_read(inner, x),
        ExprKind::Closure(c) => {
            // A capture takes ownership of the variable by name, not through
            // an `Expr`; the body of the closure references the capture, not
            // the original parameter.
            c.captures.iter().all(|&(v, _)| v != x) && only_read(c.body, x)
        }
        ExprKind::ClosureCall { target, args } => {
            only_read(target, x) && args.iter().all(|a| only_read(a, x))
        }
    }
}

/// A projection chain (`Proj(Proj(..(root)..))`) whose root is `x` is a read;
/// any other base shape is walked normally.
fn proj_base_only_read(base: &Expr<'_>, x: VarId) -> bool {
    match base.kind {
        // The chain root is read in place — `x` here is fine.
        ExprKind::Var(_) => true,
        ExprKind::Proj(inner, _) => proj_base_only_read(inner, x),
        _ => only_read(base, x),
    }
}

fn tree_only_read(tree: &DecisionTree<'_>, x: VarId) -> bool {
    match *tree {
        DecisionTree::Uncovered | DecisionTree::Unreachable => true,
        DecisionTree::Leaf { body, .. } => only_read(body, x),
        DecisionTree::Guard {
            guard,
            ref success,
            ref failure,
            ..
        } => only_read(guard, x) && tree_only_read(success, x) && tree_only_read(failure, x),
        DecisionTree::Switch { ref cases, .. } => match *cases {
            SwitchCases::Int { cases, default } => {
                cases.iter().all(|(_, t)| tree_only_read(t, x)) && tree_only_read(default, x)
            }
            SwitchCases::Char { cases, default } => {
                cases.iter().all(|(_, t)| tree_only_read(t, x)) && tree_only_read(default, x)
            }
            SwitchCases::String { cases, default } => {
                cases.iter().all(|(_, t)| tree_only_read(t, x)) && tree_only_read(default, x)
            }
            SwitchCases::Bool { if_true, if_false } => {
                tree_only_read(if_true, x) && tree_only_read(if_false, x)
            }
            SwitchCases::Nullable { non_null, null } => {
                tree_only_read(non_null, x) && tree_only_read(null, x)
            }
            SwitchCases::Ctor(arms) => arms.iter().all(|t| tree_only_read(t, x)),
        },
    }
}
