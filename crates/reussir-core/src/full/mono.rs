//! Monomorphization: the Semi → Full driver.
//!
//! Elaboration leaves functions polymorphic. Monomorphization specializes them:
//! starting from a set of **roots**, it instantiates each reachable function at
//! every concrete type-argument tuple it is used with, substitutes the generics
//! away (see [`crate::full::subst`]), and assigns each instance its v0 symbol
//! (see [`crate::full::mangle`]). The result is a flat list of ground functions
//! plus the set of ground record instances their layouts require.
//!
//! # Roots
//!
//! The roots are **every non-generic function** plus **every trampoline target**
//! (an exported C-ABI alias instantiated at concrete arguments). A non-generic
//! function is emitted even if nothing calls it — whole-program dead-code
//! elimination is the backend's job, not the frontend's — which matches the
//! observable behavior of the existing frontend. Generic functions are emitted
//! once per distinct instantiation discovered from a reachable body.
//!
//! # Worklist
//!
//! An [`Instance`] is `(DefId, &[Ty])`; its interned type-argument slice compares
//! by element pointer-identity, so structurally-equal instantiations collapse to
//! one worklist entry. We pop an instance, substitute its body to ground form,
//! then scan that ground body (and signature) for further instances — function
//! calls seed new function instances; every record type seeds a record instance.

use std::collections::VecDeque;

use reussir_syntax::kind::TokenKey;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::full::mangle::Mangler;
use crate::full::subst::{Subst, subst_expr, subst_ty};
use crate::semi::ctxt::Elaborator;
use crate::semi::hir::{DecisionTree, Expr, ExprKind, Function, SwitchCases, VarId};
use crate::semi::ty::{DefId, Ty, TyCtxt, TyKind};
use crate::surface::Visibility;

/// A ground instantiation of a top-level item: its [`DefId`] applied to an
/// interned tuple of concrete type arguments (empty for a non-generic item).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: DefId,
    pub ty_args: &'tcx [Ty<'tcx>],
}

/// A monomorphized function: a ground signature and body with its v0 symbol.
#[derive(Clone, Debug)]
pub struct FullFunction<'tcx> {
    pub instance: Instance<'tcx>,
    /// The v0 linker symbol for this instance.
    pub symbol: String,
    pub visibility: Visibility,
    pub is_regional: bool,
    /// `(source name, variable, ground type)` in declaration order.
    pub params: Vec<(TokenKey, VarId, Ty<'tcx>)>,
    pub return_ty: Ty<'tcx>,
    /// `None` for a declared-but-undefined function.
    pub body: Option<Expr<'tcx>>,
}

/// An exported C-ABI trampoline aliasing a concrete function instance.
#[derive(Clone, Debug)]
pub struct Trampoline {
    /// The exported C symbol.
    pub export: String,
    /// The C ABI string (e.g. `"C"`).
    pub abi: String,
    /// The v0 symbol of the aliased function instance.
    pub target_symbol: String,
}

/// The whole monomorphized program: ground functions, the ground record
/// instances their layouts need, and the exported trampolines. Functions and
/// record instances are sorted by symbol for deterministic output.
#[derive(Debug)]
pub struct FullProgram<'tcx> {
    pub functions: Vec<FullFunction<'tcx>>,
    pub records: Vec<Instance<'tcx>>,
    pub trampolines: Vec<Trampoline>,
}

/// Monomorphize an elaborated program into its ground Full form.
pub fn monomorphize<'a, 'tcx>(elab: &Elaborator<'a, 'tcx>) -> FullProgram<'tcx> {
    let tcx: &'a TyCtxt<'tcx> = elab.tcx;
    let by_def: FxHashMap<DefId, &Function<'tcx>> =
        elab.elaborated.iter().map(|f| (f.def, f)).collect();
    let mangler = Mangler::new(&elab.defs, elab.resolver);

    let mut driver = Driver {
        tcx,
        queue: VecDeque::new(),
        seen: FxHashSet::default(),
        records: FxHashSet::default(),
    };

    // Seed roots: every non-generic function, then every trampoline target.
    for f in &elab.elaborated {
        if f.generics.is_empty() {
            driver.enqueue(f.def, tcx.intern_tys(&[]));
        }
    }
    for t in &elab.trampolines {
        driver.enqueue(t.target, tcx.intern_tys(&t.ty_args));
    }

    let mut functions = Vec::new();
    while let Some(inst) = driver.queue.pop_front() {
        let Some(func) = by_def.get(&inst.def) else {
            // No body/prototype recorded (e.g. an item that failed to elaborate);
            // nothing to instantiate.
            continue;
        };

        // Bind this instance's generics, then specialize the signature and body.
        let mut subst = Subst::default();
        for ((_, gid), &ty) in func.generics.iter().zip(inst.ty_args.iter()) {
            subst.insert(*gid, ty);
        }
        let params: Vec<(TokenKey, VarId, Ty<'tcx>)> = func
            .params
            .iter()
            .map(|(name, var, ty)| (*name, *var, subst_ty(tcx, *ty, &subst)))
            .collect();
        let return_ty = subst_ty(tcx, func.return_ty, &subst);
        let body = func.body.as_ref().map(|b| subst_expr(tcx, b, &subst));

        // Discover further instances from the ground signature and body.
        for (_, _, ty) in &params {
            driver.scan_ty(*ty);
        }
        driver.scan_ty(return_ty);
        if let Some(body) = &body {
            driver.scan_expr(body);
        }

        functions.push(FullFunction {
            instance: inst,
            symbol: mangler.mangle_instance(inst.def, inst.ty_args),
            visibility: func.visibility,
            is_regional: func.is_regional,
            params,
            return_ty,
            body,
        });
    }

    functions.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    let mut records: Vec<Instance<'tcx>> = driver.records.into_iter().collect();
    records.sort_by_cached_key(|r| mangler.mangle_instance(r.def, r.ty_args));

    let trampolines = elab
        .trampolines
        .iter()
        .map(|t| Trampoline {
            export: t.name.clone(),
            abi: t.abi.clone(),
            target_symbol: mangler.mangle_instance(t.target, tcx.intern_tys(&t.ty_args)),
        })
        .collect();

    FullProgram {
        functions,
        records,
        trampolines,
    }
}

/// The maximum type-argument nesting depth monomorphization will instantiate
/// before treating the chain as non-terminating — in the spirit of rustc's
/// `recursion_limit`. Deep but finite generic nesting stays well under it; the
/// unbounded type growth of polymorphic recursion blows straight past it.
///
/// TODO: surface as a proper diagnostic (naming the offending instance) once
/// mono gains a report sink, instead of panicking.
const RECURSION_LIMIT: usize = 128;

/// The structural nesting depth of a ground type: `1` for a scalar, one more
/// than its deepest argument for a constructor. Polymorphic recursion grows this
/// without bound (each `f<T>` → `f<Wrap<T>>` step adds a level), which is what
/// the recursion-limit guard in [`Driver::enqueue`] watches.
fn ty_depth(ty: Ty<'_>) -> usize {
    match *ty.kind() {
        TyKind::Nullable(inner) => 1 + ty_depth(inner),
        TyKind::Record { args, .. } => 1 + args.iter().map(|&a| ty_depth(a)).max().unwrap_or(0),
        TyKind::Closure { params, ret } => {
            1 + params
                .iter()
                .copied()
                .chain(std::iter::once(ret))
                .map(ty_depth)
                .max()
                .unwrap_or(0)
        }
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Unit
        | TyKind::Bottom
        | TyKind::Generic(_)
        | TyKind::Hole(_) => 1,
    }
}

/// Mutable worklist state shared across the instantiation walk.
struct Driver<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    queue: VecDeque<Instance<'tcx>>,
    /// Function instances already enqueued (so each is emitted once).
    seen: FxHashSet<Instance<'tcx>>,
    /// Record instances whose layout is needed.
    records: FxHashSet<Instance<'tcx>>,
}

impl<'a, 'tcx> Driver<'a, 'tcx> {
    /// Enqueue a function instance if it has not been seen.
    fn enqueue(&mut self, def: DefId, ty_args: &'tcx [Ty<'tcx>]) {
        let inst = Instance { def, ty_args };
        if self.seen.insert(inst) {
            // Bound the queue by type-argument depth. The queue is the only thing
            // that grows, and polymorphic recursion makes each instance one level
            // deeper than the last, so an unbounded chain trips this rather than
            // looping forever. Ground types of depth <= the limit over a finite
            // set of `DefId`s are a finite set, so this guarantees termination.
            let depth = ty_args.iter().map(|&t| ty_depth(t)).max().unwrap_or(0);
            assert!(
                depth <= RECURSION_LIMIT,
                "monomorphize: type-argument nesting depth {depth} exceeds the \
                 recursion limit ({RECURSION_LIMIT}); this is almost certainly \
                 unbounded instantiation from polymorphic recursion"
            );
            self.queue.push_back(inst);
        }
    }

    /// Record a ground record instance and recurse into its arguments.
    fn record(&mut self, def: DefId, args: &'tcx [Ty<'tcx>]) {
        self.records.insert(Instance { def, ty_args: args });
        for &arg in args {
            self.scan_ty(arg);
        }
    }

    /// Collect every record instance reachable from a ground type.
    fn scan_ty(&mut self, ty: Ty<'tcx>) {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => self.record(def, args),
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.scan_ty(p);
                }
                self.scan_ty(ret);
            }
            TyKind::Nullable(inner) => self.scan_ty(inner),
            TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Str
            | TyKind::Unit
            | TyKind::Bottom => {}
            TyKind::Generic(_) | TyKind::Hole(_) => {
                panic!("monomorphize: non-ground type {ty:?} reached instance scan")
            }
        }
    }

    /// Collect instances from a ground expression: the node's type, any callee
    /// instance, and recursively all children (including decision-tree arms).
    fn scan_expr(&mut self, e: &Expr<'tcx>) {
        use ExprKind::*;
        self.scan_ty(e.ty);
        if let FuncCall {
            target, ty_args, ..
        } = &e.kind
        {
            self.enqueue(*target, self.tcx.intern_tys(ty_args));
            for &t in ty_args {
                self.scan_ty(t);
            }
        }
        for child in children(e) {
            self.scan_expr(child);
        }
        if let Match(_, tree) = &e.kind {
            self.scan_tree(tree);
        }
    }

    /// Collect instances from a decision tree's arm bodies, guards, and subtrees.
    fn scan_tree(&mut self, tree: &DecisionTree<'tcx>) {
        use DecisionTree::*;
        match tree {
            Uncovered | Unreachable => {}
            Leaf { body, .. } => self.scan_expr(body),
            Guard {
                guard,
                success,
                failure,
                ..
            } => {
                self.scan_expr(guard);
                self.scan_tree(success);
                self.scan_tree(failure);
            }
            Switch { cases, .. } => match cases {
                SwitchCases::Int { cases, default } => {
                    for (_, t) in cases {
                        self.scan_tree(t);
                    }
                    self.scan_tree(default);
                }
                SwitchCases::Bool { if_true, if_false } => {
                    self.scan_tree(if_true);
                    self.scan_tree(if_false);
                }
                SwitchCases::Ctor(arms) => {
                    for t in arms {
                        self.scan_tree(t);
                    }
                }
                SwitchCases::String { cases, default } => {
                    for (_, t) in cases {
                        self.scan_tree(t);
                    }
                    self.scan_tree(default);
                }
                SwitchCases::Nullable { non_null, null } => {
                    self.scan_tree(non_null);
                    self.scan_tree(null);
                }
            },
        }
    }
}

/// The immediate sub-expressions of `e` (its scrutinee/operands/arguments), used
/// to drive the instance scan. Decision-tree arms are walked separately.
fn children<'a, 'tcx>(e: &'a Expr<'tcx>) -> Vec<&'a Expr<'tcx>> {
    use ExprKind::*;
    match &e.kind {
        GlobalStr(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_) | Poison => Vec::new(),
        Negate(e) | Not(e) | Cast(e, _) | RegionRun(e) | Proj(e, _) => vec![e],
        Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => vec![l, r],
        If(c, t, f) => vec![c, t, f],
        Let { value, .. } => vec![value],
        Seq(es) => es.iter().collect(),
        FuncCall { args, .. } | CompoundCall { args, .. } | VariantCall { args, .. } => {
            args.iter().collect()
        }
        NullableCall(e) => e.iter().map(|e| &**e).collect(),
        ClosureCall { target, args } => std::iter::once(&**target).chain(args.iter()).collect(),
        Closure(c) => vec![&c.body],
        Match(scrut, _) => vec![scrut],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// Elaborate `source`, monomorphize it, and hand the program to `f`.
    fn with_full<R>(source: &str, f: impl FnOnce(&FullProgram<'_>) -> R) -> R {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let full = monomorphize(&elab);
            f(&full)
        })
    }

    fn symbols(full: &FullProgram<'_>) -> Vec<String> {
        full.functions.iter().map(|f| f.symbol.clone()).collect()
    }

    #[test]
    fn emits_every_non_generic_function_even_if_uncalled() {
        // `never_called` is dead, yet still emitted (DCE is the backend's job).
        let src = r#"
            pub fn used(n: i32) -> i32 { n }
            fn never_called(n: i32) -> i32 { n }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RC4used".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC12never_called".to_string()), "{syms:?}");
        });
    }

    #[test]
    fn instantiates_a_generic_at_each_use() {
        // `id<T>` is generic, so it is emitted once per distinct instantiation
        // discovered from the (non-generic, hence root) callers.
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
            pub fn use_bool(b: bool) -> bool { id(b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            // The generic instances mangle with their type argument.
            assert!(syms.contains(&"_RIC2idlE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC2idbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC7use_i32".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC8use_bool".to_string()), "{syms:?}");
            // No leftover polymorphic `id` and no duplicate instances.
            let mut sorted = syms.clone();
            sorted.dedup();
            assert_eq!(sorted.len(), syms.len(), "duplicate instances: {syms:?}");
        });
    }

    #[test]
    fn bodies_are_ground_after_monomorphization() {
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
        "#;
        with_full(src, |full| {
            for func in &full.functions {
                if let Some(body) = &func.body {
                    assert_ground(body);
                }
                for (_, _, ty) in &func.params {
                    assert!(is_ground(*ty), "non-ground param in {}", func.symbol);
                }
            }
        });
    }

    fn is_ground(ty: Ty<'_>) -> bool {
        match *ty.kind() {
            TyKind::Generic(_) | TyKind::Hole(_) => false,
            TyKind::Record { args, .. } => args.iter().all(|&a| is_ground(a)),
            TyKind::Closure { params, ret } => {
                params.iter().all(|&p| is_ground(p)) && is_ground(ret)
            }
            TyKind::Nullable(inner) => is_ground(inner),
            _ => true,
        }
    }

    fn assert_ground(e: &Expr<'_>) {
        assert!(is_ground(e.ty), "non-ground expr type: {:?}", e.ty);
        for child in children(e) {
            assert_ground(child);
        }
    }

    #[test]
    fn monomorphic_recursion_terminates() {
        // Self-recursion at a *fixed* type is a single instance — the depth guard
        // must not mistake ordinary recursion for a runaway.
        let src = r#"
            fn rec(n: i32) -> i32 { rec(n) }
            fn main(n: i32) -> i32 { rec(n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RC3rec".to_string()), "{syms:?}");
        });
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_polymorphic_recursion() {
        // `rec<T>` calls itself at `Wrap<T>`, so each instance is one level deeper
        // than the last: `rec<i32>`, `rec<Wrap<i32>>`, `rec<Wrap<Wrap<i32>>>`, …
        // This never converges, so mono must abort at the recursion limit instead
        // of looping forever.
        let src = r#"
            struct Wrap<T> { value: T }
            fn rec<T>(x: T) -> i32 { rec(Wrap { value: x }) }
            fn main(n: i32) -> i32 { rec(n) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    fn multiple_type_params_instantiate_per_ordering() {
        // `pick<A, B>` is generic in two parameters; the type arguments mangle in
        // declaration order, so distinct orderings are distinct instances.
        let src = r#"
            fn pick<A, B>(a: A, b: B) -> A { a }
            fn use_ib(n: i32, b: bool) -> i32 { pick(n, b) }
            fn use_bi(b: bool, n: i32) -> bool { pick(b, n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4picklbE".to_string()), "{syms:?}"); // pick<i32, bool>
            assert!(syms.contains(&"_RIC4pickblE".to_string()), "{syms:?}"); // pick<bool, i32>
            // Exactly two `pick` instances — the two orderings, no duplicates.
            let picks = syms.iter().filter(|s| s.contains("pick")).count();
            assert_eq!(picks, 2, "{syms:?}");
            for func in &full.functions {
                if let Some(body) = &func.body {
                    assert_ground(body);
                }
            }
        });
    }

    #[test]
    fn permuting_type_params_terminates() {
        // `swap<A, B>` recurses as `swap<B, A>`: the arguments permute but the
        // reachable set is finite ({<i32, bool>, <bool, i32>}), so mono converges
        // without tripping the depth guard.
        let src = r#"
            fn swap<A, B>(a: A, b: B) -> i32 { swap(b, a) }
            fn main(n: i32, b: bool) -> i32 { swap(n, b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4swaplbE".to_string()), "{syms:?}"); // swap<i32, bool>
            assert!(syms.contains(&"_RIC4swapblE".to_string()), "{syms:?}"); // swap<bool, i32>
            let swaps = syms.iter().filter(|s| s.contains("swap")).count();
            assert_eq!(swaps, 2, "{syms:?}");
        });
    }

    #[test]
    fn mutual_recursion_at_fixed_type_terminates() {
        // `ping`/`pong` call each other at the *same* type argument, so the cycle
        // closes after one instance of each.
        let src = r#"
            fn ping<T>(x: T) -> i32 { pong(x) }
            fn pong<T>(x: T) -> i32 { ping(x) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4pinglE".to_string()), "{syms:?}"); // ping<i32>
            assert!(syms.contains(&"_RIC4ponglE".to_string()), "{syms:?}"); // pong<i32>
        });
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_polymorphic_recursion_in_one_of_several_params() {
        // Only `A` grows; `B` stays fixed. The depth guard takes the max over all
        // type arguments, so a runaway in any single parameter is still caught.
        let src = r#"
            struct Wrap<T> { value: T }
            fn rec2<A, B>(a: A, b: B) -> i32 { rec2(Wrap { value: a }, b) }
            fn main(n: i32, b: bool) -> i32 { rec2(n, b) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_mutual_polymorphic_recursion() {
        // The growth is split across the cycle: each hop wraps the argument once,
        // so `ping`/`pong` together drive unbounded instantiation.
        let src = r#"
            struct Wrap<T> { value: T }
            fn ping<T>(x: T) -> i32 { pong(Wrap { value: x }) }
            fn pong<T>(x: T) -> i32 { ping(Wrap { value: x }) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |_| {});
    }
}
