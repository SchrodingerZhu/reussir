//! Monomorphization: the Semi → Full lowering driver.
//!
//! Elaboration leaves functions polymorphic. Monomorphization specializes them
//! and **lowers** them into the [`crate::full::mir`]: starting from a set of
//! **roots**, it instantiates each reachable function at every concrete
//! type-argument tuple it is used with, grounds its types (see
//! [`crate::full::subst`]), resolves every callee to its v0 [`mir::Symbol`] (see
//! [`crate::full::mangle`]), and **erases the generic apparatus** — the MIR has
//! no type arguments on any node and dispatches purely by interned symbol.
//!
//! # Roots
//!
//! Every non-generic function (emitted even if uncalled — whole-program DCE is
//! the backend's job) plus every trampoline target. Generic functions are
//! emitted once per distinct instantiation discovered from a reachable body.
//!
//! # Worklist
//!
//! An [`Instance`] is `(DefId, &[Ty])`; its interned type-argument slice compares
//! by element pointer-identity, so structurally-equal instantiations collapse to
//! one entry. Popping an instance lowers its body to MIR; each `FuncCall` found
//! there resolves to a callee symbol *and* enqueues that callee instance, so
//! discovery and symbol resolution happen in the same pass.
//!
//! # Regional check at the call boundary
//!
//! A generic used at a `[flex]` position (e.g. `bar: [flex] T`) requires its
//! instantiating type to be a *regional* record — only regional records can be
//! flex. The `[flex]` coloring itself is dropped on a bare generic (as in the
//! reference); the elaborator records the implied requirement in
//! `Function::regional_generics`, and mono verifies it here when an instance is
//! popped, reporting any non-regional (value/shared) instantiation. The
//! *flexivity* rules (flex capture/return/assign) are a separate concern enforced
//! at the Semi stage on concrete records, not here.

use std::collections::VecDeque;

use lasso::Rodeo;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::full::mangle::Mangler;
use crate::full::mir;
use crate::full::subst::{Subst, subst_ty};
use crate::semi::ctxt::{Elaborator, Report, Severity};
use crate::semi::hir::{self, DecisionTree, Expr, ExprKind, Function, SwitchCases};
use crate::semi::ty::{Capability, DefId, Ty, TyCtxt, TyKind};
use crate::surface::Span;

/// A ground instantiation of a top-level item: its [`DefId`] applied to an
/// interned tuple of concrete type arguments (empty for a non-generic item).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: DefId,
    pub ty_args: &'tcx [Ty<'tcx>],
}

/// Monomorphize an elaborated program into its ground Full MIR, alongside any
/// diagnostics raised by the call-boundary regional check.
pub fn monomorphize<'a, 'tcx>(elab: &Elaborator<'a, 'tcx>) -> (mir::Program<'tcx>, Vec<Report>) {
    let tcx: &'a TyCtxt<'tcx> = elab.tcx;
    let by_def: FxHashMap<DefId, &Function<'tcx>> =
        elab.elaborated.iter().map(|f| (f.def, f)).collect();

    let mut driver = Driver {
        tcx,
        mangler: Mangler::new(&elab.defs, elab.resolver),
        symbols: Rodeo::default(),
        queue: VecDeque::new(),
        seen: FxHashSet::default(),
        records: FxHashSet::default(),
        reports: Vec::new(),
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
            // No body/prototype recorded (e.g. an item that failed to elaborate).
            continue;
        };

        // Bind this instance's generics, then ground the signature and lower the
        // body into the MIR.
        let mut subst = Subst::default();
        for ((_, gid), &ty) in func.generics.iter().zip(inst.ty_args.iter()) {
            subst.insert(*gid, ty);
            // Call-boundary check: a generic used at a `[flex]` position must be
            // instantiated with a regional record (only regional records can be
            // flex). Value/shared records and scalars are rejected.
            if func.regional_generics.contains(gid) && !is_regional_arg(ty) {
                driver.error(
                    func.span,
                    "a `[flex]` type parameter requires a regional record, but this \
                     instantiation supplied a non-regional (value/shared) type"
                        .to_string(),
                );
            }
        }
        let params: Vec<mir::Param<'tcx>> = func
            .params
            .iter()
            .map(|(name, var, ty)| {
                let ty = subst_ty(tcx, *ty, &subst);
                driver.note_records(ty);
                mir::Param {
                    name: *name,
                    var: *var,
                    ty,
                }
            })
            .collect();
        let return_ty = subst_ty(tcx, func.return_ty, &subst);
        driver.note_records(return_ty);
        let body = func.body.as_ref().map(|b| driver.lower_ref(b, &subst));

        let symbol = driver.symbol_of(inst.def, inst.ty_args);
        functions.push(mir::Function {
            symbol,
            visibility: func.visibility,
            is_regional: func.is_regional,
            params,
            return_ty,
            body,
        });
    }

    // Resolve the discovered record instances to symbols. Collect the keys first
    // so the interner can be borrowed mutably while we map them.
    let record_keys: Vec<(DefId, &'tcx [Ty<'tcx>])> = driver.records.iter().copied().collect();
    let mut records: Vec<mir::RecordInstance<'tcx>> = record_keys
        .into_iter()
        .map(|(def, args)| mir::RecordInstance {
            symbol: driver.symbol_of(def, args),
            // Layout is capability-independent, so canonicalize the coloring.
            ty: tcx.mk_record(def, args, Capability::Irrelevant),
        })
        .collect();

    let trampolines: Vec<mir::Trampoline> = elab
        .trampolines
        .iter()
        .map(|t| mir::Trampoline {
            export: mir::Symbol(driver.symbols.get_or_intern(&t.name)),
            abi: t.abi.clone(),
            target: driver.symbol_of(t.target, tcx.intern_tys(&t.ty_args)),
        })
        .collect();

    // Deterministic output: sort by mangled text.
    functions.sort_by_cached_key(|f| driver.symbols.resolve(&f.symbol.0));
    records.sort_by_cached_key(|r| driver.symbols.resolve(&r.symbol.0));

    let Driver {
        symbols, reports, ..
    } = driver;
    (
        mir::Program {
            functions,
            records,
            trampolines,
            symbols,
        },
        reports,
    )
}

/// The maximum type-argument nesting depth monomorphization will instantiate
/// before treating the chain as non-terminating — in the spirit of rustc's
/// `recursion_limit`. Deep but finite generic nesting stays well under it; the
/// unbounded type growth of polymorphic recursion blows straight past it.
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

/// Whether a ground type argument is a regional record. Only regional records
/// carry a `Regional`/`Flex`/`Rigid` capability; value/shared records are
/// `Irrelevant` and scalars carry none.
fn is_regional_arg(ty: Ty<'_>) -> bool {
    matches!(
        ty.capability(),
        Some(Capability::Regional | Capability::Flex | Capability::Rigid)
    )
}

/// Mutable worklist + lowering state.
struct Driver<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    mangler: Mangler<'a>,
    /// Interner for every mangled symbol; moved into the final `mir::Program`.
    symbols: Rodeo,
    queue: VecDeque<Instance<'tcx>>,
    /// Function instances already enqueued (so each is emitted once).
    seen: FxHashSet<Instance<'tcx>>,
    /// Ground record instances whose layout is needed (keyed flex-independently).
    records: FxHashSet<(DefId, &'tcx [Ty<'tcx>])>,
    reports: Vec<Report>,
}

impl<'a, 'tcx> Driver<'a, 'tcx> {
    /// Intern a mangled instance symbol (deduping so a callee and its definition
    /// share one [`mir::Symbol`]).
    fn symbol_of(&mut self, def: DefId, ty_args: &'tcx [Ty<'tcx>]) -> mir::Symbol {
        mir::Symbol(
            self.symbols
                .get_or_intern(self.mangler.mangle_instance(def, ty_args)),
        )
    }

    /// Record a ground record instance (for layout) and return its symbol.
    fn record_symbol(&mut self, def: DefId, args: &'tcx [Ty<'tcx>]) -> mir::Symbol {
        self.records.insert((def, args));
        self.symbol_of(def, args)
    }

    /// Push an `Error` diagnostic.
    fn error(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Error,
            message: message.into(),
            span,
        });
    }

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

    /// Collect every record instance reachable from a ground type.
    fn note_records(&mut self, ty: Ty<'tcx>) {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => {
                self.records.insert((def, args));
                for &arg in args {
                    self.note_records(arg);
                }
            }
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.note_records(p);
                }
                self.note_records(ret);
            }
            TyKind::Nullable(inner) => self.note_records(inner),
            _ => {}
        }
    }

    /// Ground a slice of type arguments and re-intern it.
    fn ground_args(&self, ty_args: &[Ty<'tcx>], subst: &Subst<'tcx>) -> &'tcx [Ty<'tcx>] {
        let grounded: Vec<Ty<'tcx>> = ty_args
            .iter()
            .map(|&t| subst_ty(self.tcx, t, subst))
            .collect();
        self.tcx.intern_tys(&grounded)
    }

    /// Lower an expression and arena-allocate it, returning a shared reference.
    fn lower_ref(&mut self, e: &Expr<'tcx>, subst: &Subst<'tcx>) -> &'tcx mir::Expr<'tcx> {
        let lowered = self.lower_expr(e, subst);
        self.tcx.alloc(lowered)
    }

    /// Lower a list of expressions into an arena slice.
    fn lower_slice(&mut self, es: &[Expr<'tcx>], subst: &Subst<'tcx>) -> &'tcx [mir::Expr<'tcx>] {
        let lowered: Vec<mir::Expr<'tcx>> = es.iter().map(|e| self.lower_expr(e, subst)).collect();
        self.tcx.alloc_slice(&lowered)
    }

    /// Lower one Semi expression into a ground MIR expression.
    fn lower_expr(&mut self, e: &Expr<'tcx>, subst: &Subst<'tcx>) -> mir::Expr<'tcx> {
        let ty = subst_ty(self.tcx, e.ty, subst);
        self.note_records(ty);
        let kind = self.lower_kind(&e.kind, subst);
        mir::Expr {
            kind,
            ty,
            span: e.span,
        }
    }

    fn lower_kind(&mut self, kind: &ExprKind<'tcx>, subst: &Subst<'tcx>) -> mir::ExprKind<'tcx> {
        use mir::ExprKind as M;
        match kind {
            ExprKind::GlobalStr(s) => M::GlobalStr(*s),
            ExprKind::ConstInt(n) => M::ConstInt(*n),
            ExprKind::ConstFloat(f) => M::ConstFloat(*f),
            ExprKind::ConstBool(b) => M::ConstBool(*b),
            ExprKind::Var(v) => M::Var(*v),
            ExprKind::Poison => M::Poison,
            ExprKind::Negate(e) => M::Negate(self.lower_ref(e, subst)),
            ExprKind::Not(e) => M::Not(self.lower_ref(e, subst)),
            ExprKind::Arith(l, op, r) => {
                M::Arith(self.lower_ref(l, subst), *op, self.lower_ref(r, subst))
            }
            ExprKind::Cmp(l, op, r) => {
                M::Cmp(self.lower_ref(l, subst), *op, self.lower_ref(r, subst))
            }
            ExprKind::Cast(e, t) => {
                let t = subst_ty(self.tcx, *t, subst);
                self.note_records(t);
                M::Cast(self.lower_ref(e, subst), t)
            }
            ExprKind::If(c, t, f) => M::If(
                self.lower_ref(c, subst),
                self.lower_ref(t, subst),
                self.lower_ref(f, subst),
            ),
            ExprKind::RegionRun(e) => M::RegionRun(self.lower_ref(e, subst)),
            ExprKind::Proj(e, idx) => {
                let base = self.lower_ref(e, subst);
                M::Proj(base, self.tcx.alloc_slice(idx))
            }
            ExprKind::Assign(d, i, s) => {
                M::Assign(self.lower_ref(d, subst), *i, self.lower_ref(s, subst))
            }
            ExprKind::Let {
                var, name, value, ..
            } => M::Let {
                var: *var,
                name: *name,
                value: self.lower_ref(value, subst),
            },
            ExprKind::Seq(es) => M::Seq(self.lower_slice(es, subst)),
            ExprKind::FuncCall {
                target,
                ty_args,
                args,
                regional,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                self.enqueue(*target, ty_args);
                let callee = self.symbol_of(*target, ty_args);
                M::Call {
                    callee,
                    args: self.lower_slice(args, subst),
                    regional: *regional,
                }
            }
            ExprKind::CompoundCall {
                target,
                ty_args,
                args,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                let record = self.record_symbol(*target, ty_args);
                M::Ctor {
                    record,
                    args: self.lower_slice(args, subst),
                }
            }
            ExprKind::VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                let record = self.record_symbol(*target, ty_args);
                M::Variant {
                    record,
                    variant: *variant,
                    args: self.lower_slice(args, subst),
                }
            }
            ExprKind::NullableCall(opt) => {
                M::NullableCall(opt.as_ref().map(|e| self.lower_ref(e, subst)))
            }
            ExprKind::ClosureCall { target, args } => M::ClosureCall {
                target: self.lower_ref(target, subst),
                args: self.lower_slice(args, subst),
            },
            ExprKind::Closure(c) => M::Closure(self.lower_closure(c, subst)),
            ExprKind::Match(scrut, tree) => {
                M::Match(self.lower_ref(scrut, subst), self.lower_tree(tree, subst))
            }
        }
    }

    fn lower_closure(
        &mut self,
        c: &hir::ClosureExpr<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::ClosureExpr<'tcx> {
        // Flex-escape rules (a flex value may not be captured/returned) are a
        // *flexivity* concern and are enforced at the Semi stage on concrete
        // records; monomorphization only verifies the regional requirement at the
        // call boundary (see the worklist loop).
        let captures = self.lower_var_tys(&c.captures, subst);
        let params = self.lower_var_tys(&c.params, subst);
        let body = self.lower_ref(&c.body, subst);
        mir::ClosureExpr {
            captures,
            params,
            body,
        }
    }

    /// Ground a `(var, type)` list (closure captures/params) into an arena slice.
    fn lower_var_tys(
        &mut self,
        vars: &[(hir::VarId, Ty<'tcx>)],
        subst: &Subst<'tcx>,
    ) -> &'tcx [(hir::VarId, Ty<'tcx>)] {
        let grounded: Vec<(hir::VarId, Ty<'tcx>)> = vars
            .iter()
            .map(|(v, t)| {
                let t = subst_ty(self.tcx, *t, subst);
                self.note_records(t);
                (*v, t)
            })
            .collect();
        self.tcx.alloc_slice(&grounded)
    }

    /// Lower a decision tree and arena-allocate it.
    fn lower_tree_ref(
        &mut self,
        tree: &DecisionTree<'tcx>,
        subst: &Subst<'tcx>,
    ) -> &'tcx mir::DecisionTree<'tcx> {
        let lowered = self.lower_tree(tree, subst);
        self.tcx.alloc(lowered)
    }

    /// Copy pattern bindings into the arena (each scrutinee path is its own slice).
    fn lower_bindings(
        &self,
        bindings: &[(hir::VarId, hir::PatVarRef)],
    ) -> &'tcx [mir::Binding<'tcx>] {
        let copied: Vec<mir::Binding<'tcx>> = bindings
            .iter()
            .map(|(var, path)| (*var, self.tcx.alloc_slice(&path.0)))
            .collect();
        self.tcx.alloc_slice(&copied)
    }

    fn lower_tree(
        &mut self,
        tree: &DecisionTree<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::DecisionTree<'tcx> {
        use mir::DecisionTree as M;
        match tree {
            DecisionTree::Uncovered => M::Uncovered,
            DecisionTree::Unreachable => M::Unreachable,
            DecisionTree::Leaf { body, bindings } => M::Leaf {
                body: self.lower_ref(body, subst),
                bindings: self.lower_bindings(bindings),
            },
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => {
                let bindings = self.lower_bindings(bindings);
                M::Guard {
                    bindings,
                    guard: self.lower_ref(guard, subst),
                    success: self.lower_tree_ref(success, subst),
                    failure: self.lower_tree_ref(failure, subst),
                }
            }
            DecisionTree::Switch { scrutinee, cases } => {
                let scrutinee = self.tcx.alloc_slice(&scrutinee.0);
                M::Switch {
                    scrutinee,
                    cases: self.lower_cases(cases, subst),
                }
            }
        }
    }

    /// Lower a list of `(key, sub-tree)` arms into an arena slice.
    fn lower_keyed<K: Copy>(
        &mut self,
        arms: &[(K, DecisionTree<'tcx>)],
        subst: &Subst<'tcx>,
    ) -> &'tcx [(K, mir::DecisionTree<'tcx>)] {
        let lowered: Vec<(K, mir::DecisionTree<'tcx>)> = arms
            .iter()
            .map(|(k, t)| (*k, self.lower_tree(t, subst)))
            .collect();
        self.tcx.alloc_slice(&lowered)
    }

    fn lower_cases(
        &mut self,
        cases: &SwitchCases<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::SwitchCases<'tcx> {
        use mir::SwitchCases as M;
        match cases {
            SwitchCases::Int { cases, default } => M::Int {
                cases: self.lower_keyed(cases, subst),
                default: self.lower_tree_ref(default, subst),
            },
            SwitchCases::Bool { if_true, if_false } => M::Bool {
                if_true: self.lower_tree_ref(if_true, subst),
                if_false: self.lower_tree_ref(if_false, subst),
            },
            SwitchCases::Ctor(arms) => {
                let lowered: Vec<mir::DecisionTree<'tcx>> =
                    arms.iter().map(|t| self.lower_tree(t, subst)).collect();
                M::Ctor(self.tcx.alloc_slice(&lowered))
            }
            SwitchCases::String { cases, default } => M::String {
                cases: self.lower_keyed(cases, subst),
                default: self.lower_tree_ref(default, subst),
            },
            SwitchCases::Nullable { non_null, null } => M::Nullable {
                non_null: self.lower_tree_ref(non_null, subst),
                null: self.lower_tree_ref(null, subst),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// Elaborate `source`, monomorphize it (asserting no diagnostics), and hand
    /// the program to `f`.
    fn with_full<R>(source: &str, f: impl FnOnce(&mir::Program<'_>) -> R) -> R {
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
            let (full, reports) = monomorphize(&elab);
            assert!(
                reports.is_empty(),
                "unexpected mono diagnostics: {reports:#?}"
            );
            f(&full)
        })
    }

    fn symbols(full: &mir::Program<'_>) -> Vec<String> {
        full.functions
            .iter()
            .map(|f| full.symbol(f.symbol).to_string())
            .collect()
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

    /// The immediate sub-expressions of a MIR node (decision-tree arms aside).
    fn children<'tcx>(e: &mir::Expr<'tcx>) -> Vec<&'tcx mir::Expr<'tcx>> {
        use mir::ExprKind::*;
        match e.kind {
            GlobalStr(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_) | Poison => vec![],
            Negate(x) | Not(x) | Cast(x, _) | RegionRun(x) | Proj(x, _) => vec![x],
            Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => vec![l, r],
            If(c, t, f) => vec![c, t, f],
            Let { value, .. } => vec![value],
            Seq(es) => es.iter().collect(),
            Call { args, .. } | Ctor { args, .. } | Variant { args, .. } => args.iter().collect(),
            NullableCall(opt) => opt.into_iter().collect(),
            ClosureCall { target, args } => std::iter::once(target).chain(args.iter()).collect(),
            Closure(c) => vec![c.body],
            Match(scrut, _) => vec![scrut],
        }
    }

    fn assert_ground(e: &mir::Expr<'_>) {
        assert!(is_ground(e.ty), "non-ground expr type: {:?}", e.ty);
        for child in children(e) {
            assert_ground(child);
        }
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
        // `id<T>` is emitted once per distinct instantiation; calls are by symbol.
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
            pub fn use_bool(b: bool) -> bool { id(b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC2idlE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC2idbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC7use_i32".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC8use_bool".to_string()), "{syms:?}");
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
                if let Some(body) = func.body {
                    assert_ground(body);
                }
                for p in &func.params {
                    assert!(
                        is_ground(p.ty),
                        "non-ground param in {}",
                        full.symbol(func.symbol)
                    );
                }
            }
        });
    }

    #[test]
    fn calls_are_resolved_to_callee_symbols() {
        // The MIR dispatches by interned symbol: the body of `use_i32` calls the
        // ground `id<i32>` instance directly by its mangled name.
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
        "#;
        with_full(src, |full| {
            let user = full
                .functions
                .iter()
                .find(|f| full.symbol(f.symbol) == "_RC7use_i32")
                .expect("use_i32 emitted");
            let body = user.body.expect("use_i32 has a body");
            // The body is a `Seq` whose tail expression is the `id(n)` call.
            let mir::ExprKind::Seq(stmts) = body.kind else {
                panic!("expected a Seq body, got {:?}", body.kind);
            };
            let mir::ExprKind::Call { callee, .. } = stmts.last().expect("non-empty seq").kind
            else {
                panic!("expected a Call, got {:?}", stmts.last().unwrap().kind);
            };
            assert_eq!(full.symbol(callee), "_RIC2idlE");
        });
    }

    #[test]
    fn multiple_type_params_instantiate_per_ordering() {
        let src = r#"
            fn pick<A, B>(a: A, b: B) -> A { a }
            fn use_ib(n: i32, b: bool) -> i32 { pick(n, b) }
            fn use_bi(b: bool, n: i32) -> bool { pick(b, n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4picklbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4pickblE".to_string()), "{syms:?}");
            let picks = syms.iter().filter(|s| s.contains("pick")).count();
            assert_eq!(picks, 2, "{syms:?}");
        });
    }

    #[test]
    fn permuting_type_params_terminates() {
        let src = r#"
            fn swap<A, B>(a: A, b: B) -> i32 { swap(b, a) }
            fn main(n: i32, b: bool) -> i32 { swap(n, b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4swaplbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4swapblE".to_string()), "{syms:?}");
            let swaps = syms.iter().filter(|s| s.contains("swap")).count();
            assert_eq!(swaps, 2, "{syms:?}");
        });
    }

    #[test]
    fn mutual_recursion_at_fixed_type_terminates() {
        let src = r#"
            fn ping<T>(x: T) -> i32 { pong(x) }
            fn pong<T>(x: T) -> i32 { ping(x) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4pinglE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4ponglE".to_string()), "{syms:?}");
        });
    }

    #[test]
    fn monomorphic_recursion_terminates() {
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
        let src = r#"
            struct Wrap<T> { value: T }
            fn rec<T>(x: T) -> i32 { rec(Wrap { value: x }) }
            fn main(n: i32) -> i32 { rec(n) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_polymorphic_recursion_in_one_of_several_params() {
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
        let src = r#"
            struct Wrap<T> { value: T }
            fn ping<T>(x: T) -> i32 { pong(Wrap { value: x }) }
            fn pong<T>(x: T) -> i32 { ping(Wrap { value: x }) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    fn flex_generic_accepts_regional_instantiation() {
        // `foo<T>(bar: [flex] T)` requires T to be regional; instantiating it at a
        // regional record is accepted (no diagnostic).
        let src = r#"
            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }
            regional fn foo<T>(bar: [flex] T) -> i32 { 0 }
            regional fn use_ok(c: [flex] Cell<i32>) -> i32 { foo(c) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.iter().any(|s| s.contains("foo")), "{syms:?}");
        });
    }

    #[test]
    fn flex_generic_rejects_non_regional_instantiation() {
        // The same `[flex] T` parameter instantiated at a value record is rejected
        // at the call boundary — only regional records may be flex.
        let src = r#"
            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }
            struct Pair { a: i32 }
            regional fn foo<T>(bar: [flex] T) -> i32 { 0 }
            regional fn use_bad(p: Pair) -> i32 { foo(p) }
        "#;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (_full, reports) = monomorphize(&elab);
            assert!(
                reports
                    .iter()
                    .any(|r| r.message.contains("regional record")),
                "expected a regionality diagnostic, got {reports:#?}"
            );
        });
    }
}
