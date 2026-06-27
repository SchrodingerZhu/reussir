//! Ownership analysis: Perceus-style precise reference counting over the Full
//! MIR.
//!
//! The job is to decide, for every resource-relevant (RR) value, *where* the
//! backend should `dup` (rc++) or `drop` (rc--) it so that each owned value is
//! released exactly once, with drops placed as early as possible (so the backend
//! `TokenReuse` pass can pair a freed token with a same-layout allocation). The
//! MIR is immutable, so the result is a side-table keyed by [`mir::ExprId`]
//! anchors rather than a rewritten tree — see [`OwnershipTable`].
//!
//! This is **increment 1** (see `docs/ownership-analysis.md` §10): the linear
//! core (`Var` / `Let` / `Seq` / `Call` / `Ctor` / `Variant` / `NullableCall`),
//! the [`is_rr`](Rr::is_rr) predicate (including transitive value-records), and
//! the test harness. Control flow (`If` / `Match`), container borrows (`Proj`),
//! closures, and regions land in later increments; the forms they need
//! [`unimplemented!`] loudly rather than silently miscounting.
//!
//! # Discipline
//!
//! - **Owned calling convention.** A function owns its RR parameters; it must
//!   consume-or-drop each exactly once. The return value is moved out (owned by
//!   the caller, never dropped here).
//! - **Last use is the pivot.** A non-last use of an RR var `dup`s it (keeping
//!   ownership); the last use moves it (consumes, no `dup`); a var that goes dead
//!   without being consumed is `drop`ped at that dead point.
//!
//! # Operational semantics
//!
//! Judgment `L ⊢ e ⤳ 𝒜`: "with continuation-live set `L`, expression `e`
//! annotates to actions `𝒜`". `free(e)` is the set of named RR vars used in `e`;
//! `L` is the set live in `e`'s continuation. The increment-1 rules:
//!
//! ```text
//! x ∈ L                                   x ∉ L
//! ─────────────────── [Var-Shared]        ─────────────────── [Var-LastUse]
//! L ⊢ x ⤳ {before: dup x}                 L ⊢ x ⤳ {}            (move)
//!
//!  L ⊢ aₖ ⤳ 𝒜ₖ   with  Lᵢ = free(aᵢ₊₁..aₖ) ∪ L
//! ─────────────────────────────────────────────── [Args]   (Call/Ctor/Variant)
//!  L ⊢ f(a₀..aₖ) ⤳ ⊕ᵢ 𝒜ᵢ
//!  ⇒ a var reused in a later arg is live after the earlier one, so the earlier
//!    occurrence is not a last use and takes a `dup` (the `f(x, x)` case).
//!
//! L ⊢ value ⤳ 𝒜       x ∉ Lᵢ                  L ⊢ value ⤳ 𝒜     x ∈ Lᵢ
//! ─────────────────────────────────── [Let-Dead]    ──────────────────── [Let-Live]
//! L ⊢ (let x = value) ⤳ 𝒜 ⊕ {after: drop x}        L ⊢ (let x = value) ⤳ 𝒜
//!   where, inside Seq([s₀..sₙ]), Lᵢ = free(sᵢ₊₁..sₙ) ∪ L.
//!
//!  p ∈ params,  is_rr(p),  p ∉ free(body)
//! ──────────────────────────────────────── [Param-Unused]
//!  {} ⊢ body ⤳ … ⊕ {root.before: drop p}
//! ```
//!
//! # Relation to Tree Borrows
//!
//! Borrowed as *design inspiration*, not as a model: each owned RR var carries a
//! per-entity state (`Owned → {Consumed | Dropped}`), transitioned on use events,
//! and we reconcile sibling control-flow branches at joins. Unlike Tree Borrows —
//! a *runtime* semantics that defines UB — this is a *static rewrite* that
//! inserts the rc ops so the corresponding UB is unrepresentable. See
//! `docs/ownership-analysis.md` §7.1.

use std::cell::RefCell;

use rustc_hash::FxHashMap;

use crate::full::mir::{Expr, ExprKind, Function};
use crate::semi::hir::{ExprId, VarId};
use crate::semi::ty::{Capability, Ty, TyCtxt, TyKind};

// ---------------------------------------------------------------------------
// Output data model
// ---------------------------------------------------------------------------

/// A single reference-counting operation the backend must emit. Keyed by
/// [`VarId`]; richer targets (`DropField(VarId, Path)`, dropping an anonymous
/// result) arrive with the container and control-flow increments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RcOp {
    /// Increment the refcount of `x` (it has another live owner).
    Dup(VarId),
    /// Decrement the refcount of `x` (its last owner here is releasing it).
    Drop(VarId),
}

/// The rc ops to run immediately *before* and *after* evaluating one expression.
/// Timeline: `[before] → evaluate expr → [after]`.
#[derive(Clone, Default, Debug)]
pub struct Action {
    pub before: Vec<RcOp>,
    pub after: Vec<RcOp>,
}

/// The analysis result: a side-table from each [`mir::ExprId`] anchor to the rc
/// ops surrounding it. The MIR itself is untouched; codegen looks up
/// `table.before(id)` / `table.after(id)` around each node.
#[derive(Default, Debug)]
pub struct OwnershipTable {
    actions: FxHashMap<ExprId, Action>,
}

const NO_OPS: &[RcOp] = &[];

impl OwnershipTable {
    /// The full action recorded at `id`, if any.
    pub fn get(&self, id: ExprId) -> Option<&Action> {
        self.actions.get(&id)
    }

    /// The ops to run before evaluating the node `id` (empty if none).
    pub fn before(&self, id: ExprId) -> &[RcOp] {
        self.actions.get(&id).map_or(NO_OPS, |a| &a.before)
    }

    /// The ops to run after evaluating the node `id` (empty if none).
    pub fn after(&self, id: ExprId) -> &[RcOp] {
        self.actions.get(&id).map_or(NO_OPS, |a| &a.after)
    }
}

// ---------------------------------------------------------------------------
// Record management classification
// ---------------------------------------------------------------------------

/// How a record's memory is managed — its [`ctxt::Record.default_cap`] carried
/// forward. This axis is **orthogonal to the `Ty`'s capability**: the capability
/// records *regional value coloring* (and is [`Capability::Irrelevant`] for any
/// record that does not participate in it — i.e. every non-regional record), so
/// it cannot say whether a record is Rc-managed. That is decided here, from the
/// record table, for every record (regional or not).
///
/// [`ctxt::Record.default_cap`]: crate::semi::ctxt::Record
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Managed {
    /// Stored inline (by value). Rc-managed only transitively — RR iff a field is.
    Value,
    /// Heap-allocated and reference-counted (the default `struct`/`enum`).
    Shared,
    /// Region-allocated and reference-counted (a `[regional]` record).
    Regional,
}

impl Managed {
    /// Whether the record itself owns an rc (`Shared`/`Regional`), independent of
    /// its fields. A `Value` record is rc only through what it transitively holds.
    pub fn is_rc(self) -> bool {
        matches!(self, Managed::Shared | Managed::Regional)
    }
}

/// The shape of a ground record instance, as the ownership pass needs it: how it
/// is managed and (for value records) its ground field types, flattened across
/// all variants, so `is_rr` can decide transitive resource-relevance.
#[derive(Clone, Debug)]
pub struct RecordShape<'tcx> {
    pub managed: Managed,
    pub fields: Vec<Ty<'tcx>>,
}

/// How every ground record instance is managed, keyed by its **canonical** record
/// type (capability normalized to [`Capability::Irrelevant`], matching the
/// instance types `mono` records, so a record and its regionally-colored variants
/// share one entry). Built from the elaborated record table by the pipeline;
/// built directly by the test harness.
#[derive(Default)]
pub struct RecordTable<'tcx> {
    shapes: FxHashMap<Ty<'tcx>, RecordShape<'tcx>>,
}

impl<'tcx> RecordTable<'tcx> {
    pub fn new() -> Self {
        RecordTable {
            shapes: FxHashMap::default(),
        }
    }

    /// Register `canonical_ty`'s shape. `canonical_ty` must carry
    /// [`Capability::Irrelevant`] (the form `is_rr` looks up).
    pub fn insert(&mut self, canonical_ty: Ty<'tcx>, shape: RecordShape<'tcx>) {
        self.shapes.insert(canonical_ty, shape);
    }

    fn shape(&self, canonical_ty: Ty<'tcx>) -> Option<&RecordShape<'tcx>> {
        self.shapes.get(&canonical_ty)
    }
}

// ---------------------------------------------------------------------------
// The is_rr predicate
// ---------------------------------------------------------------------------

/// The resource-relevance predicate, with its memo and the record table it
/// consults. Held by the [`Analyzer`]; also usable standalone (the test harness's
/// balanced-rc checker borrows one).
pub(crate) struct Rr<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    table: &'a RecordTable<'tcx>,
    /// `Ty` → is_rr, memoized (types are interned, so the pointer key is exact).
    memo: RefCell<FxHashMap<Ty<'tcx>, bool>>,
}

impl<'a, 'tcx> Rr<'a, 'tcx> {
    pub(crate) fn new(tcx: &'a TyCtxt<'tcx>, table: &'a RecordTable<'tcx>) -> Self {
        Rr {
            tcx,
            table,
            memo: RefCell::new(FxHashMap::default()),
        }
    }

    /// Is a value of `ty` reference-counted (needs `dup`/`drop`)?
    ///
    /// An Rc-managed record (`Shared`/`Regional`, per the record table) and a
    /// closure are always RR; a `Value` record is RR iff some field transitively
    /// is; `Nullable(inner)` follows `inner`; scalars are not. The record's `Ty`
    /// *capability* is not consulted — it tracks regional value coloring, a
    /// different axis from rc-management.
    pub(crate) fn is_rr(&self, ty: Ty<'tcx>) -> bool {
        if let Some(&b) = self.memo.borrow().get(&ty) {
            return b;
        }
        // Provisional `false` before recursing: a value-record field cycle can
        // only close through an rc-managed link (which is RR regardless), so
        // reading `false` for an in-progress value type on the back-edge is
        // correct (a pure inline cycle would be an illegal infinite-size type).
        // Overwritten with the real answer below.
        self.memo.borrow_mut().insert(ty, false);
        let b = match *ty.kind() {
            // Rc-management is the record table's call, for every record — the
            // capability (regional coloring) is canonicalized away for the key.
            TyKind::Record { def, args, .. } => {
                let canonical = self.tcx.mk_record(def, args, Capability::Irrelevant);
                match self.table.shape(canonical) {
                    Some(shape) if shape.managed.is_rc() => true,
                    Some(shape) => shape.fields.iter().any(|&f| self.is_rr(f)),
                    // Unknown record: treat as non-RR. The pipeline populates
                    // every reachable instance; a miss means a scalar-only type.
                    None => false,
                }
            }
            TyKind::Closure { .. } => true,
            TyKind::Nullable(inner) => self.is_rr(inner),
            TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Str
            | TyKind::Unit
            | TyKind::Generic(_)
            | TyKind::Hole(_)
            | TyKind::Bottom => false,
        };
        self.memo.borrow_mut().insert(ty, b);
        b
    }
}

// ---------------------------------------------------------------------------
// Live-variable sets
// ---------------------------------------------------------------------------

/// A dense set of [`VarId`]s, a fixed-width bitset (var ids are dense per
/// function). Backs both `free(e)` and the threaded `live_after` sets.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct VarSet {
    words: Vec<u64>,
}

impl VarSet {
    fn insert(&mut self, v: VarId) {
        let i = v.0 as usize;
        let w = i / 64;
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1u64 << (i % 64);
    }

    fn contains(&self, v: VarId) -> bool {
        let i = v.0 as usize;
        self.words
            .get(i / 64)
            .is_some_and(|x| (x >> (i % 64)) & 1 == 1)
    }

    fn union_with(&mut self, other: &VarSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }
}

// ---------------------------------------------------------------------------
// The analyzer
// ---------------------------------------------------------------------------

/// Analyze one function body, returning the rc ops it needs. `table` classifies
/// every ground record the body mentions (value vs shared + value-record fields).
pub fn analyze_function<'tcx>(
    tcx: &TyCtxt<'tcx>,
    func: &Function<'tcx>,
    table: &RecordTable<'tcx>,
) -> OwnershipTable {
    let mut a = Analyzer {
        rr: Rr::new(tcx, table),
        free_memo: FxHashMap::default(),
        actions: FxHashMap::default(),
    };
    if let Some(body) = func.body {
        // Top-level continuation is empty: the return value is moved to the
        // caller, everything else must be settled within the body.
        a.place(body, &VarSet::default());
        // [Param-Unused]: an RR parameter never used anywhere is dead on entry.
        let body_free = a.free(body);
        for p in &func.params {
            if a.rr.is_rr(p.ty) && !body_free.contains(p.var) {
                a.add_before(body.id, RcOp::Drop(p.var));
            }
        }
    }
    OwnershipTable { actions: a.actions }
}

struct Analyzer<'a, 'tcx> {
    rr: Rr<'a, 'tcx>,
    /// `free(e)` memoized per anchor (Pass A). Computed bottom-up, read while
    /// threading `live_after` top-down.
    free_memo: FxHashMap<ExprId, VarSet>,
    actions: FxHashMap<ExprId, Action>,
}

impl<'tcx> Analyzer<'_, 'tcx> {
    fn add_before(&mut self, id: ExprId, op: RcOp) {
        self.actions.entry(id).or_default().before.push(op);
    }

    fn add_after(&mut self, id: ExprId, op: RcOp) {
        self.actions.entry(id).or_default().after.push(op);
    }

    // --- Pass A: free RR variables, memoized per anchor ---

    fn free(&mut self, e: &Expr<'tcx>) -> VarSet {
        if let Some(s) = self.free_memo.get(&e.id) {
            return s.clone();
        }
        let s = self.compute_free(e);
        self.free_memo.insert(e.id, s.clone());
        s
    }

    fn compute_free(&mut self, e: &Expr<'tcx>) -> VarSet {
        let mut s = VarSet::default();
        match e.kind {
            ExprKind::Var(x) => {
                if self.rr.is_rr(e.ty) {
                    s.insert(x);
                }
            }
            ExprKind::GlobalStr(_)
            | ExprKind::ConstInt(_)
            | ExprKind::ConstFloat(_)
            | ExprKind::ConstBool(_)
            | ExprKind::Poison => {}
            ExprKind::Negate(x) | ExprKind::Not(x) | ExprKind::Cast(x, _) => {
                s.union_with(&self.free(x));
            }
            ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
                s.union_with(&self.free(l));
                s.union_with(&self.free(r));
            }
            ExprKind::Let { value, .. } => s.union_with(&self.free(value)),
            ExprKind::Seq(es) => {
                for st in es {
                    s.union_with(&self.free(st));
                }
            }
            ExprKind::Call { args, .. }
            | ExprKind::Ctor { args, .. }
            | ExprKind::Variant { args, .. } => {
                for arg in args {
                    s.union_with(&self.free(arg));
                }
            }
            ExprKind::NullableCall(opt) => {
                if let Some(x) = opt {
                    s.union_with(&self.free(x));
                }
            }
            ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::Proj(..)
            | ExprKind::Assign(..)
            | ExprKind::RegionRun(..)
            | ExprKind::Closure(..)
            | ExprKind::ClosureCall { .. } => {
                unimplemented!(
                    "ownership::free: `{}` lands in a later increment (see ownership-analysis.md §10)",
                    kind_name(&e.kind)
                )
            }
        }
        s
    }

    // --- Pass B: placement, threading continuation liveness top-down ---

    fn place(&mut self, e: &Expr<'tcx>, live_after: &VarSet) {
        match e.kind {
            // [Var-Shared] / [Var-LastUse]: dup if still live afterwards, else
            // this is the last use and the value is moved (consumed) in place.
            ExprKind::Var(x) => {
                if self.rr.is_rr(e.ty) && live_after.contains(x) {
                    self.add_before(e.id, RcOp::Dup(x));
                }
            }
            ExprKind::GlobalStr(_)
            | ExprKind::ConstInt(_)
            | ExprKind::ConstFloat(_)
            | ExprKind::ConstBool(_)
            | ExprKind::Poison => {}
            // Scalar unary: the operand never holds an RR value, but recurse for
            // generality (and to keep liveness threading uniform).
            ExprKind::Negate(x) | ExprKind::Not(x) | ExprKind::Cast(x, _) => {
                self.place(x, live_after);
            }
            // Scalar binary, left-to-right: `l` is live across `r`.
            ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
                let mut la = self.free(r);
                la.union_with(live_after);
                self.place(l, &la);
                self.place(r, live_after);
            }
            ExprKind::Let { var, value, .. } => {
                self.place(value, live_after);
                // [Let-Dead]: bound but not used downstream ⇒ drop right after
                // binding. `live_after` here is the let's own continuation set.
                if self.rr.is_rr(value.ty) && !live_after.contains(var) {
                    self.add_after(e.id, RcOp::Drop(var));
                }
            }
            ExprKind::Seq(es) => self.place_seq(es, live_after),
            ExprKind::Call { args, .. }
            | ExprKind::Ctor { args, .. }
            | ExprKind::Variant { args, .. } => self.place_args(args, live_after),
            ExprKind::NullableCall(opt) => {
                if let Some(x) = opt {
                    self.place(x, live_after);
                }
            }
            ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::Proj(..)
            | ExprKind::Assign(..)
            | ExprKind::RegionRun(..)
            | ExprKind::Closure(..)
            | ExprKind::ClosureCall { .. } => {
                unimplemented!(
                    "ownership::place: `{}` lands in a later increment (see ownership-analysis.md §10)",
                    kind_name(&e.kind)
                )
            }
        }
    }

    /// `Seq([s₀..sₙ])`: `sₙ` is the moved-out result; earlier statements run for
    /// effect. Each `sᵢ`'s continuation-live set is `free(sᵢ₊₁..sₙ) ∪ L`.
    fn place_seq(&mut self, es: &'tcx [Expr<'tcx>], live_after: &VarSet) {
        let n = es.len();
        if n == 0 {
            return;
        }
        let mut afters = vec![VarSet::default(); n];
        afters[n - 1] = live_after.clone();
        for i in (0..n - 1).rev() {
            let mut s = self.free(&es[i + 1]);
            s.union_with(&afters[i + 1]);
            afters[i] = s;
        }
        for (i, st) in es.iter().enumerate() {
            self.place_stmt(st, &afters[i], i == n - 1);
        }
    }

    fn place_stmt(&mut self, s: &Expr<'tcx>, after: &VarSet, is_last: bool) {
        self.place(s, after);
        // A non-final statement whose RR result is neither bound (`Let`) nor
        // consumed leaves an owned value with no name to drop. Anonymous-result
        // drops arrive with the control-flow increment.
        if !is_last && !matches!(s.kind, ExprKind::Let { .. }) && self.rr.is_rr(s.ty) {
            unimplemented!(
                "ownership: an unconsumed non-`let` statement of RR type needs a \
                 result-drop (control-flow increment)"
            );
        }
    }

    /// [Args]: each argument is consumed by the call/ctor; a var reused in a
    /// later argument is live across the earlier occurrence (so it gets a `dup`).
    fn place_args(&mut self, args: &'tcx [Expr<'tcx>], live_after: &VarSet) {
        let n = args.len();
        if n == 0 {
            return;
        }
        let mut afters = vec![VarSet::default(); n];
        afters[n - 1] = live_after.clone();
        for i in (0..n - 1).rev() {
            let mut s = self.free(&args[i + 1]);
            s.union_with(&afters[i + 1]);
            afters[i] = s;
        }
        for (i, arg) in args.iter().enumerate() {
            self.place(arg, &afters[i]);
        }
    }
}

/// A short name for an [`ExprKind`], for `unimplemented!` diagnostics.
fn kind_name(kind: &ExprKind<'_>) -> &'static str {
    match kind {
        ExprKind::GlobalStr(_) => "GlobalStr",
        ExprKind::ConstInt(_) => "ConstInt",
        ExprKind::ConstFloat(_) => "ConstFloat",
        ExprKind::ConstBool(_) => "ConstBool",
        ExprKind::Var(_) => "Var",
        ExprKind::Negate(_) => "Negate",
        ExprKind::Not(_) => "Not",
        ExprKind::Arith(..) => "Arith",
        ExprKind::Cmp(..) => "Cmp",
        ExprKind::Cast(..) => "Cast",
        ExprKind::If(..) => "If",
        ExprKind::RegionRun(_) => "RegionRun",
        ExprKind::Proj(..) => "Proj",
        ExprKind::Assign(..) => "Assign",
        ExprKind::Let { .. } => "Let",
        ExprKind::Seq(_) => "Seq",
        ExprKind::Call { .. } => "Call",
        ExprKind::Ctor { .. } => "Ctor",
        ExprKind::Variant { .. } => "Variant",
        ExprKind::NullableCall(_) => "NullableCall",
        ExprKind::Closure(_) => "Closure",
        ExprKind::ClosureCall { .. } => "ClosureCall",
        ExprKind::Match(..) => "Match",
        ExprKind::Poison => "Poison",
    }
}

#[cfg(test)]
mod tests;
