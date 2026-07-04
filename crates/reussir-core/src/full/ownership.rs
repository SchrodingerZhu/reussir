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
//! Every Full MIR form is handled (see `docs/ownership-analysis.md` §10): the
//! linear core (`Var` / `Let` / `Seq` / `Call` / `Ctor` / `Variant` /
//! `NullableCall`), the [`is_rr`](Rr::is_rr) predicate (including transitive
//! value-records), `If` branch reconciliation, discarded-result drops, **pattern
//! matching** (`Match` over `Switch` + `Leaf` + `Guard`, borrow-dup), container
//! borrows (`Proj`, `Assign`), closures (`Closure`, `ClosureCall`), and regions
//! (`RegionRun`). No form is deferred — the analysis is total over the MIR.
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
//! `L` is the set live in `e`'s continuation. The core rules:
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
//!
//!  L ⊢ t ⤳ 𝒜ₜ      L ⊢ e ⤳ 𝒜ₑ      free(c) ∪ free(t) ∪ free(e) ∪ L ⊢ c ⤳ 𝒜_c
//! ──────────────────────────────────────────────────────────────────────── [If-Reconcile]
//!  L ⊢ (if c then t else e) ⤳ 𝒜_c ⊕ 𝒜ₜ ⊕ 𝒜ₑ
//!                ⊕ {t.before: drop x | x ∈ free(e) \ free(t), x ∉ L}
//!                ⊕ {e.before: drop x | x ∈ free(t) \ free(e), x ∉ L}
//!   A var owned on entry but used in only one branch and dead afterwards is
//!   settled (moved) on the branch that uses it and dropped on the branch that
//!   does not, so both arms exit owning the same set. A var in `L` survives both
//!   arms (dup'd at any in-arm use); a var used in both arms is moved in each.
//!
//!  s is a non-final Seq statement, is_rr(s), s not a `let`
//! ────────────────────────────────────────────────────────── [Discard]
//!  … ⊕ {s.after: drop-value s}     (its result is owned but unconsumed)
//! ```
//!
//! # Match (borrow-dup, à la Perceus/Koka)
//!
//! A `match` **borrows** its scrutinee to read fields; it does not consume it at
//! the scrutinee position. Per leaf (the reconciliation join is N-way over the
//! switch arms, like [If-Reconcile]):
//!
//! - a used RR field binding `(y, path≠[])` takes `dup y` (borrow-extract: the
//!   field is aliased out while the scrutinee still holds its reference);
//! - the scrutinee is **consumed** iff it is dead after the match (a `Var(s)` with
//!   `s ∉ free(arms) ∪ L`, or any temporary); a consumed scrutinee is dropped at
//!   each leaf *after* the field dups (early, so the backend can reuse the token);
//! - binding the **whole** scrutinee `(y, [])` is an identity, so it follows
//!   ordinary move/borrow: a consumed scrutinee is *moved* into `y` (no op, no
//!   drop), a borrowed one is `dup`'d;
//! - an **unused** binding needs no op at all — a consumed scrutinee's drop
//!   recursively frees its un-extracted fields, and a borrowed scrutinee is never
//!   extracted.
//!
//! Reuse *specialization* (fusing `dup field; drop parent` into an in-place,
//! uniqueness-guarded move) is the backend's job, exactly as Koka separates
//! Perceus from its reuse analysis — the frontend only emits the early drop.
//!
//! A **guard** keeps the scrutinee live across its re-testable `failure` path, so
//! the scrutinee drop is *not* hoisted above the guard — it stays in each terminal
//! leaf. The guard's pattern bindings are borrow-extracted (dup at the guard,
//! before either branch); a binding used only in the guard is consumed there,
//! while one that survives into a branch is settled there (the branch that uses it
//! moves it, the branch that does not drops it) — i.e. it joins the sub-trees'
//! per-leaf settle obligations.
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

use crate::utils::bitset::HybridBitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::full::mir::{self, DecisionTree, Expr, ExprKind, Function, SwitchCases};
use crate::semi::hir::{ExprId, VarId};
use crate::semi::ty::{Flexivity, Ty, TyCtxt, TyKind};

// ---------------------------------------------------------------------------
// Output data model
// ---------------------------------------------------------------------------

/// A single reference-counting operation the backend must emit, keyed either by
/// [`VarId`] (a named owner) or by [`ExprId`] (an anonymous intermediate result).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RcOp {
    /// Increment the refcount of `x` (it has another live owner).
    Dup(VarId),
    /// Decrement the refcount of `x` (its last owner here is releasing it).
    Drop(VarId),
    /// Decrement the refcount of the **anonymous result** of node `id` — a
    /// statement value that is produced but neither bound nor consumed (a
    /// discarded non-final `Seq` statement, or a consumed temporary that was
    /// projected from). Recorded in that node's `after`.
    DropValue(ExprId),
    /// Increment the refcount of the **anonymous result** of node `id` — a field
    /// borrow-extracted by a [`Proj`](ExprKind::Proj): the parent record still
    /// owns the original, so the extracted owned copy must be inc'd.
    DupValue(ExprId),
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
/// forward. This is a **different axis from the `Ty`'s [`Flexivity`]**: flexivity
/// is the *regional value coloring* (and is [`Flexivity::Irrelevant`] for any
/// non-regional record), while management says how the storage is reclaimed. The
/// management class comes from the record table, never from the flexivity.
///
/// The two axes are independent for `Value`/`Shared`, and meet in exactly one
/// place: a `Regional` record is rc-managed only when its flexivity is `Rigid`
/// (see [`Rr::is_rr`]).
///
/// [`ctxt::Record.default_cap`]: crate::semi::ctxt::Record
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Managed {
    /// Stored inline (by value). Rc-managed only transitively — RR iff a field is.
    Value,
    /// Heap-allocated and reference-counted (the default `struct`/`enum`). Always
    /// RR.
    Shared,
    /// Region-allocated (a `[regional]` record). RR only when the value's
    /// [`Flexivity`] is `Rigid` (frozen out of its region into a managed object);
    /// a region-local `Flex` / unrefined value is freed with its region and needs
    /// no rc. This is the one place the two axes (management vs flexivity) meet —
    /// see [`Rr::is_rr`].
    Regional,
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
/// type (capability normalized to [`Flexivity::Irrelevant`], matching the
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

    /// Build the table from a monomorphized program's record instances. Each
    /// instance's nominal type is already capability-canonicalized
    /// ([`Flexivity::Irrelevant`]), the form [`is_rr`](Rr::is_rr) looks up, so it
    /// is used directly as the key.
    pub fn from_records(records: &[mir::RecordInstance<'tcx>]) -> Self {
        use crate::semi::ctxt::DefaultCap;
        let mut table = RecordTable::new();
        for r in records {
            let managed = match r.default_cap {
                DefaultCap::Value => Managed::Value,
                DefaultCap::Shared => Managed::Shared,
                DefaultCap::Regional => Managed::Regional,
            };
            let fields = match r.layout {
                mir::RecordLayout::Compound(members) => members.iter().map(|m| m.ty).collect(),
                mir::RecordLayout::Variant(variants) => variants
                    .iter()
                    .flat_map(|v| v.fields.iter().copied())
                    .collect(),
            };
            table.insert(r.ty, RecordShape { managed, fields });
        }
        table
    }

    /// Register `canonical_ty`'s shape. `canonical_ty` must carry
    /// [`Flexivity::Irrelevant`] (the form `is_rr` looks up).
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
    /// A `Shared` record and a closure are always RR; a `Value` record is RR iff
    /// some field transitively is; a `Regional` record is RR only when its
    /// [`Flexivity`] is `Rigid` (frozen into a managed object — a `Flex` or
    /// unrefined-regional value is freed with its region); `Nullable(inner)`
    /// follows `inner`; scalars are not. Management comes from the record table;
    /// flexivity is consulted only to settle the regional case.
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
            // The management class is the record table's call (keyed by the
            // canonical, flexivity-erased type); the per-use flexivity then refines
            // the one case where it matters — a regional record.
            TyKind::Record { def, args, flex } => {
                let canonical = self.tcx.mk_record(def, args, Flexivity::Irrelevant);
                match self.table.shape(canonical) {
                    Some(shape) => match shape.managed {
                        Managed::Shared => true,
                        Managed::Value => shape.fields.iter().any(|&f| self.is_rr(f)),
                        // A region-allocated value is freed en masse when its region
                        // is torn down — no per-object rc — *unless* it has been
                        // frozen to `Rigid`, which lifts it to a managed (refcounted)
                        // object that escapes the region. Flex / unrefined-regional
                        // values stay region-local, so they need no dup/drop.
                        Managed::Regional => flex == Flexivity::Rigid,
                    },
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
            | TyKind::Char
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

/// A set of [`VarId`]s, backed by a [`HybridBitSet`]. Var ids are dense `u32`s
/// (allocated per function), so a body with a handful of locals stays in the
/// hybrid's word-backed dense form and only large bodies pay for a compressed
/// bitmap. Backs both `free(e)` and the threaded `live_after` sets.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct VarSet(HybridBitSet);

impl VarSet {
    fn insert(&mut self, v: VarId) {
        self.0.insert(v.0);
    }

    fn contains(&self, v: VarId) -> bool {
        self.0.contains(v.0)
    }

    /// In place: `self ← self ∪ other`.
    fn union_with(&mut self, other: &VarSet) {
        self.0.union_with(&other.0);
    }

    /// Remove every member of `other` (set difference, in place).
    fn subtract(&mut self, other: &VarSet) {
        self.0.subtract(&other.0);
    }

    /// The members, in ascending `VarId` order (the hybrid set iterates sorted,
    /// keeping op emission deterministic).
    fn iter(&self) -> impl Iterator<Item = VarId> + '_ {
        self.0.iter().map(VarId)
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
            | ExprKind::ConstChar(_)
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
            ExprKind::If(c, t, e) => {
                s.union_with(&self.free(c));
                s.union_with(&self.free(t));
                s.union_with(&self.free(e));
            }
            ExprKind::Match(scrut, tree) => {
                s.union_with(&self.free(scrut));
                s.union_with(&self.free_tree(&tree));
            }
            // A projection reads (borrows) its base: the base's vars are used, so
            // the enclosing scope keeps them live until the projection.
            ExprKind::Proj(base, _) => s.union_with(&self.free(base)),
            ExprKind::Assign(dst, _, src) => {
                s.union_with(&self.free(dst));
                s.union_with(&self.free(src));
            }
            ExprKind::RegionRun(x) => s.union_with(&self.free(x)),
            // A closure consumes its captured outer vars; its body is a separate
            // scope and does not contribute to the enclosing function's free set.
            ExprKind::Closure(c) => {
                for &(v, ty) in c.captures {
                    if self.rr.is_rr(ty) {
                        s.insert(v);
                    }
                }
            }
            ExprKind::ClosureCall { target, args } => {
                s.union_with(&self.free(target));
                for arg in args {
                    s.union_with(&self.free(arg));
                }
            }
        }
        s
    }

    /// The free RR variables of a decision tree: vars used in arm bodies (and
    /// guards), **minus** the variables the patterns bind (those are internal to
    /// the match, not free in it).
    fn free_tree(&mut self, tree: &DecisionTree<'tcx>) -> VarSet {
        let mut free = self.tree_body_free(tree);
        let bound = tree_bound(tree);
        free.subtract(&bound);
        free
    }

    /// The union of the free sets of every arm body / guard in the tree (still
    /// including pattern-bound vars; [`free_tree`](Self::free_tree) removes them).
    fn tree_body_free(&mut self, tree: &DecisionTree<'tcx>) -> VarSet {
        let mut s = VarSet::default();
        match *tree {
            DecisionTree::Uncovered | DecisionTree::Unreachable => {}
            DecisionTree::Leaf { body, .. } => s.union_with(&self.free(body)),
            DecisionTree::Guard {
                guard,
                success,
                failure,
                ..
            } => {
                s.union_with(&self.free(guard));
                s.union_with(&self.tree_body_free(success));
                s.union_with(&self.tree_body_free(failure));
            }
            DecisionTree::Switch { cases, .. } => {
                for sub in subtrees(&cases) {
                    s.union_with(&self.tree_body_free(sub));
                }
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
            | ExprKind::ConstChar(_)
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
            ExprKind::If(c, t, e) => self.place_if(c, t, e, live_after),
            ExprKind::Match(scrut, tree) => self.place_match(scrut, &tree, live_after),
            // Borrow-extract: the result is an owned field copy (inc'd, since the
            // parent keeps the original); the base is navigated and its root
            // dropped early if this is its last use.
            ExprKind::Proj(base, _) => {
                if self.rr.is_rr(e.ty) {
                    self.add_after(e.id, RcOp::DupValue(e.id));
                }
                self.navigate_base(base, e.id, live_after);
            }
            ExprKind::Assign(dst, _, src) => self.place_assign(dst, src, live_after),
            // Transparent: a region scope does not change rc ownership (region
            // lifetime is a separate axis); its body's result is moved out.
            ExprKind::RegionRun(x) => self.place(x, live_after),
            ExprKind::Closure(c) => self.place_closure(e.id, &c, live_after),
            ExprKind::ClosureCall { target, args } => {
                self.place_closure_call(target, args, live_after)
            }
        }
    }

    /// Navigate the base of a [`Proj`](ExprKind::Proj) / [`Assign`](ExprKind::Assign)
    /// dst — a borrow, so the root is not consumed in place. Nested projections
    /// are pure navigation; a `Var` root is dropped (early, at `drop_anchor`) if
    /// dead after; a temporary root is placed and dropped (consumed).
    fn navigate_base(&mut self, e: &'tcx Expr<'tcx>, drop_anchor: ExprId, live_after: &VarSet) {
        match e.kind {
            ExprKind::Proj(inner, _) => self.navigate_base(inner, drop_anchor, live_after),
            ExprKind::Var(s) => {
                if self.rr.is_rr(e.ty) && !live_after.contains(s) {
                    self.add_after(drop_anchor, RcOp::Drop(s));
                }
            }
            _ => {
                self.place(e, live_after);
                if self.rr.is_rr(e.ty) {
                    self.add_after(drop_anchor, RcOp::DropValue(e.id));
                }
            }
        }
    }

    /// `dst->field := src`: store `src` into a field of `dst`.
    ///
    /// An assignment target is always a **flex regional** record — mutation is
    /// only legal on a mutable, region-local value, and such a value is not RR
    /// (it is freed with its region, never individually rc'd). So neither `dst`
    /// nor its root takes any rc op, and the old field value is *not* dropped
    /// here (flex/regional field links are reclaimed with the parent / region,
    /// not by a per-store dec). The only work is to place `src`, moved into the
    /// field: a rigid/managed `src` is inc'd like any other move into an owning
    /// slot.
    fn place_assign(&mut self, dst: &'tcx Expr<'tcx>, src: &'tcx Expr<'tcx>, live_after: &VarSet) {
        debug_assert!(
            !self.rr.is_rr(dst.ty),
            "assign dst must be a flex regional (non-RR) record",
        );
        self.place(src, live_after);
    }

    /// Creating a closure consumes its captured outer vars (dup any still live
    /// after), and analyzes the body as its own owned scope (captures + params
    /// owned on entry, result moved out).
    fn place_closure(&mut self, id: ExprId, c: &mir::ClosureExpr<'tcx>, live_after: &VarSet) {
        for &(v, ty) in c.captures {
            if self.rr.is_rr(ty) && live_after.contains(v) {
                self.add_before(id, RcOp::Dup(v));
            }
        }
        let owned = c.captures.iter().chain(c.params.iter());
        self.analyze_owned_body(c.body, owned);
    }

    /// `target(args)`: the closure value and every argument are consumed, in
    /// left-to-right evaluation order (a value reused later is dup'd).
    fn place_closure_call(
        &mut self,
        target: &'tcx Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
        live_after: &VarSet,
    ) {
        let mut target_after = self.free_slice(args);
        target_after.union_with(live_after);
        self.place(target, &target_after);
        self.place_args(args, live_after);
    }

    /// The union of the free sets of a slice of expressions.
    fn free_slice(&mut self, es: &'tcx [Expr<'tcx>]) -> VarSet {
        let mut s = VarSet::default();
        for e in es {
            s.union_with(&self.free(e));
        }
        s
    }

    /// Place a body whose `owned` vars (function params, or closure captures +
    /// params) are owned on entry: the body under an empty continuation, then an
    /// entry drop for any owned RR var the body never uses ([Param-Unused]).
    fn analyze_owned_body(
        &mut self,
        body: &'tcx Expr<'tcx>,
        owned: impl Iterator<Item = &'tcx (VarId, Ty<'tcx>)>,
    ) {
        self.place(body, &VarSet::default());
        let body_free = self.free(body);
        for &(v, ty) in owned {
            if self.rr.is_rr(ty) && !body_free.contains(v) {
                self.add_before(body.id, RcOp::Drop(v));
            }
        }
    }

    /// Place a `match`. See the module "Match" section for the model. The
    /// scrutinee is borrowed; each leaf dups its used RR fields, drops the
    /// scrutinee if it is consumed (dead after the match), and drops any
    /// dead-after outer var it does not use (N-way [If-Reconcile]).
    fn place_match(
        &mut self,
        scrut: &'tcx Expr<'tcx>,
        tree: &DecisionTree<'tcx>,
        live_after: &VarSet,
    ) {
        let tree_free = self.free_tree(tree);
        let mut scrut_after = tree_free.clone();
        scrut_after.union_with(live_after);

        // Decide how the scrutinee value is settled by the match.
        let (consumed, held) = match scrut.kind {
            // A `Var` scrutinee is borrowed in place (no op); it is consumed iff
            // dead after the whole match.
            ExprKind::Var(s) if self.rr.is_rr(scrut.ty) => {
                (!scrut_after.contains(s), Some(RcOp::Drop(s)))
            }
            // A non-RR scrutinee (matching an int/bool/…): nothing to settle.
            ExprKind::Var(_) => (false, None),
            // A temporary: its internals are placed here; the owned result is
            // consumed by the match and dropped (by anchor) in each leaf.
            _ => {
                self.place(scrut, &scrut_after);
                let held = self.rr.is_rr(scrut.ty).then_some(RcOp::DropValue(scrut.id));
                (held.is_some(), held)
            }
        };

        // Outer RR vars that die within the match: settled per leaf.
        let mut must_settle = tree_free;
        must_settle.subtract(live_after);
        let must_settle: Vec<VarId> = must_settle.iter().collect();

        self.place_tree(tree, live_after, consumed, held, &must_settle);
    }

    fn place_tree(
        &mut self,
        tree: &DecisionTree<'tcx>,
        live_after: &VarSet,
        consumed: bool,
        held: Option<RcOp>,
        must_settle: &[VarId],
    ) {
        match *tree {
            DecisionTree::Uncovered | DecisionTree::Unreachable => {}
            DecisionTree::Leaf { body, bindings } => {
                self.place_leaf(body, bindings, live_after, consumed, held, must_settle);
            }
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => self.place_guard(
                bindings,
                guard,
                success,
                failure,
                live_after,
                consumed,
                held,
                must_settle,
            ),
            DecisionTree::Switch { cases, .. } => {
                for sub in subtrees(&cases) {
                    self.place_tree(sub, live_after, consumed, held, must_settle);
                }
            }
        }
    }

    /// A guarded arm. The scrutinee stays live across the guard (its `failure`
    /// branch re-tests it), so the scrutinee drop stays in the terminal leaves.
    /// The guard's pattern bindings are borrow-extracted (dup at the guard); a
    /// binding used only in the guard is consumed there, one used in a branch is
    /// settled in that branch (success consumes, failure drops) — so it joins the
    /// sub-trees' `must_settle`.
    #[allow(clippy::too_many_arguments)]
    fn place_guard(
        &mut self,
        bindings: &'tcx [(VarId, &'tcx [u32])],
        guard: &'tcx Expr<'tcx>,
        success: &'tcx DecisionTree<'tcx>,
        failure: &'tcx DecisionTree<'tcx>,
        live_after: &VarSet,
        consumed: bool,
        held: Option<RcOp>,
        must_settle: &[VarId],
    ) {
        let guard_free = self.free(guard);
        let mut below = self.free_tree(success);
        below.union_with(&self.free_tree(failure));

        // Extract (dup) each RR binding used in the guard or a branch. Membership
        // in a free set already implies RR, so no type lookup is needed.
        let mut extracted = Vec::new();
        for &(y, _) in bindings {
            if guard_free.contains(y) || below.contains(y) {
                self.add_before(guard.id, RcOp::Dup(y));
                extracted.push(y);
            }
        }

        // The guard runs before either branch; everything used below (or in the
        // match's continuation, or still owed a settle) is live across it.
        let mut guard_after = below.clone();
        guard_after.union_with(live_after);
        for &v in must_settle {
            guard_after.insert(v);
        }
        self.place(guard, &guard_after);

        // Bindings that survive the guard into a branch must be settled there.
        let mut sub_settle = must_settle.to_vec();
        for &y in &extracted {
            if below.contains(y) {
                sub_settle.push(y);
            }
        }
        self.place_tree(success, live_after, consumed, held, &sub_settle);
        self.place_tree(failure, live_after, consumed, held, &sub_settle);
    }

    fn place_leaf(
        &mut self,
        body: &'tcx Expr<'tcx>,
        bindings: &'tcx [(VarId, &'tcx [u32])],
        live_after: &VarSet,
        consumed: bool,
        held: Option<RcOp>,
        must_settle: &[VarId],
    ) {
        let body_free = self.free(body);
        // A used binding of the whole scrutinee (path `[]`) takes ownership of the
        // borrowed root: moved if the scrutinee is consumed, else dup'd.
        let has_whole = bindings
            .iter()
            .any(|&(y, path)| path.is_empty() && body_free.contains(y));

        // 1. Field dups / whole-binding dup — used RR bindings only.
        for &(y, path) in bindings {
            if !body_free.contains(y) {
                continue; // unused (or non-RR) binding: no op
            }
            if path.is_empty() {
                if !consumed {
                    self.add_before(body.id, RcOp::Dup(y)); // borrowed root → alias
                }
                // consumed whole binding ⇒ move (no op)
            } else {
                self.add_before(body.id, RcOp::Dup(y)); // borrow-extract a field
            }
        }

        // 2. Drop the consumed scrutinee (after the field dups), unless a whole
        //    binding moved it out.
        if consumed
            && !has_whole
            && let Some(op) = held
        {
            self.add_before(body.id, op);
        }

        // 3. Reconciliation: settle dead-after outer vars this path does not use.
        for &v in must_settle {
            if !body_free.contains(v) {
                self.add_before(body.id, RcOp::Drop(v));
            }
        }

        // 4. The arm body, under the match's own continuation.
        self.place(body, live_after);
    }

    /// [If-Reconcile]: place the condition (live across both arms), then each arm
    /// under the shared continuation `L`, then settle one-sided ownership so both
    /// arms exit owning the same set — a var used in only one arm and dead after
    /// is moved on the arm that uses it and dropped on the arm that does not.
    fn place_if(
        &mut self,
        c: &'tcx Expr<'tcx>,
        t: &'tcx Expr<'tcx>,
        e: &'tcx Expr<'tcx>,
        live_after: &VarSet,
    ) {
        let used_t = self.free(t);
        let used_e = self.free(e);

        // The condition runs before either arm, so anything an arm needs is live
        // across it.
        let mut cond_live = used_t.clone();
        cond_live.union_with(&used_e);
        cond_live.union_with(live_after);
        self.place(c, &cond_live);

        self.place(t, live_after);
        self.place(e, live_after);

        // Reconciliation drops, ascending var order for determinism. `x ∉ L`
        // because a var in `L` is owned out of both arms (no drop); a var used in
        // both arms is moved (settled) in each.
        for x in used_e.iter() {
            if !used_t.contains(x) && !live_after.contains(x) {
                self.add_before(t.id, RcOp::Drop(x));
            }
        }
        for x in used_t.iter() {
            if !used_e.contains(x) && !live_after.contains(x) {
                self.add_before(e.id, RcOp::Drop(x));
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
        // [Discard]: a non-final statement whose RR result is neither bound
        // (`Let`) nor consumed leaves an owned value with no name — drop it by
        // anchor right after the statement.
        if !is_last && !matches!(s.kind, ExprKind::Let { .. }) && self.rr.is_rr(s.ty) {
            self.add_after(s.id, RcOp::DropValue(s.id));
        }
    }

    /// [Args]: each argument is consumed by the call/ctor; a var reused in a
    /// later argument is live across the earlier occurrence (so it gets a `dup`).
    fn place_args(&mut self, args: &'tcx [Expr<'tcx>], live_after: &VarSet) {
        let n = args.len();
        if n == 0 {
            return;
        }
        let mut afters = SmallVec::<[_; 4]>::with_capacity(n);
        afters.resize_with(n, VarSet::default);
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

/// The immediate sub-trees of a switch's cases, in source order.
fn subtrees<'tcx>(cases: &SwitchCases<'tcx>) -> SmallVec<[&'tcx DecisionTree<'tcx>; 8]> {
    let mut v = SmallVec::new();
    match *cases {
        SwitchCases::Int { cases, default } => {
            for (_, t) in cases {
                v.push(t);
            }
            v.push(default);
        }
        SwitchCases::Bool { if_true, if_false } => {
            v.push(if_true);
            v.push(if_false);
        }
        SwitchCases::Char { cases, default } => {
            for (_, t) in cases {
                v.push(t);
            }
            v.push(default);
        }
        SwitchCases::Ctor(arms) => {
            for t in arms {
                v.push(t);
            }
        }
        SwitchCases::String { cases, default } => {
            for (_, t) in cases {
                v.push(t);
            }
            v.push(default);
        }
        SwitchCases::Nullable { non_null, null } => {
            v.push(non_null);
            v.push(null);
        }
    }
    v
}

/// Every variable bound by a pattern anywhere in the tree (internal to the match;
/// excluded from its free set).
fn tree_bound<'tcx>(tree: &DecisionTree<'tcx>) -> VarSet {
    let mut s = VarSet::default();
    collect_bound(tree, &mut s);
    s
}

fn collect_bound<'tcx>(tree: &DecisionTree<'tcx>, s: &mut VarSet) {
    match *tree {
        DecisionTree::Uncovered | DecisionTree::Unreachable => {}
        DecisionTree::Leaf { bindings, .. } => {
            for &(y, _) in bindings {
                s.insert(y);
            }
        }
        DecisionTree::Guard {
            bindings,
            success,
            failure,
            ..
        } => {
            for &(y, _) in bindings {
                s.insert(y);
            }
            collect_bound(success, s);
            collect_bound(failure, s);
        }
        DecisionTree::Switch { cases, .. } => {
            for sub in subtrees(&cases) {
                collect_bound(sub, s);
            }
        }
    }
}

#[cfg(test)]
mod tests;
