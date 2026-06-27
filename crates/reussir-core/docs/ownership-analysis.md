# Ownership Analysis (Full MIR)

> Status: **design / proposed** (pm/07). This is a fresh Perceus-style analysis,
> not a port of the Haskell `Reussir.Core.Ownership` (which used a forward
> ledger + flux). We keep only its vocabulary (RR types, before/after
> annotations).

## 1. Goal & scope

Insert reference-counting operations — `dup` (`rc.inc`) and `drop`
(`rc.dec` / `ref.drop`) — onto the Full MIR so that **every resource-relevant
(RR) value is freed exactly once**, with `drop`s placed **as early as possible**
so the backend `TokenReuse` pass can pair a freed token with a same-layout
allocation.

In scope: **shared rc, rigid rc, regional values, closures, and any container
that transitively holds one of those** (the RR set). Out of scope here:

- emitting MLIR — that is codegen (pm/08), which *consumes* this analysis;
- the backend reuse pairing — already a backend pass; the frontend only has to
  emit `drop`s early and near allocations.

## 2. The ownership discipline

- **Owned calling convention.** A function owns its RR parameters (the caller
  transfers ownership in). It must consume-or-drop each owned RR value exactly
  once. The return value is moved out — owned by the caller, never dropped.
- **Last use is the pivot.** For a use of RR variable `x`:
  - last use → **move** (consume; no `dup`), `x` becomes dead;
  - non-last use → **`dup x`** before the use, keep ownership;
  - a var that goes dead without being consumed (bound-but-unused, or dead after
    a branch) → **`drop x`** at the earliest dead point.
- **Borrow ≠ own.** Reading a field (`Proj`) borrows from the parent; if the
  borrowed RR value escapes as owned, **`dup`** it — the parent still owns the
  original.

We start with the **owned** discipline only (no borrowed-parameter
optimization). That is the simplest correct base; borrow-passing is a later
refinement.

## 3. The RR-type predicate

```text
is_rr(ty) =
  match ty.kind():
    Record{def, args} =>                                  // capability ignored
        match record_table.shape(def, args).managed:      // Value | Shared | Regional
            Shared | Regional => true                     // rc-managed
            Value             => any field is_rr           // transitive
    Closure{..}      => true            // owns a captured environment
    Nullable(inner)  => is_rr(inner)
    Int | Float | Bool | …             => false
```

- **Two orthogonal axes.** A record's `Ty` *capability* records **regional value
  coloring** (`Regional`/`Flex`/`Rigid`, or `Irrelevant` for anything that does
  not participate — i.e. every non-regional record). That is a *different axis*
  from **rc-management** (is it heap/region reference-counted, or inline by
  value), which the elaborator stores in `ctxt::Record.default_cap`
  (`Value | Shared | Regional`). The capability cannot answer "is this rc'd": the
  dialect's `Capability` has no `Shared`/`Value` variant, and `[value]`/`[shared]`
  both elaborate to `Irrelevant` (`ty_eval.rs`).
- **So `is_rr` decides rc-management from the `RecordTable` for *every* record**
  (regional or not), keyed by the canonical (`Irrelevant`) record type — never
  from the capability. The table (`canonical-Ty → RecordShape { managed, fields }`)
  carries `default_cap` forward.
- Memoized per interned `Ty` (pointer-identity key — types are arena-interned);
  a provisional `false` guards the (illegal but defensively-handled) value-record
  field cycle.
- The pass takes the `RecordTable` as an explicit argument
  (`analyze_function(tcx, &Function, &RecordTable)`); the test harness builds it
  directly, and the real pipeline will populate it from `mono`'s record table
  when codegen wires the analysis in (pm/08). The MIR `Program` need not grow new
  ser/de surface for this.

## 4. Anchors: recording op positions on an immutable MIR

The MIR is immutable and `mir::Expr` (`{kind, ty, span}`) carries no id.

**Decision (locked): add `pub id: ExprId` to `mir::Expr`** (`ExprId(u32)`; the
type already exists in `semi::hir`).

- Stamped at the two allocation choke points — `mir/build.rs::expr_ref` and
  `mono.rs` — via a single monotonic counter on the build/mono context. One
  helper (`mk_expr`) becomes the only site that assigns ids.
- The textual printer **need not emit ids**; they are regenerated
  deterministically on parse, so round-trip soundness is preserved (ids are not
  semantic content).

Alternatives considered and rejected:

- **Arena pointer-identity** (`*const Expr` as key): zero IR change, but
  unidiomatic, undebuggable, and breaks if a node is ever copied / re-interned.
- **Traversal-order index** (the i-th visited expr): no field, but couples the
  analysis and codegen to walk in lock-step — fragile.

Explicit `ExprId` is robust, debuggable (`id → action` is printable), and
survives traversal changes. The cost is a contained build/mono ripple.

**Non-`Expr` positions.** Match arm bodies and `Let` are `Expr`s, so they anchor
directly. Operations with no natural `Expr` — dropping a variant shell at a
switch, dropping an unused parameter — attach to the nearest `Expr`'s
`before`/`after` list (arm-body `before`, function-body root). Match-internal
anchoring is resolved in the control-flow increment.

## 5. Output data model

A side-table — the MIR stays immutable; codegen does lookups.

```rust
pub struct OwnershipTable { actions: HashMap<ExprId, Action> }

pub struct Action { pub before: Vec<RcOp>, pub after: Vec<RcOp> }

pub enum RcOp {
    Dup(VarId),
    Drop(VarId),
    // grows later: DropField(VarId, Path), …
}
```

Timeline per expression: `[before ops] → evaluate expr → [after ops]`. Codegen
(`reussir-codegen::lower`) consults `table.get(expr.id)` around each node.

## 6. The algorithm

Two passes over each function body, pure (no MLIR):

**Pass A — free RR variables (bottom-up).** `free(e): BitSet<VarId>` = the RR
vars referenced in `e`, memoized per `ExprId`. Drives last-use detection.

**Pass B — placement (top-down, continuation liveness).** Threads `live_after`
sets through the tree and emits `Dup`/`Drop` per the rules below.

### 6.1 Last-use detection — structured backward liveness, one pass, no fixpoint

**Approach: a structured, syntax-directed backward pass — a *degenerate*
liveness dataflow that converges in a single sweep — rather than an iterative
CFG worklist.**

*Why one pass is exact.* A function body is a tree whose only control constructs
are `Seq`, `Let`, `If`, and `Match`. None of them loop: intra-function iteration
does not exist in the MIR — repetition is recursion through `Call`, which is a
dataflow **leaf** (it neither defines nor keeps the caller's locals alive beyond
its own argument list). The induced control-flow graph is thus a DAG, and
backward liveness over a DAG is exact after one reverse-topological visit; for a
structured tree, reverse-topological order *is* right-to-left / post-order. So we
never build an explicit CFG and never iterate to a fixpoint — distinguishing this
from general dataflow, which needs a worklist precisely because of back-edges.

*What we track.* Liveness of **named RR variables only** (parameters + `Let`
bindings). Unnamed intermediate RR values are owned temporaries threaded by the
tree itself — each is immediately consumed (arg/ctor position), bound (`Let`),
returned (moved), or discarded (a non-final `Seq` statement → `Drop`) — so they
never enter the liveness set.

*The quantity.* `live_after(e)` = the set of named RR vars used anywhere in `e`'s
dynamic continuation (everything that runs after `e`, within the function). A use
of `x` at `e` is its **last use** ⟺ `x ∉ live_after(e)`.

*Threading rules* (each child's `live_after` is its right-siblings' `free` sets
∪ the parent's `live_after`):

- `Seq([s₀..sₙ])` under live `L`: `live_after(sₙ) = L`;
  `live_after(sᵢ) = free(sᵢ₊₁) ∪ … ∪ free(sₙ) ∪ L`.
- `Let{x, value}` with continuation `k`: `live_after(value) = live(k)`; `x` is
  granted after `value`, so `x ∈ live(k)` ⟺ used later.
- `Call`/`Ctor`/`Variant{args=[a₀..aₖ]}` under `L`:
  `live_after(aᵢ) = free(aᵢ₊₁) ∪ … ∪ free(aₖ) ∪ L`. ⇒ a var reused in a later
  argument is live after the earlier occurrence, so the earlier one is **not** a
  last use and gets a `Dup` — the `foo(x, x)` case falls straight out.
- `If(c, t, e)` under `L`: `live_after(c) = free(t) ∪ free(e) ∪ L`;
  `live_after(t) = live_after(e) = L`.
- `Match(scrut, dt)`: like `If` over the arms;
  `live_after(scrut) = (⋃ free(armᵢ)) ∪ L`.

*Conditional (path-dependent) last use* is handled by branch reconciliation, not
a separate mechanism. Two canonical cases:

- **Dies on one path.** `let x; if c { use(x) } else { /* x unused */ }` with
  `x ∉ L`: `x ∈ free(then)`, `x ∉ free(else)` ⇒ `Drop(x)` in the `else` arm; in
  `then`, `use(x)` sees `x ∉ live_after (= L)` ⇒ move. Both paths leave `x`
  settled.
- **Used in a branch *and* after.** `let x; if c { use(x) } else { use(x) };
  use(x)` with `x ∈ L`: inside each arm `live_after(use x) = L ∋ x` ⇒ **not** a
  last use ⇒ `Dup(x)`; the trailing `use(x)` has `x ∉ live_after` ⇒ move. The rc
  stays balanced (enter 1 → arm `dup` 2 → arm-use 1 → trailing-use 0), and only
  the taken arm's `dup` runs.

*Representation.* `VarId`s are dense per function, so live sets are fixed-width
**bitsets** (union/diff/membership in O(words)); `free(e)` is computed once
bottom-up and memoized per `ExprId`; `live_after` is threaded top-down and only
materialized where an op is emitted.

### 6.2 Placement rules

`place(e, live_after)` reads off the live-after sets above and emits, per MIR
form:

| Form | Rule |
| --- | --- |
| `Var(x)` (RR) | `x ∈ live_after` ⇒ `before: Dup(x)`; else move (last use). |
| `Seq([s0..sn])` | right-to-left; `live_after(sᵢ) = ⋃_{j>i} free(sⱼ) ∪ outer`. A `Let`-bound `x` unused later ⇒ `Drop(x)` in the let's `after`. A non-final statement producing an unbound/unconsumed RR value ⇒ `Drop` its result. `sn` is moved out. |
| `Let{x, value}` | `value` placed with `x`'s downstream liveness; binding grants ownership of `x`. |
| `If(c,t,e)` / `Match` | **branch reconciliation** — both arms must exit owning the same set. Per arm: owned-on-entry but unused-in-arm and not-needed-after ⇒ `Drop` inside that arm; needed-after but consumed-in-arm ⇒ `Dup` as needed. Balanced ownership at the join. |
| `Call`/`Ctor`/`Variant`/`ClosureCall` args | each arg position consumes its value; a var in ≥2 arg positions ⇒ `Dup` on all but the last occurrence (left-to-right). |
| `Proj(e, path)` | borrow; if the projected RR field escapes as owned, `before: Dup` it. Parent's drop is unaffected (dropped at its own last use). |
| `Closure{captures}` | creating the closure consumes its captures (dup any still live afterwards). Closure drop-glue is the backend's. |
| `RegionRun` / regional | result owned; detailed region lifecycle deferred. |

**Early-drop / reuse.** The rules above already drop at last-use / dead-branch
(not deferred to function end), which is "early". A dedicated
**drop-specialization** refinement — push `drop` into match arms and just after a
destructuring `Proj`, so the freed token sits right before a sibling allocation —
is staged separately; the backend `TokenReuse` does the actual pairing, so the
frontend only needs the dec emitted at the destructure point.

## 7. Operational semantics (sequent calculus)

Judgment `L ⊢ e ⤳ 𝒜`: "with continuation-live set `L`, expression `e` annotates
to actions `𝒜`." Representative rules (the full set ships as module
doc-comments, one sequent per MIR form, checkable against the code):

```text
x ∈ L                                   x ∉ L
─────────────────── [Var-Shared]        ─────────────────── [Var-LastUse]
L ⊢ x ⤳ {before: dup x}                 L ⊢ x ⤳ {}

L ⊢ e ⤳ 𝒜     x ∉ free(rest)
──────────────────────────────────────────────── [Let-Dead]
L ⊢ (let x = e; rest) ⤳ 𝒜 ⊕ {after: drop x}

L_t = L│t     L_e = L│e     Δ owned at join
──────────────────────────────────────────── [If-Reconcile]
L ⊢ if c t e ⤳ balance(Δ, t, e)
```

### 7.1 Relation to Tree Borrows (design inspiration)

Tree Borrows (Villani, Hostert, Dreyer, Jung; PLDI 2025 — the successor to
Stacked Borrows as Rust's aliasing model) gives each *(location, reference)* a
**permission state** in a small lattice and transitions it on every access,
propagating the effect along a **tree of derivations** — each reference is a node
whose parent is the reference it was derived from. Its states:

- **Reserved** — a fresh `&mut` not yet written; tolerates *foreign* reads.
- **Active** — written through; enforces uniqueness, so a foreign read demotes it.
- **Frozen** — read-only (shared).
- **Disabled** — any further access is undefined behavior.

Transitions: a *child* write `Reserved → Active`; a *foreign* read
`Active → Frozen`; a *foreign* write `* → Disabled`.

We borrow three *ideas*, not the model:

1. **Per-entity state machine.** Each owned RR variable carries an ownership
   state, transitioned on *use* events — the analysis analogue of TB
   transitioning a permission on *access* events.
2. **Propagation along a tree with reconciliation at joins.** TB propagates an
   access to a node's relatives; we propagate liveness backward through the
   expression tree and **reconcile sibling control-flow branches** at each
   `If`/`Match` join (§6.1).
3. **Terminal ("settled") states must not be re-used.** TB makes a
   post-`Disabled` access UB; we make the corresponding situation
   *unrepresentable* — the analysis guarantees every owned var reaches a settled
   state exactly once per path, so the runtime is always well-defined.

Our lattice is far simpler than TB (no aliasing, no reborrow stack, no
protectors, no interior-mutability variants):

```text
        bind / param-in
              │
              ▼
           Owned ──── non-last use ──▶ dup (rc++), stay Owned
            │   │
       last │   │ goes dead
      use   ▼   ▼ (no further use)
        Consumed   Dropped          ── both terminal ("settled")
            ▲
   return = Consumed (moved to caller, never dropped)
```

The crucial difference: TB is a **runtime semantics** (it *defines* UB, checked
dynamically by Miri); ours is a **static rewrite** that *inserts* the rc ops so
no such UB can arise. There is no "foreign access" notion — we never alias, we
transfer ownership. (A future borrowed-parameter optimization would add a
read-only `Borrowed` state, the rough analogue of `Frozen` — no drop
responsibility.)

## 8. Placement in the tree

`crates/reussir-core/src/full/ownership.rs` — pure, no MLIR dependency (fits the
`reussir-core::full` "no MLIR" rule). Entry points:

- `ownership::analyze(&mir::Program) -> OwnershipTable`
- `ownership::analyze_function(&Function, &RecordTable) -> …` (for unit tests)

Methods over free functions where it reads naturally.

## 9. Testing (pure, no execution)

The analysis is a pure `&Program → OwnershipTable`, so it is unit-tested without
any MLIR or runtime. The framework has three pieces: a way to **construct** MIR,
a way to **assert** on the result, and a generic **safety-net** property.

### 9.1 Construction — an in-arena `MirBuilder`

A test-support `MirBuilder` builds `mir::Function` / `Program` directly in an
arena, sharing the production `ExprId`-stamping path so anchors match real
builds. Fluent constructors mirror the MIR forms:

```rust
let mut b = MirBuilder::new(&tcx);
let x = b.param("x", rc_ty);                 // an RR parameter
let y = b.local();
let body = b.seq([
    b.let_(y, b.call(foo, [b.var(x)])),      // let y = foo(x);
    b.var(y),                                //   y            (returned)
]);
let f = b.function("f", [x], rc_ty, body);
```

Types are **real interned `Ty`s** (`mk_record(.., Shared)` for an rc, `i64` for a
scalar, `mk_nullable`, …) backed by a minimal record table, so `is_rr` — the
predicate everything keys off — is exercised for real, not stubbed.

*Why a builder over textual MIR:* it is precise, **parser-independent** (analysis
tests don't break on unrelated ser/de gaps), and each increment can target
exactly the forms it adds. Textual-MIR-driven tests are layered on later for
round-trip coverage once the parser covers every form — at which point the same
cases can be expressed as `.mir` strings.

### 9.2 Assertion — annotated rendering + structured probes

An **annotated pretty-printer** walks the body together with the
`OwnershipTable`, weaving each emitted op into the tree:

```text
fn f(x: Rc) -> Rc {
  let y = foo(x);     // x: last use → move
  y                   //    returned → move
}                     // (no drops: everything moved)
```

Tests compare this rendering against an inline expected snapshot — readable, and
doubling as live opsem documentation. Pinpoint cases also use structured probes:
`table.before(id)` / `table.after(id)` assert the exact `Dup`/`Drop` set at a
given node.

### 9.3 Corpus

One constructed function per rule, each with its expected annotated body:

1. one use → `Drop` after last use;
2. two uses → one `Dup` before the first;
3. `if` / `match` arm reconciliation (var live in one arm only);
4. borrow-only `Proj` → parent still dropped, field dup'd;
5. returned var → moved, **no** drop;
6. nested value-record transitively holding rc → treated RR;
7. unused RR parameter → dropped at entry.

### 9.4 Property check — a generic safety net

Beyond goldens, a **balanced-rc checker** abstractly interprets the annotated
tree: along every control path, a `dup` is `+1`, a consuming use or a `drop`
settles the unit. It asserts that every owned RR variable reaches a settled state
**exactly once per path** — no double-settle, nothing left live at `return`
except the moved result. This catches placement bugs the hand-written goldens
miss, and is cheap to run over every corpus case.

Execution / ASAN-LSAN gating arrives in **pm/08**, when codegen consumes the
table.

## 10. Incremental stack (bottom-up PRs)

1. **Anchors + foundation — landed (`full::ownership`).** `ExprId` on
   `mir::Expr` (+ build/mono stamping via a shared `ExprIdGen`, printer
   unaffected), `is_rr` + `RecordTable`, `OwnershipTable` / `RcOp`, the
   **`MirBuilder` test harness + annotated renderer + balanced-rc checker** (§9),
   and the **linear core** (`Var` / `Let` / `Seq` / `Call` / `Ctor` / `Variant` /
   `NullableCall`) with the §9.3 corpus. Deferred forms (`If`/`Match`/`Proj`/…)
   `unimplemented!` loudly rather than miscount.
2. **Control flow — `If` landed.** `If` branch reconciliation (settle one-sided
   ownership so both arms exit owning the same set), multi-use `Dup` (already in
   increment 1), and discarded-result drops via `RcOp::DropValue(ExprId)` for an
   unconsumed non-`let` `Seq` statement. **`Match` reconciliation over decision
   trees** (switch/guard/bindings) is split into its own follow-up — it overlaps
   the container increment (destructuring `Proj`/partial moves).
3. **Containers:** `Proj` borrow-dup, `Nullable`, `Variant`, transitive
   value-records.
4. **Closures & regions:** capture consumption, region-run lifecycle.
5. **Drop-specialization** (optional, reuse-targeted).

Then **pm/08:** codegen consumes `OwnershipTable` → emits `rc.inc` / `rc.dec` /
`ref.drop`, gated by ASAN/LSAN execution tests.

## Decisions

- **Anchor:** add `ExprId` to `mir::Expr`, stamped at the build/mono allocation
  choke point via a shared `ExprIdGen` (§4). *Locked, landed.*
- **Owned-only convention** to start (no borrowed-parameter optimization); the
  simplest correct base, refined later. *Resolved.*
- **RR set:** rc-management (`Shared`/`Regional`) and closures are always rc;
  `Value` records are rc iff transitively so; `Nullable` follows its inner.
  Rc-management comes from the `RecordTable` for *every* record — the `Ty`
  capability (regional coloring) is an orthogonal axis and is not consulted
  (§3). *Resolved.*
- **Increment 1 = linear core + `is_rr` + harness.** *Resolved, landed.*

## Open questions

1. **`RecordTable` provenance for the real pipeline.** Increment 1's table is
   built by the test harness. When codegen consumes the analysis (pm/08), build
   it from `mono`'s `records: &FxHashMap<DefId, Record>` (which has `default_cap`
   + fields) — confirm that hand-off rather than enriching `mir::Program`.
2. **Anonymous-result drops.** A non-`let` `Seq` statement of RR type whose
   result is discarded has no `VarId` to `Drop`; the var-only `RcOp` can't name
   it. Lands with the control-flow increment (likely a `DropValue(ExprId)` op).
