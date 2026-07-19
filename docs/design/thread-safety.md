# Thread safety: the `Sync` discipline, `Arc`, and cells

Status: **design accepted, partially implemented.** The frontend has the `Arc`
type and the cell surface (`Cell`/`RefCell`); the MLIR dialect has the full
cell-kind lattice and the atomic box axis; the `Sync` structural rules, the
per-member atomic axis, and `Arc` lowering are not implemented yet. Sections marked
*(open)* record decisions that are deliberately deferred.

This document is the single source of truth for the rules. When code and this
document disagree, one of them is a bug — file it.

## 1. The model: atomicity is a property of the box

Every rc-managed object lives in a box `rc<T, capability, atomicKind>` where
`atomicKind ∈ {normal, atomic}`. A *normal* box bumps its refcount with plain
loads/stores; an *atomic* box uses atomic ops (`rc.fetch_sub` with acq-rel
ordering on the decrement path; blind count stores are forbidden —
`ReussirRcSetOp` rejects atomic boxes).

Thread safety in Reussir is therefore a property of **which boxes a thread can
reach**, and the entire discipline reduces to one invariant:

> **No normal-count box, no plain/exclusive cell, and no regional object is
> ever reachable from more than one thread.**

`Arc` and the sync cell kinds are the two surface mechanisms that mark a box
atomic; the `Sync` predicate is the type-level check that the invariant holds
transitively.

Unlike Koka/Perceus (which re-colors boxes *dynamically* when a value becomes
thread-shared, and is sound only because Koka is pure), Reussir's coloring is
**static and set at birth**: an arc is never re-colored from an existing plain
shared value, and the type system — not a runtime graph walk — guarantees the
interior is safe. This is required, not a preference: a language with writable
cells cannot use Koka's dynamic argument, because a cell reachable from a
shared box is a channel through which a still-thread-local value could later be
smuggled into shared space. A type-level bound on what cells may *ever* hold
closes that channel; a creation-time value scan does not.

## 2. The `Sync` trait

There is **one** thread-safety predicate, the auto trait `Sync`:

> `Sync(τ)` — a value of type `τ` may be reached (aliased, read, cloned,
> stored, dropped) from any thread, concurrently with other threads doing the
> same.

The name is borrowed from Rust, but the meaning is deliberately stronger:
Reussir's `Sync` subsumes **both** of Rust's `Send` and `Sync` (§2.1). When
this document refers to Rust's weaker predicate it always says "Rust's
`Sync`."

### 2.1 Why one trait and not Rust's Send/Sync pair

Rust needs two predicates because it can express *exclusive transfer*: moving a
`Cell<T>` to another thread is safe (`Send`) even though sharing it is not
(`!Sync`), and the borrow checker proves the move leaves no aliases behind.
Reussir cannot express that weaker capability: rc handles are freely duplicable
within a thread and carry no linearity, so "this value has no remaining aliases
on the origin thread" is not provable at the type level. Any value that crosses
a thread boundary must be assumed to be simultaneously reachable from both
sides — transfer collapses into sharing: Rust's `Send` and `Sync` coincide,
and the single surviving predicate keeps the name `Sync`.

Two consequences we accept explicitly:

- In Rust, `Mutex<T>` is `Sync` given only `T: Send`. Reussir's mutex cell
  demands the stronger `Sync(τ)` (§4.3) — the Send-only row of Rust's table is
  unreachable without linearity, so nothing is actually lost.
- If uniqueness/linear tracking ever lands (a proven-unique rc, a
  consume-once closure), a transfer-only predicate weaker than `Sync` (i.e. a
  genuine `Send`) becomes meaningful and can be added *then* without breaking
  this scheme. The Phase-0 `Send` trait declaration in
  `semi/traits/builtins.rs` is retired; `Sync` remains as the sole predicate,
  carrying the strengthened meaning above.

### 2.2 Structural rules

Base cases:

| Type | `Sync`? | Reason |
| --- | --- | --- |
| primitive scalars (`iN`/`fN`/`bool`/`char`/unit) | yes | no identity, copied by value |
| `str` (global lifescope) | yes | immortal immutable bytes |
| `str` (local lifescope) | no | borrows a scope on the origin thread |
| `Arc<X>` (WF per §3) | **yes** | atomic box; WF guarantees the interior |
| plain `[shared]` record / array / closure box | **no** | normal refcount races (`Rc: !Send` in Rust terms) |
| atomic / mutex / flatlock / rwlock cell | **yes** | born in an atomic shared box, §4 |
| plain (`Cell`) / exclusive (`RefCell`) cell | **no** | unsynchronized store / non-atomic in-use flag |
| regional (`flex` / `rigid` / `field`), region handles, views, refs, holes, cctx | **no** | §6 |

Structural propagation (an auto trait: derived, never written by the user):

- `[value]` record: `Sync` iff every member type is `Sync`.
- `Nullable<P>`: `Sync` iff `Sync(P)`.
- Generic parameters: `Sync` obligations are deferred and re-checked at
  monomorphization, exactly like `Arc`'s inner check today
  (`mono.rs::check_arc_inners`).

Note carefully what is *absent*: a `[shared]` record is never `Sync` **as a
bare type**, no matter what its members are. Shareability of an rc object is
carried by the coloring of its box (`Arc<X>`), not by the nominal type `X`.
That stratification is the next section.

Diagnostics requirement: when a type fails `Sync`, the error must name the
offending member path ("`Foo` is not `Sync` because member `bar.baz` is a
plain `[shared]` record"), not just the root type. Rust's auto-trait errors
are famously non-local; we should not reproduce that.

## 3. `Arc` well-formedness: coloring one box, requiring `Sync` inside

`Arc<X>` is the same nominal `X` behind a box whose refcount is atomic. Reads
(projection, matching) see through the coloring; only the rc discipline
differs. The coloring is established exactly once, at construction — an
existing plain value is never promoted.

The well-formedness rule is stratified: **`Arc` supplies atomicity for exactly
the one box it wraps; everything strictly inside must already be
self-sufficiently thread-safe.**

```
WF(Arc<X>)          where X is a [shared] record
    ⟺  Sync(member_i′)  for every member type of X (every arm, for variants),
        where member_i′ re-colors a member in X's recursive group as its own
        Arc (§3.3) and is member_i unchanged otherwise

WF(Arc<Array<T>>)   ⟺  Sync(T)

WF(Arc<(A₁,…,Aₙ) -> R>)   ⟺  Sync(Aᵢ) for all i        (R unconstrained, §3.2)
```

Everything else is rejected as an `Arc` inner, matching
`ty_eval.rs::arc_inner_rejection` today: `[value]` records (not an rc box —
embed them; their members are checked via `Sync` propagation), `[regional]`
records, cells (§4.1), `Nullable`, scalars, and nested `Arc`.

The canonical example of the stratification: let `FooBase` be a `[shared]`
record whose members are all `Arc<…>`, sync cells, or `Sync` value types.
Then:

- `FooBase` itself is **not** `Sync` — its own box count is normal, and two
  threads cloning the same `FooBase` handle race that count.
- `Arc<FooBase>` is well-formed and **is** `Sync` — the arc makes the outer
  count atomic, and the members were already safe.

Consequences:

1. **Gradual atomicity.** A single-threaded `FooBase` holding `Arc` members
   pays atomic ops only on those members, and is promoted to a shareable value
   by constructing it as `Arc<FooBase>` — no restructuring, no deep copy.
2. **No runtime graph walk.** Arc construction never traverses the object
   graph; the WF check already forced the interior to be atomic where it needs
   to be. Contrast Verona's `freeze` (a deep runtime operation) and Koka's
   dynamic share-marking.
3. **Both halves of Rust's `Arc` bound are covered by `Sync`.** In Rust,
   `Arc<T>` needs `T: Sync` (concurrent readers) *and* `T: Send` (the last
   drop runs on an arbitrary thread). Reussir's `Sync` is the conjunction:
   every box the drop glue of an arc can reach is atomic, so a final
   decrement on any thread is safe; every value a reader can clone out is
   `Sync`, so concurrent reads are safe.

### 3.1 Reads through the coloring

Projection and matching on `Arc<X>` see `X`'s members. A member that is itself
an rc object comes out **with its own coloring**: a declared `Arc<Y>` member
projects as `Arc<Y>`, and a bare same-group member of an arc'd recursive
record projects *promoted* (§3.3). Representationally, an `Arc<Y>` member
lowers to the explicit member type `rc<Y, shared, atomic>` — the one rc
member form the dialect admits, because the container's capability cannot
derive it: the container may be a normal, thread-local box while the member's
count is shared with other threads (its drop glue must decrement atomically
even though the container's own count is plain). `Arc` members are legal in
shared and `[value]` records; a `[regional]` record rejects them **for now
(not implemented)** — nothing in the current machinery is known to break,
but the interaction with the deferred sync-regional design (§6) is
deliberately left undecided until that design is written down.

### 3.2 Arc closures: type-level WF plus a creation-site capture check

Reussir closures are partial-application objects: `closure.apply` writes each
argument into the payload after the closure is created, `closure.eval` consumes
a reference, and a non-unique closure is cloned (vtable `clone` acquires every
payload slot) before in-place application. For an arc'd closure this means:

- it can be **called from many threads simultaneously** — calling is shared
  access (Rust: needs `F: Sync`, strictly more than the `F: Send` that
  `thread::spawn` demands for its one-shot transfer);
- clone and drop run `acquire`/`drop` on the captures from arbitrary threads
  concurrently, so every payload slot must hold an atomically-counted or
  count-free value;
- **capture time is not one point**: every later `apply` adds a capture.

A creation-time check alone is therefore insufficient. The rule has two parts
that together cover every payload slot statically:

1. **Type-level WF**: `Arc<(A₁,…,Aₙ) -> R>` requires `Sync(Aᵢ)` for every
   argument type. Any future `apply` goes through the closure's type, so
   applied arguments are `Sync` by construction — no per-apply flow check.
2. **Creation-site check**: at the point where a closure receives the arc
   coloring, every value captured from the environment must be `Sync`
   (Swift `@Sendable`'s capture rule, upgraded from "sendable" to our single,
   stronger predicate).

`R` is unconstrained: the result materializes on the evaluating thread, which
owns it outright; it never crosses a thread by virtue of the closure being
shared. (If a future combinator does move results across threads, *that*
combinator bounds `R`, not the closure type.)

A bare (non-arc) closure remains `!Sync` like any plain shared box, with no
constraint on captures or argument types — single-threaded code pays nothing.

*(future)* A linear, consume-once closure kind could soundly weaken the
capture bound to a transfer-style check (Rust `FnOnce` + `spawn`, Futhark's
"a consuming function may not be called twice" restriction), but that requires
linearity the type system does not have; out of scope here.

### 3.3 Recursive types: color propagation through the recursive group

The flat WF rule alone cannot express an arc'd recursive structure. In

```
enum List<T> { Nil, Cons(T, List<T>) }
```

the `Cons` tail is a bare `[shared]` box — never `Sync` — so a literal reading
of §3 makes `Arc<List<T>>` permanently ill-formed. The unsoundness it guards
against is real (an atomic head over a non-atomic spine), but the *shareable
list as such* is not unsound; the declaration simply has no way to say "under
`Arc`, this link is also `Arc`". Rust answers with duplication (an `Rc` list
and an `Arc` list are unrelated types, one per pointer flavor); because our
`Arc` is a coloring of the same record rather than a wrapper type, we can do
better.

**Rule (decided 2026-07-19): `Arc<X>` selects the atomic instantiation of
`X`'s whole recursive group.** When wf-checking `Arc<X>`, a member whose type
belongs to `X`'s SCC — the same def, or a mutually recursive partner
(`Tree`/`Forest`) — is checked *as its arc'd version* instead of bare; the
coinductive cycle rule then closes the recursion. Equivalently: the promotion
rewrites the declaration into what would be written by hand
(`Cons(T, Arc<List<T>>)`), one declaration serving both worlds:

- bare `List<T>`: the fast thread-local list, normal refcounts everywhere;
- `Arc<List<T>>`: the same declaration with every spine link atomic.

Promotion is **automatic** — no `Self` keyword or per-member opt-in. Nothing
expressible is lost: under static coloring, a bare same-group member behind an
`Arc` could never be legal (it would have to be `Sync`), so for such a member
the only possible meanings are *promoted* or *rejected*; auto-promotion turns
a permanent error into the one meaning that can exist. An explicit-`Self`
surface was considered and set aside: it cannot name a mutual-recursion
partner, so the SCC rule is needed underneath regardless.

Members *outside* the group are untouched — the §3 stratification still
demands they be independently `Sync` (already `Arc`'d, sync cells, `Sync`
value types). `Cons`'s head `T` is bounded exactly as before.

**The colored world is closed; conversion is an explicit rebuild.**
`Arc<List<T>>::Cons{h, t}` requires `t : Arc<List<T>>` — an atomic list is
built atomic from `Nil` up — and matching an `Arc<List<T>>` binds the tail at
`Arc<List<T>>`. There is no implicit demotion, and no implicit promotion of an
existing bare list: converting one is an O(n) deep rebuild, which stays
user-written (a dedicated deep-copy operation may be added later, but is not
committed here). The cost profile is the one already accepted in §1: an arc'd
spine pays atomic refcounting from birth, where Koka would pay only after
actual sharing.

**Representation: the whole-subtree rule.** Atomicity needs *no* record
encoding at all — no per-member flag array, no colored record instances, no
dual mangling. The `Sync` discipline guarantees nothing nonatomic is ever
stored inside an atomically counted box, so a member link's atomic kind is
the **join of what the member type writes and what the parent context
inherits**: the box's `rc<…, atomic>` floods its discipline down through
`rc.borrow`/`ref.project` (interior references carry the parent's atomic
kind), deriving every bare shared link, nullable wrapper, and value-record
interior atomically — one `List` record type serves both colorings, and
`Arc<X>` is `rc<X's own record, shared, atomic>` exactly as before. The only
written atomicity is the `Arc` *island in normal space* (§3.1's explicit
`rc<…, shared, atomic>` member type). Outlined drop/acquire glue forks per
atomic kind (`drop_in_place_atomic…`), since the glue's body derives from
its argument reference's kind.

One consequence of context-carried coloring: a **`[value]` intermediate**
containing a bare same-group link cannot promote — loaded out of the box by
value, nothing would carry its coloring, so its drop glue would release the
link at the wrong discipline. The checker therefore folds value-record
members outside the promotion group; such shapes are expressed with an
explicit `Arc` member inside the value record instead. (Variant *arms* are
the enum's own members and promote normally — `Cons`'s tail is the canonical
case.)

## 4. Cells

The MLIR cell lattice (`CellKind`): `plain`, `exclusive`, `atomic`, `mutex`,
`flatlock`, `rwlock`. A cell is always managed through `rc<cell<T, kind>>`.
The frontend currently surfaces only `Cell` (plain) and `RefCell` (exclusive);
the sync kinds are reachable from MLIR and will get surface types with the
bounds below.

### 4.1 `Arc` never applies to a cell

`Arc<Cell<…>>` (any kind) is rejected — this is already enforced
(`arc_inner_rejection`: "a cell"). Rationale: the atomicity of a cell is part
of its **kind**, not a coloring you stack on top.

- Sync-kind cells don't need `Arc`: the dialect *requires* them to live in an
  atomic shared box (`CellType::requiresAtomicSharedBox`, verified by
  `RcType::verify`) — they are born with exactly the discipline `Arc` would
  add, so `Arc` on them would be a redundant second discipline.
- Plain/exclusive cells must not get `Arc`: an atomic refcount on the box does
  nothing for the unsynchronized element store (`cell.set`) or the non-atomic
  exclusive in-use flag. `Arc<RefCell<T>>` is precisely Rust's canonical
  unsoundness example; the kind lattice makes it unrepresentable instead of
  merely rejected.

The two disciplines meet only by *composition*: interior mutability behind an
arc is expressed by an `Arc<X>` whose member is a sync cell, never by wrapping
a cell in `Arc`.

### 4.2 Kind table

| Kind | `Sync(cell)` | Element bound | Element access |
| --- | --- | --- | --- |
| plain | no | any type | `get` (clone) / `set` (replace) |
| exclusive | no | any type | plain ops + `rmw` (move-out region, in-use flag, panics on reentry) |
| atomic | yes | signless int/float primitive, power-of-two width ≥ 8 bits (verified in `ReussirTypes.cpp`) | atomic `get` (acquire) / `set` (release) / `rmw` (direct `atomicrmw` or CAS-retry region, acq-rel) |
| mutex | yes | **`Sync(τ)`**, and τ a valid memref element (rc pointer or primitive; records/nullables don't qualify) | ops inside `sync.mutex.critical_section`; `rmw` region runs exactly once, reentry deadlocks |
| flatlock | yes | **`Sync(τ)`**, memref-element | ops inside `sync.combining_lock.critical_section`; the body may execute on the *combining thread*, results return via a captured stack slot |
| rwlock | yes | **`Sync(τ)`**, memref-element | `get`/`rdlock` under the read lock (concurrent), `set`/`rmw` under the write lock |

### 4.3 Why lock-guarded cells demand `Sync(τ)`, not something weaker

Rust's `Mutex<T>: Sync` needs only `T: Send` because `&T` never escapes the
lock. Reussir cannot reproduce that weaker bound, for a reason **independent of
how the accessor ops are implemented**:

- **Stores.** Any thread may `cell.set` a value into the cell while retaining
  other rc handles to it (handles are duplicable; no linearity proves
  otherwise). If `τ` were not `Sync`, the storer's remaining handles and a
  later reader/dropper on another thread would race a normal refcount outside
  the lock. The lock guards the *slot*; it never guards the element's own
  count traffic.
- **Escapes.** Symmetrically, any op that lets the element (or a clone of it)
  leave the critical section — `cell.get`/`cell.rdlock` return a clone; an
  `rmw` region can yield it as output — places the element's counts in
  unsynchronized hands once the value is out. With `Sync(τ)` those counts are
  atomic and this is fine.

To be precise about the lowering (verified in `ConvertToSTD.cpp` and
`AcquireDropExpansion.cpp`): the acquire/drop that get/set/rmw/rdlock and the
box drop glue perform always execute *inside* the critical section — it is
the returned clone that subsequently lives outside the lock, and whose later
drop happens on the caller's thread. That escape is designed behavior, not a
defect, and it is exactly what the `Sync(τ)` bound makes sound. Even if the
accessor surface were narrowed to region-only access, the bound would not
weaken (the store argument above stands on its own).

The bound also closes the smuggling hole from §1: a cell reachable from shared
space can only ever *hold* `Sync` values, so no later store can leak a
thread-local (normal-count) object across threads. This is the type-level
analogue of Rust bounding `Mutex<T>`'s element rather than scanning values.

For the common case, note the stratification does real work here:
`Mutex<Arc<FooBase>>` is legal while `Mutex<FooBase>` is not — sharing a
mutable slot of rc data means the data itself must be arc-colored.

### 4.4 Sync cells without `Arc` (rule "some cells work sync/send style alone")

A `rc<cell<τ, mutex>, shared, atomic>` (and its atomic/flatlock/rwlock
siblings) is `Sync` by itself — it is *the* sanctioned primitive for sharing
mutable state across threads, no `Arc` involved. `Arc` covers the immutable
half of the world (records/arrays/closures shared read-only); sync cells cover
the mutable half; composition (`Arc<X>` with sync-cell members) covers the
rest.

## 5. Interaction summary

```
Sync(τ)?
  scalar / global str                       yes
  Arc<X>            (WF: members Sync)      yes
  Arc<Array<T>>     (WF: Sync(T))           yes
  Arc<(A…)->R>      (WF: Sync(Aᵢ) + creation-site captures Sync)    yes
  cell: atomic/mutex/flatlock/rwlock (element: Sync + kind bounds)  yes
  [value] record                            iff all members Sync
  Nullable<P>                               iff Sync(P)
  bare [shared] record / array / closure    no
  cell: plain / exclusive                   no
  regional anything                         no (§6)

Arc<·> applicable to:   [shared] records, arrays, closures   — nothing else
Arc<cell>               never (kind carries atomicity instead)
Arc nesting             never (already an arc)
Arc member in a record  intended, blocked on per-member atomic axis (§3.1)
Arc<recursive X>        WF via automatic SCC promotion — every same-group
                        member link re-colors atomic (§3.3); closed world,
                        bare→arc conversion is an explicit rebuild
```

## 6. Regional objects *(open, deliberately deferred)*

Regional objects (`flex`, `rigid`, `field`, region handles) are `!Sync` and
`Arc` rejects `[regional]` inners. Verona-style precedent says more is
possible — a uniquely-entered region is transferable as a unit (`iso`), and a
frozen region could be shareable once the rigid/SCC release path has an atomic
variant. There is a working scheme for sync regional objects, but it is
intentionally not designed here; until it is written down, the rule is simply
that regions and threads do not mix.

## 7. Implementation status and open items

Implemented today:

- Frontend: `TyKind::Arc`, inner rejection table, arc construction for
  `[shared]` records, transparent reads, mono-time re-check of generic inners;
  `Cell`/`RefCell` surface types; reserved names.
- Dialect: full `CellKind` lattice with per-kind verifiers; atomic box axis
  with `fetch_sub` decrement discipline; sync-dialect lock lowerings
  (mutex/flatlock/rwlock critical sections); atomic-shared-box requirement for
  sync cells.

Not yet implemented (ordered roughly by dependency):

1. **`Sync` auto trait** — retire the Phase-0 `Send` declaration, keep `Sync`
   as the single predicate, and add its structural propagation + member-path
   diagnostics.
2. ~~Per-member atomic axis~~ — superseded and **implemented** as the §3.3
   whole-subtree rule: no record encoding at all; interior references
   inherit the box's atomic kind, `Arc` members are explicit
   `rc<…, shared, atomic>` member types, `Arc<List<T>>` is accepted with the
   whole spine atomic, and the SCC promotion ships in the `Sync` checker
   (`semi/traits/sync.rs::recursive_group`).
3. **`Arc` lowering** — implemented for records (atomic create, borrow,
   projection, dispatch, atomic drop/acquire glue); arc'd arrays and
   closures remain item 4.
4. **Arc construction for arrays and closures** (annotations are accepted;
   constructors don't exist), including the §3.2 creation-site capture check.
5. ~~Surface types for sync cells~~ — **implemented**: `Atomic<τ>`,
   `Mutex<τ>`, `FlatLock<τ>`, `RwLock<τ>` with the §4.2 element bounds
   enforced at annotation/sweep/mono, the `core::intrinsic::cell` matrix
   over every kind, and `rdlock` read transactions. One narrowing: surface
   `rmw` is rejected on `Atomic` cells (the CAS-retry region must be
   effect-free, which a closure call is not) — dedicated atomic arithmetic
   intrinsics (`fetch_add`, …) are follow-up work.
6. *(later)* escape hatches — an unsafe "assert Sync" for FFI types and an
   opt-out poison for structurally-Sync but thread-affine types (Rust
   `unsafe impl` / `MutexGuard` precedents); a linear closure kind (§3.2);
   sync regional objects (§6).

## Appendix: comparison map

| | Reussir | Rust | Swift | Verona | Koka/Perceus |
| --- | --- | --- | --- | --- | --- |
| predicate(s) | `Sync` (single, = Rust's Send∧Sync) | `Send` + `Sync` | `Sendable` (single) | none — per-reference capabilities | none — runtime marking |
| where checked | Arc WF + creation site + cell element bounds | auto traits + API bounds (`spawn: F: Send`) | conformance + `@Sendable` capture check | capability of each reference (`iso`/`imm` sendable, `mut` not) | dynamic, on first share |
| atomicity granularity | per box, at birth | per type (`Rc` vs `Arc`) | n/a (ARC always atomic) | per region | per box, dynamic one-way |
| shared mutability | sync cells (kind = discipline) | `Mutex`/`RwLock`/atomics | actors | cowns + `when` behaviours | none (pure) |
| cyclic data across threads | deferred (§6) | `unsafe`/arena | SE-0414 region transfer | `iso` region transfer, `freeze` | n/a |

Key sources: Rustonomicon "Send and Sync"; std docs for `Arc`/`Mutex`/
`RwLock` (the `RwLock: Sync ⇐ T: Send + Sync` vs `Mutex: Sync ⇐ T: Send`
distinction); Swift SE-0302 (`Sendable`, `@Sendable` capture rules) and
SE-0414 (region-based isolation); Verona BoC (OOPSLA 2023, DOI
10.1145/3622852); Perceus TR (thread-safe RC with dynamic atomicity marking).
