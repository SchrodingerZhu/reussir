# Trait System Design (reussir-core)

Status: **proposal / RFC**. This document proposes the type-class / trait
subsystem for the Rust port of the core, replacing the inherited primitive
design. It states the model, the algorithms, the IR, and a staged plan.

---

## 1. What we inherited, and why it has to change

The Haskell core never had a real type-class system. It had a *bound* system
bolted onto type inference, with the class hierarchy hard-coded:

- A "class" is just a path: `newtype Class = Class Path`. It carries **no
  methods, no associated items, no instances** — only a name.
- The hierarchy is a `ClassDAG` built once at startup, in code, for primitives
  only (`populatePrimitives`):

  ```
  PtrLike
  Num
  ├── Integral      (impl: i8…u64)
  └── FloatingPoint (impl: f16, f32, f64, bf16, f8)
  ```

  There is no way to *declare* a class or *give an instance*. "Membership" is a
  hand-written table (`addClassToType`).
- A bound is a flat list, `type TypeBound = [Class]`. Inference holes carry a
  bound; `satisfyBounds` checks a candidate type against it by walking the DAG
  (`isSuperClass`, heavy-light decomposition for fast ancestor queries).
- The surface language can *use* bounds (`<T: Num + Integral>` parses into
  `[name, [paths]]`) but cannot *declare* a trait or write an `impl`. The whole
  vocabulary is closed.

This is enough to type integer/float literals and reject `"x" + 1`, and nothing
more. It cannot express `Eq`, `Ord`, `Hash`, `Iterator`, user abstractions, or
anything with a method. The DAG is a special-purpose subtyping engine for a
fixed lattice; it is not a trait system.

Separately — and this is the key design constraint — the core already
**monomorphizes**. `Semi/FlowAnalysis.hs` implements the type-based flow
analysis of *The Simple Essence of Monomorphization* (Lutze et al., OOPSLA
2025): a flow graph whose nodes are `GenericID`s and whose edges are

- **concrete seed** — a concrete type instantiates a generic (`addConcreteFlow`);
- **direct edge** — `T` flows unchanged into another generic (`addDirectLink`),
  e.g. calling `g<T>` from `f<T>`;
- **ctor edge** — `T` flows *under a constructor* into another generic
  (`addCtorLink`), e.g. passing `List<T>` where `U` is expected.

`solveGeneric` propagates concrete seeds to a fixpoint. A cycle through a ctor
edge is a *growing edge* — monomorphization would not terminate (polymorphic
recursion), and it is rejected. Direct cycles are fine.

That algorithm decides the whole shape of the trait system, because it means
**we resolve polymorphism away at compile time**. We are Rust, not Haskell.

---

## 2. The core decision: System F (dictionaries) vs Rust (monomorphization)

Two coherent end-states:

| | System-F / GHC style | Rust style |
|---|---|---|
| Polymorphism at runtime | preserved; one copy of code | erased; one copy per instantiation |
| Trait evidence | runtime **dictionary** passed as a value | resolved at compile time; **no value** |
| Dispatch | dynamic (through the dictionary) | static (direct call); `dyn` is opt-in |
| Separate compilation of generics | yes | no (needs the instantiation) |
| Cost | boxing / indirection | zero-cost, code bloat |
| Heterogeneous collections | natural | needs `dyn Trait` (vtables) |

Reussir already monomorphizes, targets MLIR/LLVM, and is built around regions,
in-place reuse, and no GC. Carrying runtime dictionaries would fight every one
of those goals, and we already pay the monomorphization cost for generics
regardless of traits. **Recommendation: the Rust model** — nominal, coherent,
statically resolved at monomorphization time, zero runtime evidence.

We keep one hedge: resolution produces an explicit **`Evidence`** value (a proof
tree of which impl was chosen). Today we *consume* it statically (inline / emit
a direct call). The day we want `dyn Trait`, the same `Evidence` becomes a
vtable — dynamic dispatch is then an additive feature, not a redesign.

---

## 3. Design principles

- **D1 — Nominal & coherent.** A trait is a named declaration; an impl names its
  trait and self type. Global coherence: at most one impl per (trait, type),
  enforced by orphan + overlap rules. This makes resolution deterministic and
  is what lets monomorphization pick *the* method.
- **D2 — Static dispatch by default.** No dictionaries. `dyn Trait` is a
  deferred, additive phase.
- **D3 — Built-ins are not special.** `Num`, `Integral`, `FloatingPoint`,
  `PtrLike` become ordinary built-in trait declarations with built-in impls
  ("lang items"). The solver has *no* hard-coded hierarchy; the DAG disappears.
  Super-traits subsume what the DAG did (`Integral: Num`).
- **D4 — Bounds desugar to obligations.** `<T: Trait>` and `where` clauses are
  sugar for predicates `T: Trait` that the solver must discharge.
- **D5 — Capability is a bound.** A trait ranges over a value *type*; but a
  value's `Flexivity` (`Irrelevant | Regional | Flex | Rigid`) is itself
  expressible in bound position: `<T: Flex>`, `where T: Regional`. These are
  *built-in capability predicates*, not traits — the capability lattice is
  closed (no user impls, no orphan rule), and a capability bound is discharged
  by **lattice subsumption** rather than by impl search. They share the
  obligation/fulfillment plumbing (§4, §5.3) with trait bounds, so receiver
  capability and region requirements are written and solved uniformly with the
  rest of the bound language. The capability remains carried on the type
  (`Ty::Record { flex }`); the bound constrains what may instantiate the
  parameter.
- **D6 — Design for associated types, implement later.** The IR reserves room
  for associated items and type projection; the first solver cut omits them.

---

## 4. The IR

Sketch (illustrative Rust, in `reussir-core`):

```rust
// ---- types (port of Semi/Type.hs) -------------------------------------
pub enum Ty {
    Record { path: Path, args: Vec<Ty>, flex: Flexivity },
    Int(IntTy), Fp(FpTy), Bool, Str, Unit,
    Closure(Vec<Ty>, Box<Ty>),
    Nullable(Box<Ty>),
    Generic(GenericId),   // rigid: a bound type parameter in scope
    Hole(HoleId),         // unification variable
    Bottom,               // error recovery
}

// ---- traits & impls ---------------------------------------------------
pub struct TraitDef {
    pub id: TraitId,
    pub name: Path,
    pub params: Vec<GenericId>,          // params[0] is the implicit `Self`
    pub supertraits: Vec<TraitRef>,      // replaces the ClassDAG edges
    pub methods: Vec<MethodSig>,
    pub assoc_tys: Vec<AssocTyDef>,      // reserved; empty for now (D6)
}

pub struct ImplDef {
    pub generics: Vec<GenericId>,        // impl<…>
    pub trait_ref: TraitRef,             // Trait<Args> being implemented
    pub self_ty: Ty,                     //   for Self
    pub where_clauses: Vec<TraitRef>,    // obligations the impl assumes
    pub methods: Vec<MethodImpl>,        // bodies (each a FnDef)
}

/// `Trait<args>`. By convention args[0] is the Self type.
pub struct TraitRef { pub trait_id: TraitId, pub args: Vec<Ty> }

/// A predicate the solver must discharge.
pub enum Obligation {
    Trait(TraitRef),        // `τ: Trait<…>` — discharged by impl search (§5.1)
    Cap(Ty, Capability),    // `τ: Flex` etc. — discharged by lattice subsumption
    // later: projection-equality `<T as Trait>::Assoc == U`, etc.
}

/// The capability lattice (built-in, closed). `Cap(τ, c)` holds when the
/// capability of τ is at least `c` in this ordering.
pub enum Capability { Irrelevant, Regional, Flex, Rigid }

/// The proof that an obligation holds — consumed statically now, a vtable later.
pub enum Evidence {
    Impl { impl_id: ImplId, args: Vec<Ty>, sub: Vec<Evidence> }, // chose an impl
    Param(usize),         // discharged by an in-scope assumption (T: Trait)
    Super { of: Box<Evidence>, idx: usize },                     // supertrait
}
```

The **instance database** indexes impls for fast candidate lookup, keyed by
`(TraitId, head-constructor of Self)` — e.g. `(Ord, Record "List")`. Blanket
impls (`impl<T> Tr for T`) go in a per-trait fallback bucket.

---

## 5. The algorithms

### 5.1 Trait resolution / instance selection — the heart of it

`select(ob: &TraitRef, env: &ParamEnv) -> Result<Evidence>`:

1. **Gather candidates** from three sources:
   - **assumptions** in `env` (the function's own `T: Trait` bounds) — these
     discharge obligations whose Self is an abstract `Generic`/`Hole`;
   - **impls** from the database whose `(trait, self-head)` matches;
   - **super-traits** reachable from an assumption/impl already in hand.
2. **One-way match** the obligation's args against each candidate header
   (matching, not full unification: only the candidate's own variables may be
   assigned). On success an impl yields *sub-obligations* from its
   `where_clauses`, instantiated with the match substitution.
3. **Recurse** on sub-obligations; assemble an `Evidence::Impl { sub }`.
4. **Coherence ⇒ ≤1 viable candidate**, so the result is unique (no
   backtracking, no committed-choice ambiguity). If Self is still a hole and no
   assumption applies, **suspend** the obligation (see 5.3).
5. **Termination.** Memoize by normalized `TraitRef`. Bound recursion depth
   (overflow error, like rustc's recursion limit). Crucially, in the *whole-
   program* phase this is also bounded by the flow analysis (5.5): an
   ever-growing instance chain is a growing edge.

### 5.2 Coherence (checked once, when items are collected)

- **Orphan rule.** An impl is admissible only if its trait *or* the head
  constructor of its self type is local to the current compilation unit.
  Prevents two crates from defining conflicting impls.
- **Overlap rule.** For every pair of impls of the same trait, unify their
  heads (Self + args) treating impl generics as unifiable. If they unify, the
  impls overlap → error. (Specialization — allowing a strictly-more-specific
  impl to win — is deliberately deferred.)

This is the part that *replaces* `populatePrimitives` + the DAG: built-in impls
go through the exact same coherence check as user impls.

### 5.3 Integration with inference (port of `Tyck` + `Unification`)

- Holes are still a union-find (`UnificationState` → Rust `UnionFind<HoleId>`).
  A hole no longer carries `[Class]`; instead the inference context owns a
  **fulfillment context**: a worklist of pending `Obligation`s, each pointing at
  the holes it mentions.
- `unify` is unchanged structurally (port of `unifyForced`). When it solves a
  hole, it wakes any pending obligation mentioning that hole and re-runs
  `select`. Progress-or-suspend, to a fixpoint — exactly rustc's fulfillment
  loop.
- A function `fn f<T: Trait>(…)` pushes `T: Trait` into the `ParamEnv` as a
  **given** while checking its body; calls that need `T: Trait` discharge
  against that assumption (`Evidence::Param`).
- **At the end of a function**, every obligation must be discharged. One whose
  Self is concrete must find an impl (else "trait bound not satisfied"). One
  whose Self is the function's own `Generic` must be *implied by the declared
  bounds* (else "the bound `T: Trait` is not declared on `f`") — this is how we
  re-derive a function's required `where` clause and check it.

This subsumes today's `satisfyBounds`/`subsumeBound`/`meetBound` trio: bound
subsumption is just "discharge `T: Super` given `T: Sub`" via super-trait edges.

### 5.4 Method resolution

For `recv.m(args)` (and `Trait::m(...)`):

1. Find in-scope traits declaring a method `m`.
2. For each, form the obligation `typeof(recv): Trait<…>` and the call's
   expected signature; keep the candidates whose receiver can satisfy.
3. Coherence + the fulfillment loop pick one. The call node records "method `m`
   of this resolved trait obligation". After monomorphization (5.5) the
   obligation's `Evidence` is concrete, so the node lowers to a **direct call**
   to that impl's method symbol.

### 5.5 Monomorphization: a reachable-instantiation worklist

We **do not** port the inherited flow analysis. It keys its graph on individual
`GenericID`s and computes, independently per variable, the set of concrete types
that reach each parameter. With more than one parameter that loses the
*correlation* between them. Given

```
fn pair<A, B>(a: A, b: B) -> Pair<A, B> { … }
…
pair::<i32, bool>(1, true);
pair::<f64, str >(1.0, "s");
```

the per-variable view yields `A ∈ {i32, f64}`, `B ∈ {bool, str}` — and to
specialize you must take a *cartesian product*, 4 combinations, of which only 2
ever occur. The phantom copies are wasted code at best and *ill-typed* at worst
(a `where A: Convert<B>` that holds only for the real pairs would spuriously
fail). Per-variable reachability is a non-relational over-approximation;
monomorphization needs the relational truth.

**The robust algorithm: an instantiation worklist over whole substitutions.**
This is the standard, precise approach (MLton, rustc's mono-item collector, C++
template instantiation). Track *substitutions*, not variables.

- A **mono request** is `(Def, σ)` where `Def` is a polymorphic definition
  (function, record, or impl-method) and `σ` is a **ground** substitution
  mapping *all* of `Def`'s generics to concrete types. `σ` is a tuple, so it
  keeps the parameters correlated by construction.
- **Roots.** Seed from everything that must exist regardless of generics:
  exported / entry functions, and `extern`/`trampoline` items whose type args
  are given concretely. Non-generic defs have `σ = ∅`.
- **Step.** Pop `(Def, σ)` and apply `σ` to the body. Every call/ctor site in
  the substituted body now has *ground* type arguments (the site's args are
  built from `Def`'s generics, which `σ` has made concrete — this is exactly why
  inference must leave no site under-determined). For each site calling `Callee`
  with ground args `ts`, build `σ' = (Callee.generics ↦ ts)` and push
  `(Callee, σ')` if unseen.
- **Memoize** seen `(Def, σ)` (normalize `σ`). This terminates direct recursion
  and de-duplicates the emitted specializations.
- **Output.** The visited request set *is* the monomorphic program / "LangMono":
  emit one specialized copy per request, each call a direct symbol (existing v0
  mangling). No cartesian product is ever formed — only tuples that arise.

**Traits ride the same worklist.** When stepping `(Def, σ)`, a method call
`recv.m(args)` has a ground receiver type `τ = σ(…)`. Resolve `τ: Trait` with
`select` (§5.1); coherence makes the chosen `ImplDef` and its ground
impl-substitution `σ_impl` unique. Push `(impl-method, σ_impl)`; resolve the
impl's `where`-clause obligations recursively, each chosen impl-method becoming
another request. Trait resolution is *just more edges in the same graph*.

**Termination / polymorphic recursion.** The request set is infinite **iff**
there is polymorphic recursion: a cycle `Def → … → Def` along which the type
*strictly grows* (`f<T>` calls `f<List<T>>`). Detect it with a **strict-subterm
(homeomorphic-embedding) check**: walking a call chain, if we reach `(Def, σ')`
where an ancestor `(Def, σ)` on the same chain has `σ` a proper subterm of `σ'`,
the type is growing without bound — reject with a diagnostic naming the cycle.
Keep a type-size cap as a coarse backstop. This is the inherited "growing edge"
idea lifted from single variables to whole substitutions, and it is exactly the
"up to polymorphic recursion" boundary that *The Simple Essence of
Monomorphization* (Lutze et al., OOPSLA 2025) draws. We borrow that boundary,
not its per-variable flow encoding.

### 5.6 Capability obligations

> **Status (revised): deferred.** This section assumed a *total* capability
> lattice with a `⊒` subsumption. That is wrong: `Flex` (mutable, but cannot be
> materialized out of its region) and `Rigid` (immutable, but materializable)
> are **incomparable** — different axes, not a chain. Because the subsumption
> semantics are unsettled, the capability-as-a-bound machinery (`Obligation::Cap`
> / `Evidence::Cap` / the `satisfies` lattice) has been **removed** from the
> code for now; `Capability` is just the coloring stored on `Ty::Record`. The
> design below is retained as a sketch and will be reworked (likely as a partial
> order, or folded into coloring/region checking) before capability bounds
> return.

`Obligation::Cap(τ, c)` is discharged separately from impl search:

- If `τ` is concrete, read its capability off the type (`Ty::Record { flex }`,
  or the built-in capability of primitives) and check `cap(τ) ⊒ c` in the
  lattice (§4). Succeed or report "`τ` is not `c`".
- If `τ` is an in-scope `Generic` carrying its own capability bound `c'`,
  discharge by **subsumption**: `c' ⊒ c`. (This is the capability analogue of
  super-trait subsumption — and is what the old `subsumeBound` did, restricted
  to the fixed lattice.)
- If `τ` is still a `Hole`, **suspend** like any other obligation (§5.3) and
  retry when the hole is solved.

Because the lattice is closed and finite, there is no search, no coherence, and
no termination concern — capability discharge is a constant-time lattice
comparison riding the same fulfillment worklist as trait obligations.

---

## 6. Code overview (module layout in `reussir-core`)

Everything for the Semi phase lives under `semi::`; `full::*` (the
monomorphized, mangled representation) is its future sibling. As built:

```
crates/reussir-core/src/
  surface.rs          typed surface AST + direct CST lowering / aeson serde
  semi/               the whole Semi phase
    ty.rs             Ty, TyKind, TyCtxt (interned), IntTy, FpTy, Capability, ids
    infer.rs          ena union-find holes, unify/occurs/zonk, lazy instantiation
    traits.rs         TraitRef, Obligation, Evidence
    traits/db.rs      instance database, select(), super-trait evidence chains
    traits/def.rs     TraitDef, ImplDef, MethodSig, AssocTyDef
    traits/builtins.rs  lang-item traits + impls (Num/Integral/FP/PtrLike/Send/Sync)
    hir.rs            Semi HIR: Expr/ExprKind, DecisionTree, Function
    ctxt.rs           collected items + Elaborator state + scan/collect driver
    ty_eval.rs        surface-type → Ty + the regional capability coloring
    fulfill.rs        fulfillment context (deferred obligations, fixpoint discharge)
    check.rs          bidirectional infer/check over all expression forms
    pattern.rs        decision-tree pattern compilation
```

Future (not yet built):

```
  full/               monomorphized, mangled representation
    ty.rs             Full type (mangled Symbol, resolved capability, no generics)
    collect.rs        instantiation worklist over whole substitutions (§5.5)
    specialize.rs     emit monomorphic IR with statically-resolved calls
```

Still pending inside `semi`: coherence (orphan/overlap), generic-impl one-way
matching, associated types, and surface `trait`/`impl` syntax.

Haskell → Rust mapping:

| Haskell | Rust |
|---|---|
| `Data/Semi/Type.hs` | `ty/` |
| `Data/Class.hs`, `Semi/Context.hs::populatePrimitives`, `Class.hs` (DAG/HLD) | **deleted**; replaced by `traits/builtins.rs` + super-traits |
| `Data/Semi/Unification.hs`, `Semi/Unification.hs` | `infer/unify.rs` + `infer/fulfill.rs` |
| `Semi/Tyck.hs` | `infer/tyck.rs` |
| `Data/Generic.hs`, `Generic.hs`, `Semi/FlowAnalysis.hs` | **not ported**; replaced by `mono/collect.rs` (§5.5 worklist) |
| (new) | `traits/{db,select,coherence}.rs`, `mono/specialize.rs` |

---

## 7. Surface syntax impact

Today only **use sites** parse: `<T: A + B>` lowers to `[name, [paths]]`. There
is no `trait` or `impl` declaration anywhere in the grammar (Rust *or* Haskell).
A user-facing trait system needs new surface constructs:

```
trait Ord<Self>: Eq { fn cmp(self, other: Self) -> Ordering; }
impl Ord for i32 { fn cmp(self, other: i32) -> Ordering { … } }
impl<T: Ord> Ord for List<T> where … { … }
```

That is: lexer keywords (`trait`, `impl`, `for`, `where`), grammar rules, CST
kinds, and AST/JSON lowering in `reussir-syntax`. **Decision: this is in the
first cut** — user-declared traits from the start, not deferred.

Capability bounds reuse the existing bound position. The names `Irrelevant`,
`Regional`, `Flex`, `Rigid` in a bound list are recognized as built-in
capability predicates (§5.6) rather than trait references; everything else in
bound position resolves to a `TraitRef`.

---

## 8. Staged plan

- **Phase 0 — Foundations, behavior-preserving.** Port `Ty` + subst. Stand up
  `traits/` with `Obligation` (`Trait` + `Cap`) / `Evidence` and a trivial
  solver. Re-express `Num/Integral/FloatingPoint/PtrLike` as built-in trait defs
  + built-in impls in `builtins.rs`, and the `Flexivity` lattice as the `Cap`
  predicate. Delete the `ClassDAG`/HLD machinery. **Net effect: identical
  type-checking behavior, zero hard-coded hierarchy.**
- **Phase 1 — Resolution + user traits.** Implement `select` (candidate gather,
  one-way match, recursion, memo, overflow) + coherence (orphan/overlap) +
  capability discharge (§5.6). Add surface `trait`/`impl`/`where` to
  `reussir-syntax` (lexer, grammar, CST, AST/JSON); collect user traits/impls;
  method resolution; wire the fulfillment context into `tyck`.
- **Phase 2 — Monomorphization.** Implement the instantiation worklist (§5.5)
  over whole substitutions; resolve trait obligations per request and push
  impl-method instantiations; strict-subterm growth check for polymorphic
  recursion; emit `dyn`-free LangMono.
- **Phase 3 — Future.** Associated types & projection, default methods, blanket
  impls, `dyn Trait` (Evidence → vtable), specialization.

---

## 9. Decisions

- **Model — DECIDED: Rust, monomorphized static dispatch.** No runtime
  dictionaries; `dyn Trait` is a deferred, additive phase (Evidence → vtable).
- **Dispatch — DECIDED: static-only** for the first cut (follows from the
  model).
- **Capabilities — DECIDED: capability is a bound.** The `Flexivity` lattice is
  a built-in `Cap` predicate writable in bound position, discharged by lattice
  subsumption (§5.6). Not a trait; closed; no orphan rule.
- **Surface traits — DECIDED: in the first cut.** User-declared
  `trait`/`impl`/`where` from the start (Phase 1).
- **F1 — Coherence model (open).** Global coherence + orphan rules (recommended;
  what makes monomorphic resolution unambiguous) vs. scoped/local instances
  (Scala-like; more flexible, much harder to monomorphize). Proposal: global
  coherence.
- **F4 — Associated types (open).** Now or later. Proposal: later (Phase 3); the
  IR reserves room (`assoc_tys`).
