# Whole-program devirtualization for closures

## The vtable hierarchy

A closure box is `{refcnt, vtable, cursor, payload…}` behind an rc pointer;
its vtable is three slots, `{drop, clone, evaluate}`. The slot ABIs depend on
nothing beyond the closure's **return type**:

- `evaluate` is `fn(rc<box>) -> c` — `closure.eval` is verified fully-applied
  and the outlined function reads **every** argument from the payload;
- `drop` is `fn(rc) -> void` and `clone` is `fn(rc) -> rc` — signature-free.

`closure.apply`/`closure.uniqify`/`closure.clone` never change the vtable —
only the static type (one fewer leading parameter) and the cursor. So the
vtable created for a signature `(a, b) -> c` may back a value of static type
`(b) -> c` or `() -> c`: in C++ terms, **`apply` is an upcast** — the static
type moves toward the base, the vtable pointer is unchanged. That induces a
suffix hierarchy per return type:

```
closure.uid  >  closure<(a, b) -> c>  >  closure<(b) -> c>  >  closure<() -> c>
(one vtable)     (exact signature)        (a applied)          (family root)
```

The set of vtables that can back a static type grows as leading parameters
are dropped: `vtables((a,b)->c) ⊆ vtables((b)->c) ⊆ vtables(()->c)`. Edges
require exact structural equality of the suffix — there is no variance. It is
a DAG keyed by suffixes, not a tree (`(b)->c` is the parent of `(x,b)->c` for
every `x`). There is no type-erased `closure` root: the type system has no
such type, so no call site could ever test it.

Captures fold into the same hierarchy for free: the frontend materializes a
capture as a pre-applied leading parameter (`|x: i32| x + k` outlines as
`(i32, i32) -> i32` with `k` applied at creation), so a capturing closure
shares its suffix tiers with every capture-free closure of the same visible
type — precisely the set a call site can actually observe.

This maps 1:1 onto LLVM's WPD model: each vtable global carries one `!type`
id per suffix of its original signature (all at offset 0 — a single address
point), and each indirect slot call tests the id of its operand's **static**
type with `llvm.type.test` + `llvm.assume`. WPD devirtualizes per
(type id, slot byte offset), so drop (0), clone (8) and evaluate (16)
devirtualize independently.

Type ids are structural and follow the project's v0 mangling scheme
(`ClosureWpd.h`, mirroring the subset in
`crates/reussir-core/src/full/mangle.rs`): a suffix `(tᵢ, …, tₙ) -> c`
mangles through the v0 closure-type production as
`_RFK17ReussirClosureWpd {C<digest(tⱼ)>}* E (C<digest(c)> | u)`, where each
type appears as a crate-root nominal holding the base-62 BLAKE3 digest of
its canonical textual form (the `Blake3Symbol.h` convention) — identical
signatures produce identical ids in any module. The most-derived tier is the
path `<vtable>::wpd` (`_RNv<vtable path>3wpd`), unique per vtable and useful
for exact-match reasoning, though locally created closures already
devirtualize without it (`closure.create` stores a constant vtable and every
vtable/slot load is `invariant.group`-tagged).

## Call-site strengthening

`closure.eval` requires `closure<() -> c>`, so a naive eval test uses the
**family root** id — whose candidate set is every closure in the program
returning `c`. The hierarchy only pays off by strengthening: since
apply/uniqify/clone preserve the vtable, any id valid before an apply is
valid after it. The basic-ops pass prologue runs a sparse forward dataflow
analysis (`SparseForwardDataFlowAnalysis`) whose lattice per SSA closure
value is the fullest statically known suffix of its backing vtable's
original signature: apply/uniqify/clone forward their operand's view
unchanged, every other producer contributes the static type, and merge
points (block arguments, over live predecessors only) join by longest
common suffix — loop-carried closures reach the fixpoint instead of being
given up on. Each indirect call site then asserts its operand's resolved
view — e.g. a parameter typed `(a, b) -> c` applied twice locally lets the
eval test `closure<(a,b)->c>` instead of `closure<()->c>`.

## Closed-world soundness

`llvm.type.test` + `assume` is UB the moment a vtable **without** the
metadata reaches a tested call site. The artifacts are therefore only emitted
when one module provably contains every closure vtable its call sites can
observe:

- whole-program AOT (`rrc`) with a **single codegen unit** — a partitioned
  build hands closures created in one unit to tests in another;
- `-O aggressive` or `-O size` — the opt levels that ask for whole-program
  effort (and the type tests only lower inside the optimizing backend
  pipeline; `llvm.type.test` cannot reach instruction selection);
- closures never cross the C ABI today (trampolines are scalar-only, the
  polymorphic FFI has no closure lowering, the runtime has no closure entry
  points) and every vtable is `internal`. If an exported signature ever
  carries a closure type, emission must be disabled for that module.

The REPL/JIT exchange closures between incremental modules and therefore
never opt in (`LoweringOptions::default()` keeps `closure_wpd` off).

Each vtable is stamped with translation-unit `!vcall_visibility`, which is
what makes devirtualization legal without LTO summaries.

## Pipeline

`rrc` (on by default at `-O aggressive`/`-O size`; `--no-closure-wpd` opts
out) → `LoweringOptions::closure_wpd` → the basic-ops lowering records
`reussir.wpd.vtables` (vtable symbol → id tiers) on the module and inserts
a `reussir.closure.wpd_test` with the strengthened id in front of each
indirect slot call site → the conversion patterns carry each vtable's tiers
onto its global (`reussir.wpd.type_ids`) and rewrite each `wpd_test` onto
its loaded vtable pointer → at `translateModuleToLLVMIR`, the dialect's
LLVM translation interface — the LLVM dialect can express neither `!type`
metadata nor `llvm.type.test`'s metadata-string operand — stamps the
metadata and lowers each `wpd_test` into `llvm.type.test` + `llvm.assume`
→ the backend LLVM pipeline folds the assertion's vtable load into the
call site's (both are `invariant.group` loads of the same slot), runs
`WholeProgramDevirt` before the per-module pipeline (so devirtualized calls
inline), and lowers away the remaining type tests.

## Measured results (x86-64, LLVM 22)

- An **exported** higher-order function evaluating a single-implementation
  family goes from a genuine indirect dispatch to the closure body inlined
  outright (`closure_wpd_devirt.rr`). Locally created and consumed closures
  already devirtualized before this work via the constant vtable store +
  `invariant.group` loads; WPD's contribution is exactly the cases local
  reasoning cannot see (exported functions, merges, data structures).
- **nbe-hoas** (HOAS normalizer, one closure family in the whole program):
  all 8 remaining indirect calls (evaluate/clone/drop in the recursive
  evaluator) fold to direct calls and inline at `-O aggressive`.
  Wall-clock is neutral-to-slightly-negative (~0–3%
  slower on a small container): dispatch was never the bottleneck there —
  the indirect branches were perfectly predictable — and inlining the large
  evaluator body into its callers costs some code size. The expected wins
  are small-closure HOF code (map/filter-style), where devirtualization
  lets the body fold into the loop.
- **Branch funnels are disabled** (`wholeprogramdevirt-branch-funnel-threshold=0`):
  `llvm.icall.branch.funnel` only survives instruction selection when the
  CFI-mode LowerTypeTests has rebuilt the vtables into one combined global,
  and we run the drop-mode lowering instead. Multi-implementation families
  simply stay indirect.

## Invariant canary

Everything above rests on the `evaluate` ABI reading all arguments from the
payload. `tests/integration/conversion/closure_wpd_ids.mlir` pins both that
signature and the id tiers; if evaluation ever passes remaining arguments in
registers, the suffix hierarchy collapses to exact-signature families and the
tiered ids must be regenerated accordingly.
