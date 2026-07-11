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
valid after it. At lowering time the eval walks its operand's SSA chain back
through those ops (and through block arguments, meeting over predecessors)
to the fullest statically visible suffix, and tests *that* id — e.g. a
parameter typed `(a, b) -> c` applied twice locally lets the eval test
`closure<(a,b)->c>` instead of `closure<()->c>`.

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
`reussir.wpd.vtables` (vtable symbol → id tiers) on the module; the
conversion patterns carry each vtable's tiers onto its global
(`reussir.wpd.type_ids`) and assert the strengthened id at each indirect
slot call site with `reussir.closure.wpd_test` → at
`translateModuleToLLVMIR`, the dialect's LLVM translation interface — the
LLVM dialect can express neither `!type` metadata nor `llvm.type.test`'s
metadata-string operand — stamps the metadata and lowers each `wpd_test`
into `llvm.type.test` + `llvm.assume` → the backend LLVM pipeline runs
`WholeProgramDevirt` before the per-module pipeline (so devirtualized calls
inline) and lowers away the remaining type tests.

## Invariant canary

Everything above rests on the `evaluate` ABI reading all arguments from the
payload. `tests/integration/conversion/closure_wpd_ids.mlir` pins both that
signature and the id tiers; if evaluation ever passes remaining arguments in
registers, the suffix hierarchy collapses to exact-signature families and the
tiered ids must be regenerated accordingly.
