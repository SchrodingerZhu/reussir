# Per-constructor variant box sizing (#325)

## Motivation (measured, 2026-07-08, x86_64 Ryzen 9950X)

`nbe-closure` runs 1.74× slower than Koka at **fewer** executed instructions
(0.67×) — the gap is memory stalls: 13× the LLC misses, IPC 2.3 vs 5.9.
Padding Koka's constructors to Reussir's uniform 32-byte cells reproduced
Reussir's misses (0.22M → 2.51M vs our 2.42M) and runtime (125 → 238ms vs
our 224ms) almost exactly, so node size is causally ~the whole gap. Koka
sizes each constructor individually (`Var`/`VVar` = 16B); Reussir boxes
every variant at its max-arm width (32B), doubling the numerous leaf nodes.
Ruled out by experiment: borrow inference (Koka has none — `^` is
annotation-only and it dups `env` at fan-outs exactly like us), the
immediate encoding (PR #359: TaggedBox tied immortal), reuse pairing (both
compilers do size-keyed Perceus reuse), and allocation volume (+39% allocs
= +5% time).

## Design

A boxed **fused-header** variant cell is sized `header + arm[k]` (its own
immutable tag `k`) instead of `header + max(arms)`.

* **Offsets never move.** The payload offset is a function of the
  variant-wide alignment, which per-arm sizing keeps; only the trailing
  padding up to the max arm is dropped. Every projection/load/store/tag
  read is byte-identical, so record access lowering is untouched.
* **Value (inline) variants stay uniform.** They embed in other layouts
  (`ref.spilled`, `ref.memcpy`, array strides) and are never a heap cell.
* **No unsized free / no `token<?>`.** Every free is downstream of a tag
  discriminator: a tag-known dec (post-match, `reussir-infer-variant-tag`)
  frees with the static per-arm size; a tag-unknown dec is pushed to the
  outlined dtor whose per-arm switch (it must read the tag anyway to drop
  fields) frees with that arm's constant — placed *after* the
  special-pointer-tag immediate check. Works on caller-supplies-layout
  allocators (wasm/talc), not just mimalloc.
* **Reuse is untouched structurally.** `TokenReuse` pairs via
  `sameAllocatorBin(old, new)`; per-arm sizes flow in as values and
  cross-size pairings (16 ↮ 32) are refused automatically. NB: mimalloc has
  no 24-byte bin (24 → 32), so the practical win is precisely the 16-byte
  bin for one-word arms — the leaves the padding experiment indicted.

**Invariant** (what the verifier + ASan gates enforce): for any boxed
variant, `allocated size == token<> size == freed size == header + arm[k]`.

## Switch

Module unit attribute `reussir.per_constructor_box_sizing`
(`include/Reussir/Support/VariantBoxSizing.h`), stamped by
`rrc --variant-box-sizing=per-constructor` via
`reussirModuleSetPerConstructorBoxSizing` /
`pipeline::set_per_constructor_box_sizing`. Default `uniform`. An attribute
(not dialect state) so lit tests opt in by writing it on the module.

## Landing plan — stacked PRs, each ~300–400 LOC, each targeting main

1. **[this PR] Switch + arm-size helper + create-side + verifier.**
   `RecordType::getVariantArmAllocSize(dataLayout, tag)` (variant-wide
   alignment, arm-k payload); `ReussirRcCreateVariantOp::getTokenType()`
   returns the per-arm token under the attribute (fused-header, complete,
   non-regional only); `verifyRcCreateLikeOp` now checks the provided token
   against the op's own `TokenAcceptor::getTokenType()` — the single source
   of truth — so a wrong-size pairing is a *compile* error, not a heap
   overflow. EXPERIMENTAL until step 2: allocation is per-arm but
   tag-unknown frees still claim max-arm; safe only on mimalloc (free
   ignores the stated size).
2. **Free side.** (a) Tag-known decs: when `reussir-infer-variant-tag` (or
   dispatch fusion) pinned the arm, `ReussirRcDecOp::getTokenType()`
   returns the per-arm token so the unique-path
   `rc.reinterpret → token.free` carries the right size. (b) Tag-unknown
   decs: `AcquireDropExpansion`'s outlined dtor emits the per-arm constant
   free in each switch arm, after the tagged-immediate guard. Gate: an
   ASan-executing mixed-arm lit test driving both paths; wasm/talc
   correctness run. **This is the highest-risk step; do not stack further
   until ASan is green.**
3. **Reuse + lowering audit.** Confirm `SCFOpsLowering`'s same-bin
   `token.realloc` fast path keys on concrete arm sizes; lit-pin that a
   16→32 pairing no longer fires while 32→32 still does; poison-reuse
   guard uses the token size (already per-arm by construction).
4. **Indirect producers.** Regional `rc.create*` (bump path in
   `crates/reussir-rt/src/region/mod.rs` + `initializeRcCreateStorage`)
   and TRMC constructor contexts (`cctx.extend/apply` hole allocation in
   `TRMCRecursionAnalysis`) take the concrete arm size. `quote` is
   TRMC-heavy and hot — measure before/after.
5. **Validate + flip default.** Targets from the padding control:
   nbe-closure LLC ~2.4M → ~0.2M, ~224 → ~130ms, RSS ~246 → ~155MB.
   Measure the lost 16↔32 reuse pairings on the tree benches; full suite +
   aarch64 + wasm; then flip `--variant-box-sizing` default and keep the
   flag one release as escape hatch.
6. **FFI marshalling** (separate): non-uniform + fused-header + tagged
   layouts already rule out plain-cast FFI; generated marshalling is the
   answer regardless.

## Verified so far (PR 1)

Token instantiation on the mixed-arm probe:
`Cons{i64, rc}` → `token<align 8, size 24>`, `Var{i64}` →
`token<align 8, size 16>` with the attribute; both 24 without. Full lit
suite green with the verifier now routed through `getTokenType()`.
