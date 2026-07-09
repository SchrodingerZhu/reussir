# Per-constructor variant box sizing

## Motivation (measured, x86_64 Ryzen 9950X)

`nbe-closure` ran ~1.74× slower than Koka at *fewer* executed instructions —
the gap is memory stalls (13× the LLC misses, IPC 2.3 vs 5.9). Padding Koka's
constructors to Reussir's uniform 32-byte cells reproduced Reussir's misses
(0.22M → 2.51M vs our 2.42M) and runtime (125 → 238 ms vs our 224 ms) almost
exactly, so node size is causally ~the whole gap. Koka sizes each constructor
individually (`Var`/`VVar` = 16 B); Reussir boxed every variant at its
max-arm width (32 B), doubling the numerous leaf nodes. (Ruled out by
experiment: object algebra/footprint via Koka padding, the immediate encoding,
borrow inference — Koka has none — and allocation *volume*.)

## What the feature does

A boxed **fused-header** variant cell is allocated `header + arm[k]` (its own
constructed tag `k`) instead of `header + max(arms)`. Field offsets never
move — the payload offset comes from the variant-wide alignment, so per-arm
sizing only drops the trailing padding up to the max arm. Value (inline)
variants stay uniform (they embed in other layouts). `nbe-closure` allocations
go from `218× 32 B` to `77× 16 B + 140× 24 B + 4× 32 B`.

Note the mimalloc bin geometry: 24 rounds up to the 32-byte bin, so the real
cache win is the **16-byte bin for one-word arms** (`Var`/`VVar`); the 24-byte
arms don't shrink their footprint until an 8-granular bin exists
(`MI_MAX_ALIGN_SIZE=8`, a separate lever). On stock mimalloc this measured as a
~1.10× win with the requested-size bookkeeping halved.

## What "size" means, per allocator

The shipping runtime is mimalloc behind a `GlobalAlloc` whose `dealloc`
**ignores the layout** — `dealloc(ptr, _layout) { mi_free(ptr) }` — `mi_free`
recovers the true block from mimalloc's own metadata. So the size passed to
`__reussir_deallocate` is never used on mimalloc; per-constructor sizing needs
only the *allocation* side to get the cache win there.

The exact free/realloc size matters only for a **size-strict** allocator (the
wasm/talc future). To keep that path correct too, decrements track the size:

- A decrement that knows its arm statically (a destructuring match, or the
  tag pinned by `reussir-infer-variant-tag`) produces a token of that arm's
  exact size.
- An **unpinned** decrement of a non-uniform variant produces a **dynamic**
  token, `token<align, ?>`. It lowers to a fat `{ptr, size}` pair: the size is
  read at production from the box's still-live fused tag (a per-arm switch), so
  free and realloc pass the exact runtime size on any allocator. A variant
  whose arms are all one size stays a static token.

No unsized `dealloc` and no set-typed token were needed: the size is always
recoverable — statically at a pinned dec, or carried in the fat token at an
unpinned one.

## Reuse

`TokenReuse` pairs a dead donor token with a construction. `token<?>` is a
**universal fallback donor**: it can be resized to any acceptor at runtime via
`token.realloc`. Scoring tiers (`hueristic`):

- `>= 2` — exact static match → `token.ensure` (preferred; locality-scored).
- `1` — a fixed-size donor in the same allocator bin → `token.realloc`.
- `0` — a dynamic `token<?>` → `token.realloc` (fallback; used only when no
  statically-sized donor fits).

**Binning (`sameAllocatorBin`) is only a pairing heuristic** — it decides which
donor to offer to which acceptor. It is *not* a lowering-correctness property,
so it does not gate any lowering. `token.realloc` always lowers to
`__reussir_reallocate` (a real resize) followed by an invariant-group
**launder** of the result: an LTO'd allocator fast path may return the
identical pointer, whose stale invariant-group metadata must be stripped; the
ptr-eq assume keeps value propagation across the barrier. (`token.realloc`'s
result is always a static token, so the launder applies to a bare pointer.)

Follow-up: the reuse `realloc` copies the dead donor's contents when it
relocates — wasted work. A no-copy resize is tracked in #362.

## Status / follow-ups

Landed: the switch (`reussir.per_constructor_box_sizing` module attr,
`rrc --variant-box-sizing`), per-arm allocation, dynamic-token free, and
`token<?>` fallback reuse. Sound (requested alloc/free sizes round-trip;
executing gate `per_ctor_box_sizing_mixed_arms.rr` under both sizings).

**Now the default** (`rrc --variant-box-sizing per-constructor`; `uniform`
is the opt-out). Every construction path was already per-arm-correct through
the `getTokenType()` / `getVariantArmAllocSize()` single source of truth —
the heap path (#360/#361), the **regional** bump `rc.create` (token-driven:
the cell size is the token's, never `getTypeSize(rcBoxType)`; the region is a
linked bump, so per-arm just shortens each cell), and **TRMC** holes (the
constructor's token is threaded through unchanged). Flipping the flag needed
no lowering change; the full integration suite is green under it.

Remaining: the 8-granular mimalloc bin to capture the 24-byte arms (a runtime
knob, orthogonal — `AllocatorBinModel.h` is only a *pairing hint*, so its
geometry never gates correctness); benchmark the full suite to quantify the
win at the new default; generated FFI marshalling (the layout was never a
plain cast, so non-uniform sizing doesn't move that goalpost).
