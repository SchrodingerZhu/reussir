# Tensor kernels: value semantics over rc arrays

Adds a frontend `Tensor` type — transient, non-materializable, fusion-
transparent — so users write value-semantic kernels that compile to
`linalg` on tensors, with rc arrays as the only durable citizens. The
framing parallel is the one the language already teaches: **flex objects
are to `freeze` what tensors are to `materialize`.** A flex object cannot
escape its region unfrozen; a tensor cannot escape its chain
unmaterialized. (`str<local>`'s transient lifescope and the #344 note
"`modify`'s view must not escape its region" are the other in-tree
precedents.)

## Why combinators cannot live on arrays directly

The two in-place systems do not compose per-op. Rc/CoW decides
in-place-ness at *runtime, per object* (`rc.is_unique`); one-shot
bufferize decides it *statically, per SSA region*. If every combinator
round-trips through an rc array, each op puts a uniqueness branch, a
`token.alloc`, and a region boundary between itself and the next linalg
op — and linalg fusion, which walks tensor SSA use-def chains, sees
opaque `reussir.*` ops and stops. Fusing across that would mean
implementing `BufferizableOpInterface` on rc ops and feeding a static
analysis a runtime predicate. The scoped-tensor design dissolves it: the
runtime decision happens **once at each boundary**, the static analysis
owns everything inside. This is also the spot no one else occupies —
Lean/Koka have runtime-uniqueness in-place updates but no tensor
compiler; every MLIR frontend has the tensor compiler but *asserts*
`restrict`/`writable` unverified. Reussir's CoW machinery manufactures
the aliasing contract one-shot needs.

## What already exists

`array.view` accepts a tensor result, verified and lowered to
`bufferization.to_tensor %m restrict [writable]` (`writable` iff the view
came from `array.with_unique_view`). `ConvertToSTD` declares
tensor/linalg/bufferization as dependent + legal dialects; the `linalg`
transform-dialect extension is registered; the `kernel` transform anchor
runs in the pipeline; and `array_tensor_view_e2e.mlir` is a working,
executed `linalg.matmul` + `bufferization.materialize_in_destination`
over an rc array with the RC-aliasing assertions passing. What is missing
is the frontend type, the boundary facilities, and the bufferization
passes in the shipping pipeline.

## The `Tensor` type

Scoping comes from typing rules, not a syntactic block — one-shot needs
tensors function-local, nothing more, so the "scope" is the maximal
tensor-typed SSA subgraph inside a function:

- may **not**: be a record field, be captured by a closure, cross FFI,
  be a `[field]`, appear in a plain `fn` signature;
- **may**: flow through let-bindings and `if`/`match` (→ `scf.if` on
  tensors, which bufferizes).

Combinators (`map`, `zip_with`, `reduce`, and the retargeted
`splat`/`tabulate`/`fold`) live on `Tensor` and lower 1:1 to linalg
named/generic ops. Element functions must reduce to scalar IR — a
`linalg.generic` body is a region of scalar ops, so a runtime closure is
a type error there, and `ClosureBetaReduction` (today `-Oaggressive`-
gated) becomes mandatory on kernel paths or the frontend inlines the
lambda during elaboration.

## Boundary facilities and reuse

| surface | lowers to | reuse |
|---|---|---|
| `Tensor::of(xs)` | `rc.borrow` + `array.view : tensor` (`to_tensor restrict`) | zero copy |
| `xs.modify(\|t\| …)` | `with_unique_view` tensor body + `materialize_in_destination` | unique → in place, zero allocs; shared → one clone |
| `t.materialize_into(ys)` | same branch on `ys`; runtime shape check → `reussir.panic` on mismatch | as above |
| `t.materialize()` | fresh box + `with_unique_view` + `materialize_in_destination` | `TokenReuse` below |

**Ownership of the view.** `Tensor::of` consumes a dup of `xs`; the
tensor owns that reference until its last use, then `rc.dec`. A later
`xs.set(…)` therefore sees count ≥ 2 and clones instead of mutating the
viewed buffer — the `restrict` contract holds by construction, no static
aliasing proof needed. Guardrail the frontend enforces: one live tensor
materialization per array per scope (two `to_tensor restrict` over
aliasing memrefs is UB).

**The anchoring rule.** Every materialization — the fresh case included —
is anchored as `bufferization.materialize_in_destination %t in %view`
where `%view` is the Reussir-owned `restrict writable` view of the
destination box. One-shot's empty-tensor elimination then folds the fused
loop nest's output directly into the rc payload. The failure shape to
guard against is the result surfacing as a bufferization-owned buffer
that gets copied into the box afterward: it silently doubles every
kernel's cost and is invisible in the surface language.

**Fresh materialize reuses for free.** In
`map(Tensor::of(xs), f).materialize()` where the view is `xs`'s last use,
the tensor's rooted dup dies at the kernel boundary; its `rc.dec` token
(dynamic `token<align, ?>` or an exact static size) is a donor for the
materialize's `token.alloc`. Same shape → same bin → `token.realloc`
pointer-identity: the kernel writes into the buffer it read from. This
requires the usual window discipline — the dec and the alloc sit in the
same reuse window with no intervening call — which the lowering controls
since it emits both boundary ops.

## Pipeline and registry

- Register tensor/linalg/bufferization in the CAPI registry, keeping the
  invariant that no op of theirs survives past
  `convert-bufferization-to-memref` — then `reussir-convert-to-llvm`
  (which walks loaded dialects for `ConvertToLLVMPatternInterface`)
  never meets their promised interfaces, preserving the reason they were
  excluded.
- Insert, between the `kernel` transform anchor and `SCFToControlFlow`:
  `one-shot-bufferize`, the buffer-deallocation pipeline,
  `convert-linalg-to-loops`, `convert-bufferization-to-memref` (+
  canonicalize/cse) — the sequence `array_tensor_view_e2e.mlir`
  hand-spells, plus deallocation, which that test dodges only because
  matmul-into-destination has no intermediates. The anchor thereby fires
  on linalg-on-tensors: tiling/fusion via the surface `transform [{ … }]`
  item at upstream's intended scheduling point, before bufferization.
- Fusion-surviving intermediates are plain `memref.alloc`s — scope-
  confined, invisible to rc and to token reuse. Acceptable first;
  unifiable later by pointing bufferization's allocation hooks at
  `__reussir_allocate`.
- `--token-reuse-remarks` learns to tag materialization sites, so "did
  my kernel run in place" is checked from the remarks JSON, not by
  reading LLVM IR.

## Deferred

- `kernel fn` (tensors in function signatures): needs
  function-boundary bufferization or mandatory inlining; region-local
  tensors sidestep it entirely, so it waits until composition pressure
  is real.
- `vector`-dialect lowering and GPU targets: the linalg layer is the
  entry ticket; neither dialect is registered today.
- Stride-0 broadcast views: borrowed-only if ever (aliasing vs. the
  drop traversal), see `docs/design/dynamic-extent-arrays.md`.
- Dynamic shapes in kernels compose with the dynamic-extent array work:
  `tensor<?x?xf64>` views fall out of the strided-header box design.
