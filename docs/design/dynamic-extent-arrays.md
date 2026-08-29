# Dynamic-extent arrays: the strided-header box

Extends `!reussir.array` from statically shaped only (#344 Phase A) to
per-dimension dynamic extents (`!reussir.array<? x f64>`), keeping rank
static, exactly as memref spells it. The motivation is ecosystem-shaped:
the array view is already a memref, and once extents can be runtime
values an rc-managed Reussir array is a legal operand for the entire
memref/tensor-consuming MLIR universe — `linalg` on both forms, and the
OpenMP/Async/Enzyme directions named in AGENTS.md all speak memref. The
fixed-shape restriction, not the rc box, is what keeps Reussir arrays a
niche citizen today. Growable buffers are explicitly *not* this feature:
`std::collections::cow::vec` remains the growable story, and nothing here
adds a resize operation.

## Box layout: the header is the memref descriptor

A dynamic array's rc box stores the full strided-memref encoding —
offset, sizes, strides, static dims included — and derives only the two
descriptor pointers (both are `box + header`):

```
{ i32 count | pad to align(index) | index offset
  | index size[0..r] | index stride[0..r] | pad to align(elem) | payload }
```

The header is exactly what `memref.extract_strided_metadata` returns,
minus the base buffer. It is never a hand-computed byte table: the
members `{i32, index, index×r, index×r}` go through `deriveCompoundLayout`
against the module's `dlti.dl_spec`, so index width and padding follow
the target — on a 32-bit target index is 4 bytes and the header has no
padding at all. This is the same discipline that makes `str`'s `{ptr,
len}` lowering take `converter.getIndexType()` and that
`reussir-convert-to-llvm` exists to preserve (its type converter is built
from the stamped `llvm.data_layout`, not upstream's frozen 64-bit
default). The header prefix is static per rank, so `RcBoxType` keeps
answering alignment queries; only the tail size becomes dynamic.

Static arrays keep today's headerless box and identity-layout
`memref<NxT>` view — no regression, and the two forms stay distinct
types.

## Views and projection

`getArrayViewMemRefType` for a dynamic array returns
`memref<S0x…xT, strided<[?,…], offset: ?>>`: static dims may stay static
in the shape, but the layout is always the dynamic `StridedLayoutAttr`.
`array.view` lowering becomes a straight header copy — GEP to the payload
for both descriptor pointers, then `2r+1` loads — replacing
`MemRefDescriptor::fromStaticShape`. Uniform for every static/dynamic
mix; no constants-vs-loads case split.

`array.project` becomes pure descriptor arithmetic: `offset += i *
stride[0]`, drop the front of sizes/strides. The current lowering's trick
of folding the row shift into the aligned pointer and pinning descriptor
offset 0 is deleted for dynamic arrays — the offset field is live, and
consumers reach elements through `getStridedElementPtr` as upstream
intends. Bounds checks compare against header loads instead of constants,
feeding the same cold-`reussir.panic` guard.

The strided layout also buys mutation-free layout ops: transpose,
reverse, and leading-dim slice rewrite the header only, gated by the same
unique-or-clone discipline as `array.with_unique_view` (the header is
part of the CoW'd box). Stride-0 broadcast is excluded from owning boxes:
aliased elements would break the drop traversal's exactly-once contract.

## Invariants

- **Canonical on construction.** Constructors and clones always produce
  offset 0 and row-major suffix-product strides. Non-canonical headers
  arise only from in-place restride of an existing allocation, which
  never changes the footprint — so `token.alloc` size is always
  `header + ∏sizes × elemsize`, computed on the canonical form. The
  touched-extent formula (`offset + Σ(size_i − 1)·stride_i + 1` elements)
  is for verifier/debug assertions only, never allocation.
- **Clone compacts.** The `with_unique_view` clone branch checks the
  loaded header: canonical → flat `ref.memcpy` with an SSA length;
  non-canonical → a strided-to-canonical copy nest, and the clone's
  header is written canonical. Sharing never entrenches a degenerate
  layout.
- **Tokens.** `rc.dec` of a dynamic array produces `token<align, ?>`,
  the existing universal fallback donor; `token.alloc` gains an SSA size
  operand (the `__reussir_allocate` entry point already takes a runtime
  size — only the constant-gated `_small` fast path stays static-only).
  Boxes past `kAllocatorBinModelMax` are excluded from reuse pairing,
  implementing the #344 integration note.

## What has to change (audited)

- `ArrayType::verify` and the parser/printer: accept and print `?`
  (`ShapedType::kDynamic`); today any negative extent is rejected.
- `ArrayType::getTypeSizeInBits`: must fail loudly on a dynamic shape
  instead of multiplying the sentinel; audit every caller
  (`TokenInstantiation::getTokenType`, the `ConvertToSTD` clone branch's
  `getTypeSize(rcBoxType).getFixedValue()`, `memberStorageType`).
- `RcBoxType::getTypeSizeInBits`/`getABIAlignment`: the
  `llvm_unreachable("must have a fixed size")` paths route to the
  runtime size computation; alignment is already shape-independent.
- Type converter: nested `LLVM::LLVMArrayType` has no dynamic form — the
  dynamic box lowers as the header struct plus a trailing
  zero-length-array-style tail.
- `emitArrayElementTraversal` (drop/acquire): loop bounds from header
  loads; the ≤4 unroll threshold gates to static shapes; iteration walks
  the logical index space through strides (exactly-once by the
  no-aliasing invariant).
- Pipeline: insert `expand-strided-metadata` before
  `reussir-convert-to-llvm`. No memref-level pass runs today, and with
  strided layouts first-class, transform-anchor tiling can legitimately
  leave `memref.subview`/`extract_strided_metadata` in the IR.
- Frontend: `TyKind::Array` dims become per-dimension static/dynamic;
  `eval_extent` stops requiring literals (a non-literal extent makes the
  dim dynamic and the value an argument of `splat`/`tabulate`); add an
  `array::dim` intrinsic; lengths stay `u64` until `usize` lands.

## Status

Frontend landed: `?` extents, the `DYNAMIC_EXTENT` sentinel through the
type system, leading runtime extents on `splat`/`tabulate`, `array::dim`,
display/mangling/textual-IR spellings, and `err(…)` stubs at every
lowering path (`--emit mlir` reports "do not lower yet"). The
strided-header box and the descriptor lowering are the open backend half.

## Deferred

- Const-generic extents (the monomorphization half of #344 Phase C)
  produce *static* arrays and stay orthogonal; do not couple.
- Borrowed subviews that outlive an expression re-raise the `str`
  lifescope split (`docs/design/str.md`); owner-level ops ship first.
- A growable/resizable array is a different feature with a different
  contract (`token.realloc` resizes dead donors only, #362); out of
  scope here.
