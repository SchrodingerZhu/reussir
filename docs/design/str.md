# `str`: the immutable static string

Reussir's `str` is exactly a string *literal*: a borrowed `{ptr, len}` view
of an immutable global, interned by content (BLAKE3 `StringToken`, one
`reussir.str.global` per distinct payload, shared across compilation units
by construction of the name). There is no mutation and no concatenation on
`str` — every growable or mutable string lives outside the language, as an
FFI-wrapped Rust `String` (`reussir_rt::collections::string::String`, an
rc-boxed copy-on-write value crossing as an opaque `ffi_object`) or a
future functional rope built on the `cow`/`pure` collections.

## What a `str` can do

- **Exist and match**: `"hello"` types as `str`; `match` on string patterns
  compiles to a compressed-trie dispatch (`reussir.str.select`).
- **Be read**: `core::intrinsic::str::{len, byte_at, slice}` — byte
  granularity throughout. `byte_at` is bounds-checked (`0` out of bounds),
  `slice` clamps past the end and may split a multi-byte character; both
  mirror the dialect ops one for one.
- **Compare**: the full builtin tower — `PartialEq`/`Eq`/`PartialOrd`/`Ord`
  (`core::cmp`). Equality and ordering are byte-wise lexicographic with
  length as the shared-prefix tiebreak, through `reussir.str.equal` /
  `reussir.str.compare` (`compare` follows the `memcmp` sign convention).
  Both expand in `reussir-convert-to-std` — the layering every high-level
  op follows — into outlined internal `func.func` helpers built from `scf`:
  a `str.ref_eq` view-identity fast path (same `{ptr, len}` pair, no byte
  scan), the length gate for equality, then an `scf.while` byte scan over
  the `unsafe_byte_at` residue. Only the straight-line residues reach the
  LLVM conversion; `ref_eq` lowers to a two-field compare.
- **Hash**: `std::hash::Hash for str` feeds the caller's hasher one `u64`
  content digest computed by the runtime (`Str::content_hash`) — a proper
  hasher consumes bytes in word-sized chunks, which needs raw-pointer
  reads the surface language cannot express yet (TODO: go native when
  bare-pointer intrinsics land). The outer hasher's seed still mixes the
  digest. The ordered containers key on the runtime `Str`'s byte-wise
  `Ord`, which matches `str.compare`.
- **Cross the FFI boundary**: a `str` parameter or return renders as
  `::reussir_rt::collections::string::Str<'static>` — a `#[repr(C)]`
  `{ptr, len}` bit-identical to the lowered value. Sound because every
  surface `str` today is a `'static` global literal.
- **Sit in records**: `str` carries `DataLayoutTypeInterface`
  (two pointer-words) and is trivially copyable — no rc glue, plain loads
  and stores.

## Lifescope

The dialect type is `!reussir.str<global | local>`; the frontend type
carries no lifescope and every surface value is `<global>` (`<local>`
appears only transiently inside `match` lowering, where `str.cast`
downgrades before `str.select`). When borrowed/derived strs surface —
e.g. views into an FFI `String` — the frontend type needs the lifescope
split, and the `'static` FFI spelling and the unconditional `Sync`
verdict (`semi/traits/sync.rs`) must be revisited together.

## Deferred

- Debug info for `str` locals: needs a pointer-carrying DBG attribute in
  the dialect (`ReussirAttrs.td` has int/fp/record/boxed only); `-g`
  currently omits `str` variables, unchanged from before.
- `OnceODR` dedup of `str.global` across compilation units (TODO in
  `BasicOpsLowering.cpp`): today identical literals in two units are two
  private globals — correct, not pointer-identical.
- `const` items: a named literal is spelled as a nullary `fn` today.
- `panic` with a message: the intrinsic takes no arguments; the dialect op
  already carries a message attribute.
- Surface `startswith`: the dialect op takes an attribute pattern;
  `match` covers prefix dispatch, and a dynamic prefix test can be built
  from `len`/`slice`/`==`.
