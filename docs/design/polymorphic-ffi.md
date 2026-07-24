# Polymorphic FFI

## Motivation

Reussir's escape hatch to the systems world is Rust. Rather than pinning a
uniform boxed ABI (Lean4-style) on every foreign call, the compiler
*monomorphizes the foreign side too*: for each ground instance of an FFI
function it renders a small self-contained Rust source (a **texture**),
compiles it to LLVM bitcode with `rustc`, and links the bitcode into the
final module. Cross-boundary calls are ordinary same-module calls after
linking — LLVM inlines through them at `-O`.

The MLIR layer for this predates the frontend surface: `reussir.polyffi`
(texture → bitcode → link), `reussir.ffi_object` (an opaque foreign payload
behind an `rc`), and `reussir.trampoline` (the platform-independent
boundary). This design gives that machinery a source-level surface.

## Surface

```rust
// Opaque Rust items shared by this file's wrappers: use declarations,
// helper functions, ... Spliced ahead of every wrapper the file generates.
extern "rust" [{
    use reussir_rt::collections::vec::Vec as RVec;
}];

// An opaque FFI type: a field-less struct aliasing a Rust type.
#[ffi(rust = "::reussir_rt::collections::vec::Vec")]
pub struct Vec<T>;

// An imported function: a Reussir signature with an inline Rust body.
#[ffi(import)]
pub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ RVec::push(v, x) }];
```

* `extern "rust" [{ ... }];` — a **foreign prelude**, file-scoped. Only the
  `rust` ABI is supported.
* `#[ffi(rust = "path")] struct Name<T...>;` — an **opaque record**. No
  fields, no constructors, no patterns, no projections; always `[shared]`.
  The `path` must be an absolute (`::`-rooted) path to a Rust type that is a
  `#[repr(transparent)]` wrapper over `reussir_rt::rc::Rc` (see the contract
  below).
* `#[ffi(import)] fn f<T...>(...) -> R [{ body }];` — an **imported
  function**. The signature is elaborated and type-checked as ordinary
  Reussir; the body is opaque Rust, evaluated with the parameters in scope
  at their Rust spellings. `[:T:]` in the body substitutes the instance's
  Rust spelling of generic `T` (rarely needed — inference from the bound
  parameters usually suffices).

## Ownership: the boundary consumes

The ownership system is untouched: **every call consumes its arguments**,
imported functions included. The generated wrapper receives each argument
owning one reference count; ordinary Rust move semantics take over from
there — dropping releases, `.clone()` acquires. Nothing borrow-shaped ever
crosses the boundary (that would need borrow-aware analysis, out of scope);
`&self` borrows appearing inside a wrapper body act on values the wrapper
already owns and end before it returns. The Perceus analysis inserts the
usual `inc` before a call when the caller still needs the value afterwards.

## Clone/drop across the boundary

Both directions reduce to reference counts:

* **Reussir holding a Rust value** (`Vec<T>`): the value is
  `!reussir.rc<!reussir.ffi_object<"name", @hook>>`. `rc.inc` bumps the
  count inline — sound because the exposed Rust type is a transparent
  wrapper over `reussir_rt::rc::Rc`, whose `u32` count sits at offset 0,
  mirroring the compiler's box header. `rc.dec` lowers to a call of the
  per-instance drop hook `<inst>_ffi_drop`, a generated Rust function that
  takes the value and drops it.
* **Rust holding a Reussir value** (`Vec<List>`): the element is a generated
  `#[repr(transparent)]` pointer wrapper named by the instance's v0 symbol,
  whose `Clone`/`Drop` call compiler-emitted glue functions —
  `<inst>_ffi_acquire` / `<inst>_ffi_release`, tiny Reussir functions
  containing a single `rc.inc` / `rc.dec` (so fused headers, atomicity, and
  drop dispatch are handled by the compiler's own lowering, uniformly for
  structs and enums).

## The boundary convention

Wrappers use the trampoline's platform-independent boundary — only integers
and pointers cross, so `rustc`'s and the compiler's lowerings agree by
construction on every target (no `sret`/`byval`, no C aggregate
classification):

* **Trivial** (fewer than four parameters, all integer-like — integers,
  `bool`, `char`, and rc pointers — and an integer-like or `unit` result):
  the wrapper has the native signature and the import trampoline forwards
  directly.
* **Nontrivial**: parameters pack into a `#[repr(C)]` struct passed by
  pointer; a non-integer-like result returns through a leading out-pointer.

The frontend's classification (`full/ffi.rs::classify_trivial`) mirrors
`evaluateCABISignatureForC` in
`lib/Conversion/BasicOpsLowering/CABISignatureConversion.cpp`; the e2e
tests pin both sides. Per instance, the compiler emits: the bodyless native
declaration (what callers call), a `reussir.trampoline import "C"
@<sym>_ffi = @<sym>` (whose lowering materializes the native definition,
packing and calling the boundary symbol), and a `reussir.polyffi` texture
defining `@<sym>_ffi` in Rust with `#[linkage = "weak_odr"]` (duplicate
monomorphizations across units dedupe at link).

## Symbols

All names derive from v0 manglings by suffixing — injective, since no
valid v0 mangling is a prefix of another:

| symbol                      | role                                    |
|-----------------------------|-----------------------------------------|
| `<fn instance>`             | the native function (callers use this)  |
| `<fn instance>_ffi`         | the Rust-side boundary wrapper          |
| `<record instance>_ffi_drop`| an opaque instance's drop hook          |
| `<record instance>_ffi_acquire` / `_ffi_release` | Reussir rc glue    |

## Boundary types (v1)

Signatures of imported functions may use: integers, `f32`/`f64`, `bool`,
`char`, `unit` (return only), opaque `#[ffi]` types, and **shared Reussir
records** (struct or enum). Everything else — value/regional records,
`Nullable`, `Arc`, `str`, cells, arrays, closures — is rejected with a
diagnostic at the ground instance (monomorphization), where the offending
type is concrete.

## Exposing a new type

Nothing in the compiler is Vec-specific. To expose another Rust type:

1. wrap it in the runtime (or any crate on the `rustc` deps path) as a
   functional value: `#[repr(transparent)] pub struct X(Rc<Inner>);` with
   owning operations taking `self` and returning `Self`
   (`reussir_rt::collections::vec::Vec` is the model);
2. declare it: `#[ffi(rust = "::path::to::X")] pub struct X<...>;`
3. import its operations as `#[ffi(import)]` functions.

The `Rc`-wrapper contract is what makes the inline `rc.inc` sound and keeps
clone free of FFI calls. (An acquire hook for non-`Rc` foreign types is a
possible extension; it was deliberately left out of v1.)

## Pipeline and toolchain

The `reussir-compile-polymorphic-ffi` pass substitutes and compiles each
texture with `rustc`. The driver flags `--polyffi-rust-path` (the `rustc`
executable; a bare name resolves through `PATH`) and `--polyffi-libdir`
(the package directory passed as `-L`, holding `libreussir_rt` and
friends) pin the toolchain explicitly; they take precedence over the
`REUSSIR_RUSTC` / `REUSSIR_RUSTC_DEPS` environment variables, which in
turn beat the built-in probe list. The flags are the first step toward
building against a host `cargo`/`rustc` toolchain instead of shipping a
full Rust sysroot alongside the compiler. `translateToModule` links the
gathered bitcode into the final LLVM module. The JIT/REPL path does not
gather polyffi modules yet and rejects programs containing them.

## Implementation map

* surface/parsing: `crates/reussir-syntax` (`ExternSourceStmt`, foreign
  function bodies, field-less structs, `key = "value"` attribute args);
* validation and collection: `crates/reussir-core/src/semi/ctxt.rs`
  (`ffi_attr`, `validate_ffi_record`, `validate_ffi_function`,
  `FfiPrelude`/`FfiImport` tables, `RecordFields::Opaque`);
* rendering: `crates/reussir-core/src/full/ffi.rs` (Rust spellings,
  classification, wrapper/drop textures), driven per instance from
  `full/mono.rs`; MIR carries `ffi_imports`/`ffi_textures`/`ffi_rc_glue`
  and `RecordLayout::Opaque`, round-tripping through both textual IRs;
* codegen: `crates/reussir-codegen/src/lower/{mod,ty,expr}.rs` (rc'd
  `ffi_object` types, polyffi/trampoline emission, glue and hook
  functions);
* MLIR: `reussir.trampoline` import direction materializes the native
  marshaling body (`lib/Conversion/BasicOpsLowering/BasicOpsLowering.cpp`).
