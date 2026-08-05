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
* **Rust holding a Reussir value** (`Vec<List>`): the element is rendered as
  `reussir_rt::bridge::Bridge<Inner>`. `Inner` is the generated
  `#[repr(transparent)]` one-pointer wrapper named by the instance's v0
  symbol. It remains the pointer's owner: its `Clone`/`Drop` call
  compiler-emitted `<inst>_ffi_acquire` / `<inst>_ffi_release` functions,
  tiny Reussir functions containing a single `rc.inc` / `rc.dec` (so fused
  headers, atomicity, and drop dispatch are handled by the compiler's own
  lowering, uniformly for structs and enums). `Bridge` is another transparent
  layer; its `Clone` delegates to `Inner` and ordinary field destruction drops
  `Inner`. It stores no descriptor, callback, or additional pointer.

## Comparison bridges for foreign containers

Rust containers place standard-trait bounds on their elements. For example,
`std::collections::BTreeSet<T>` requires Rust's `Ord`, while a shared Reussir
record implements Reussir's `core::cmp::Ord`. The two traits are distinct
across the language boundary. Emitting every standard-trait adapter directly
on every generated pointer wrapper would duplicate the generic Rust-facing
part of the bridge in each PolyFFI texture.

The runtime therefore owns a transparent adapter and four local behavior
traits:

```rust
pub trait PartialEqBridge {
    fn eq_bridge(&self, other: &Self) -> bool;
}
pub trait EqBridge: PartialEqBridge {}
pub trait PartialOrdBridge: PartialEqBridge {
    fn partial_cmp_bridge(&self, other: &Self) -> Option<std::cmp::Ordering>;
}
pub trait OrdBridge: EqBridge + PartialOrdBridge {
    fn cmp_bridge(&self, other: &Self) -> std::cmp::Ordering;
}

#[repr(transparent)]
pub struct Bridge<T>(T);

impl<T: PartialEqBridge> PartialEq for Bridge<T> { /* delegate to T */ }
impl<T: EqBridge> Eq for Bridge<T> {}
impl<T: PartialOrdBridge> PartialOrd for Bridge<T> { /* delegate to T */ }
impl<T: OrdBridge> Ord for Bridge<T> { /* delegate to T */ }
```

PolyFFI emits the Reussir-specific bridge-trait implementations on `Inner`,
which is local to the generated texture; `reussir-rt` owns `Bridge` and the
standard Rust implementations. This division satisfies Rust's coherence rules
without a runtime trait object. `Bridge<Inner>` is still exactly one pointer.
It does not use `reussir_rt::rc::Rc<Marker>`: that type owns a Rust `RcBox` and
would run the wrong payload destruction for a compiler-owned Reussir record.

### Uniform rendering and bound discovery

A shared Reussir ground type is rendered uniformly as `Bridge<Inner>` at every
position in a texture, whether or not a particular position supplied the
comparison requirement. Thus an import such as

```rust
fn insert<T: Ord>(set: BTreeSet<T>, value: T) -> BTreeSet<T>
```

has one consistent Rust shape:

```rust
fn insert(
    set: RuntimeBTreeSet<Bridge<KeyInner>>,
    value: Bridge<KeyInner>,
) -> RuntimeBTreeSet<Bridge<KeyInner>>
```

A binder's declared bounds are what a rendered argument must satisfy, and
both kinds of binder count. When rendering an opaque FFI record, the compiler
matches its formal generic parameters to the ground arguments and examines
their declared bounds. An `#[ffi(import)]` signature's own generics are the
other source: `fn same<T: PartialEq>(lhs: T, rhs: T) -> bool` promises
equality that its Rust body then uses directly, with no container in sight.
Comparison lang items are found through trait identity and supertrait
implication, not by spelling. Requirements from repeated appearances of the
same shared-record instance are unioned before its inner declaration is
rendered. Primitive arguments keep their native Rust spelling and use Rust's
built-in comparison implementations; they are never wrapped in `Bridge`.

The comparison tower is normalized and only the necessary foreign entries are
emitted:

| Reussir requirement | generated behavior on `Inner` | foreign entries |
|---------------------|--------------------------------|-----------------|
| `PartialEq` | `PartialEqBridge` | `_ffi_eq` |
| `Eq` | `PartialEqBridge`, `EqBridge` | `_ffi_eq` |
| `PartialOrd` | `PartialEqBridge`, `PartialOrdBridge` | `_ffi_eq`, `_ffi_partial_cmp` |
| `Ord` | all four bridge traits | `_ffi_eq`, `_ffi_cmp` |

For `Ord`, `PartialOrdBridge::partial_cmp_bridge` returns `Some(cmp_bridge(...))`,
so there is no separate partial-order entry. `EqBridge` is marker-only, matching
Reussir's `Eq`; equality is supplied by `PartialEq`. A missing ground Reussir
implementation is diagnosed during monomorphization, before the generated Rust
is handed to `rustc` — once per rejected entry, however many textures and
container instances wanted it.

### Borrowed comparison entries

Rust's comparison methods borrow both operands, whereas Reussir's comparison
methods consume shared values. Each generated entry therefore accepts two raw
RC handles borrowed from `Inner`, performs exactly two compiler-side acquires
(`rc.inc`, one per operand), and directly calls a synthesized semantic adapter.
The adapter consumes those acquired values under the ordinary Reussir ownership
rules and returns a scalar ABI code. The entry performs no matching release:
the selected method and ownership lowering settle the two acquired references.
There are no Rust-side clones, acquire callbacks, dictionaries, or allocations
on this path, and optimized builds can inline through the linked texture.

The scalar contracts are stable and independent of enum layout:

* `_ffi_eq` returns `u8` zero or one after calling the selected
  `PartialEq::eq` implementation;
* `_ffi_cmp` calls `Ord::cmp` and classifies `Ordering` by variant name as
  `Less = -1`, `Equal = 0`, `Greater = 1` in an `i8`;
* `_ffi_partial_cmp` makes one foreign entry call and classifies through
  `PartialEq::eq`, then `PartialOrd::lt`, then `PartialOrd::gt` as needed. Its
  `i8` codes are `Less = -1`, `Equal = 0`, `Greater = 1`, and
  `Incomparable = 2`; Rust maps the final code to `Option<Ordering>`.

The semantic adapter is an ordinary monomorphic MIR function. Consequently its
trait calls use the same selection, ownership, optimization, and diagnostics as
source-level Reussir calls. The borrowed entry itself is small and direct: two
increments, one adapter call, and one scalar return.

### Comparison implementations from another package

No comparison dictionary or global instance registry is added for PolyFFI.
Suppose package A defines a public shared `A::Key` and implements the core
comparison traits, while only package B instantiates
`BTreeSet<A::Key>`. A's RRI already exports the relevant impl metadata and
method prototypes or generic bodies. When B loads that interface, its ordinary
trait database reconstructs those impls. Bridge synthesis in B performs normal
ground trait selection, selects A's implementation, and enqueues the same
method symbol an ordinary trait call would use.

A ground implementation method resolves from A's linked archive; an exported
generic body can be instantiated in B. `rene` supplies the transitive RRI and
archive dependency cone, while a direct `rrc` invocation must pass the
corresponding `--extern` interface and `--link-lib` archive. Bridge entry names
derive from the ground record symbol and use coalescible linkage, so identical
downstream bridge generation does not create a second instance mechanism.

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
| `<record instance>_ffi_eq`  | borrowed equality entry (`u8`)          |
| `<record instance>_ffi_partial_cmp` | borrowed partial classifier (`i8`) |
| `<record instance>_ffi_cmp` | borrowed total-order entry (`i8`)       |

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
(a package directory passed as `-L`, holding `libreussir_rt` and
friends; repeatable, each directory becoming its own `-L`) pin the
toolchain explicitly; they take precedence over the `REUSSIR_RUSTC` /
`REUSSIR_RUSTC_DEPS` environment variables (the latter a list separated
by the platform's environment path separator), which in turn beat the
built-in probe list. The flags are the first step toward
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
  comparison-bound discovery, `Bridge<Inner>` declarations,
  wrapper/drop textures), driven per instance from `full/mono.rs` (ground
  comparison selection and semantic adapters); MIR carries
  `ffi_imports`/`ffi_textures`/`ffi_rc_glue`/`ffi_trait_glue` and
  `RecordLayout::Opaque`, round-tripping through both textual IRs;
* runtime adapters: `crates/reussir-rt/src/bridge.rs` (the four behavior
  traits and transparent `Bridge<T>` standard-trait implementations);
* codegen: `crates/reussir-codegen/src/lower/{mod,ty,expr}.rs` (rc'd
  `ffi_object` types, polyffi/trampoline emission, lifecycle glue, borrowed
  comparison entries, and hook functions);
* MLIR: `reussir.trampoline` import direction materializes the native
  marshaling body (`lib/Conversion/BasicOpsLowering/BasicOpsLowering.cpp`).
