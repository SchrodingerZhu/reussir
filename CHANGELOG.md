# Changelog

## Unreleased

### Language

- `&&` and `||` now short-circuit. Elaboration desugars them into the
  conditional — `l && r` to `if l { r } else { false }` and `l || r` to
  `if l { true } else { r }` — so the right operand becomes a branch body
  and lowers to `scf.if` like every other conditional. They previously
  elaborated to `ArithOp::And`/`Or` alongside the bitwise operators and
  emitted `arith.andi`/`arith.ori`, which evaluate both operands: a guard
  such as `d != 0 && n / d > 1` divided by zero, and an early-exit
  traversal written with `||` visited every node. The eager `ArithOp`
  forms stay reachable from hand-written HIR/MIR text, where they keep
  their bitwise meaning.
- Record fields now carry visibility: fields default to private and accept a
  leading `pub` marker. A private field is accessible (projection,
  assignment, construction) only from the record's defining module and its
  descendant modules, inside a package and across packages alike; the marker
  round-trips through HIR text and ships in `.rri` interfaces.
  Enum-variant payloads are unaffected — variants stay
  exactly as visible as their enum.

- `str` is now usable, not just constructible (see `docs/design/str.md`:
  `str` is exactly the immutable static literal; growable strings stay
  outside the language behind the FFI). String literals gain the read
  intrinsics `core::intrinsic::str::{len, byte_at, slice}` (byte
  granularity, checked/clamped), the full builtin comparison tower
  (`==`/`!=`/orderings/`cmp` via new `reussir.str.equal` and
  `reussir.str.compare` ops, expanded in `convert-to-std` into outlined
  `scf` helpers with a `str.ref_eq` view-identity fast path), record-field
  placement (`str` now carries a data layout and is trivially copyable),
  and FFI crossing: a `str` parameter or return renders as the runtime's
  `#[repr(C)]` `collections::string::Str<'static>`, bit-identical to the
  lowered `{ptr, len}` pair. With `std::hash::Hash for str` (a runtime
  content digest fed to the caller's hasher) a string literal keys every
  container: the hash containers hash it on the Reussir side, and the
  Rust-backed ordered containers order the runtime `Str` byte-wise,
  matching `str.compare`.

### Standard library

- `WavlSet` and `WavlMap` now carry the same interface as the hash
  containers, plus the operations only an ordered container can offer.
  Shared with `HashSet`/`HashMap`: `singleton`, `clear`, `get`/`get_entry`,
  `take`/`remove_entry`, `insert_result`, `insert_with`, `adjust`, `update`,
  `alter`, `keys`, `values`, `filter`, `filter_map`, `map`, `map_values`,
  `map_keys`, `map_entries`, `count_if`, `any`, `all`, `find`, the set and
  map algebra (`union`, `union_with`, `intersection`, `intersection_with`,
  `difference`, `symmetric_difference`), the containment relations, and
  `PartialEq`/`Eq`. Ordered-only: `min`/`max` gain `pop_min`/`pop_max`,
  the neighbour queries `floor`, `ceiling`, `predecessor`, `successor`
  (`*_entry` on the map), the half-open window queries `range` and
  `range_to_list`, the descending traversals `to_list_desc` and
  `fold_right`, and `depth`, which reports the tree height so the balance
  bound is observable from outside the module.
- The ordered containers' `any`, `all`, and `find` stop at the first
  decisive element rather than folding the whole container, and `find`
  returns the least match rather than an unspecified one.
- New `std::collections::cow` module: copy-on-write `HashMap` and
  `HashSet` backed by the runtime's hashbrown table
  (`reussir_rt::collections::{hash_map, hash_set}`, built on
  `hashbrown::HashTable`). Keys hash on the Reussir side with
  `FastHasher` and the digest crosses the FFI boundary with the key, so
  the runtime never hashes: each entry caches its digest, growth and the
  copy-on-write clone reuse it, and key equality — bridged to the
  Reussir `Eq` impl — is consulted only when the cached digests match.
  Core operations for both containers: `new`, `with_seed`, `singleton`,
  `len`, `is_empty`, `clear`, `insert`, `get`/`contains`, `remove`.
  Prefer them for linear build-query-discard usage; retained versions
  pay a full-table clone per divergence, where the `pure` tries share.
- `std::hash` implements `Hash` for the primitives (`u8`–`u64`,
  `i8`–`i64`, `bool`, `char`): each scalar hashes as its own typed
  write, so primitives feed hashers, `write` chains, and hash-container
  key positions directly.
- `std::collections::cow` grows five more containers over the runtime's
  Rust structures (`reussir_rt::collections::{vec, vec_deque, btree_map,
  btree_set, binary_heap}`): `Vec` (push/pop/get/set/insert_at/
  remove_at/last), `VecDeque` (push/pop at both ends, front/back/get),
  `BTreeMap` and `BTreeSet` (insert/get/contains/remove plus
  first/last), and `BinaryHeap` (push/pop/peek — Rust's max-heap;
  reverse the `Ord` for a min-heap). The ordered containers compare
  through the `Ord` bridge; the sequences need no bounds beyond
  FFI-crossability. All follow the module's copy-on-write contract:
  in-place when uniquely owned, one clone when shared, and empty pops
  and lookup misses never clone.

### Compiler and runtime

- Trait impls with method bodies are now accepted for builtin types
  (`impl Hash for u64 { … }`) when the trait is local to the package,
  taking the ordinary impl path with the scalar as the self type; dot
  dispatch on scalars and generic-bound dispatch select them like any
  impl. The intrinsic method-less special form stays reserved to lang
  traits declared by the current package, and implementing an extern
  trait for a builtin now reports the ordinary orphan violation. Member
  paths name the scalar head by its spelling
  (`std::hash::Hash::u64::hash`), which the interface grammar accepts
  as path segments.
- Interface loading registers a dependency's traits before its records,
  so a record binder bounding an interface-local trait
  (`struct Holder<T : Digest>`) no longer crashes every downstream
  compile of that dependency.
- Debug builds no longer fail LLVM verification on cross-package
  monomorphized instances: once a function carries a `DISubprogram`,
  location-less synthesized ops in its body are stamped with the
  conventional line-0 location.

- `rrc --instrument-nonlinear-ffi` (and the matching
  `instrument_nonlinear_ffi` profile knob in `rene`) instruments non-linear
  ffi/array usage: the new `reussir-instrument-nonlinear-ffi` pass guards
  every FFI-import call that consumes an rc'd `ffi_object`/`array` argument
  with a reference-count check, and a count other than one calls the new
  runtime entry point `__reussir_report_nonlinear_usage(file, line,
  col)`, which reports the call's source location on stderr. A still-shared
  value at the consuming boundary is the signature that the foreign side
  degrades to copy-on-write instead of updating in place; the check is
  purely diagnostic and runs after all rc optimization passes, so the
  observed count is the final one.

## 0.1.0

The first tagged release of Reussir: an MLIR-based compiler framework for
token-based memory reuse in RC-managed functional programs, together with the
toolchain built on it.

### Language

- An ownership-aware functional surface language: records (`struct`/`enum`)
  with `value`/`shared`/`regional` capabilities, pattern matching with
  exhaustiveness checking, closures, generics with monomorphization, and
  modules with `mod`/`import`.
- Region-based local mutation: `regional` records and functions, `[field]`
  links, flex/rigid flexivity tracking, and region freezing.
- Polymorphic FFI against Rust (`#[ffi]`, raw MLIR bodies), `extern`
  monomorphization, and cross-package interfaces (`.rri`).
- Prelude intrinsics: cells (with synchronized variants), `Arc`, `Nullable`
  links, fixed-size multidimensional arrays, math intrinsics, and `#[main]`
  entry points.

### Compiler and runtime

- `rrc`, the clang-style pipeline driver over the stage chain
  `.rr → hir → mir → mlir → LLVM IR → objects/executables`, re-entrant at
  every dump; machine-target selection (`--target-triple`, `--target-cpu`,
  `--target-features`) including `wasm32-wasip1[-threads]`.
- The Reussir MLIR dialect: RC-managed objects as first-class IR, token-based
  reuse analysis across branches and regions, inc/dec cancellation, closure
  whole-program devirtualization, and LLVM lowering with debug info.
- `reussir-rt`, the Rust runtime for RC objects and regions, with
  sanitizer-instrumented builds (ASan/LSan/MSan/TSan) and Miri-checked core.

### Tooling

- `rene`, the package manager: Nickel manifests, path dependencies with
  pubgrub version solving, cross-package builds with a process pool (`-j`),
  profiles, machine targets (`--target`), product selection
  (`--bin`/`--lib`), freshness tracking, and JSON introspection
  (`rene inspect`).
- `rrepl`, the JIT-backed REPL with TUI, script mode, and state inspection.
- MLIR tools (`reussir-opt`, `reussir-translate`, `reussir-llvm-opt`) and
  LLDB pretty-printers.

### Platforms

Linux (x86_64, aarch64), macOS, and Windows (MSVC), on LLVM/MLIR 22.
