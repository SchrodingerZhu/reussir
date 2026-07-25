# Shipping the Runtime as Source: Using System Rust/Cargo

Status: draft / proposal

## Motivation

Reussir currently ships a private Rust "sysroot": the nightly-release workflow
packages `rustc`, `cargo`, `rustdoc`, `librustc_driver`, the whole
`rustlib` tree, the toolchain's `libstd-*.so`, plus a prebuilt
`libreussir_rt.so` and the entire `target/<profile>/deps` directory of rlibs.
The compiler then hunts for these at run time via relative-path heuristics
(`RUSTC_HINTS` / `RUSTC_DEPS_HINTS` in `lib/RustCompiler/RustCompiler.cpp`).

This is fragile and heavy:

- The rlib/`deps` directory is only usable by the *exact* rustc that produced
  it (rlib metadata and symbol hashes are version-locked), forcing us to ship
  a full pinned toolchain alongside every release.
- Emitted programs link `libreussir_rt.so` as a Rust *dylib*, which drags in a
  runtime dependency on the toolchain's `libstd-*.so` — another reason the
  toolchain must travel with the release.
- Relative-path discovery breaks whenever the binary layout changes.

Instead, we adopt the Koka model (Koka ships the `kklib` runtime as C source
and compiles it with the user's system cmake/cc): **ship `reussir-rt` as
vendored Rust source and build it on demand with the user's system
rust/cargo.** The compiler stops shipping any Rust toolchain bits.

## Current couplings (what makes this nontrivial)

Polymorphic FFI is the sensitive consumer. The `CompilePolymorphicFFI` pass
monomorphizes Rust source templates and invokes
`rustc --crate-type cdylib --emit=llvm-bc -L <deps>` at Reussir compile time,
then embeds the bitcode in the module; `gatherCompiledModules` later
`parseBitcodeFile`s and `llvm::Linker`-merges it into the final LLVM module.
This creates three hard version couplings:

1. **rlib ↔ rustc**: `extern crate reussir_rt` in a template resolves against
   an rlib that must have been built by the same rustc version that compiles
   the template.
2. **Symbol hashes ↔ rt build**: the emitted bitcode references non-generic
   `reussir_rt` internals through Rust-mangled names that embed the crate
   metadata hash (SVH) of the *specific* rt build it was compiled against.
   The final binary must link that same build's staticlib/dylib, or symbols
   go unresolved.
3. **LLVM bitcode ↔ Reussir's LLVM**: `parseBitcodeFile` in Reussir's LLVM
   (currently pinned to LLVM 22) can only read bitcode from an equal-or-older
   LLVM. A system rustc built on a newer LLVM emits bitcode we cannot parse;
   today there is no fallback.

Additionally, templates today require **nightly** (`#![feature(linkage)]` for
`weak_odr` dedup) and compile at the implicit **edition 2015** (no `--edition`
flag), and the runtime's sanitizer variants need `-Zsanitizer`/`-Zbuild-std`.

The good news: `crates/reussir-rt/src` contains no `#![feature]` attributes.
The only nightly dependencies are the `mlir_sync/nightly` cargo feature
(optimization-only, per its own comment), the template `#![feature(linkage)]`,
and the sanitizer builds.

## Design

### 1. Distribution contents

Ship, under `<prefix>/share/reussir/runtime/`:

- The `reussir-rt` crate source (self-contained: its own `Cargo.toml`,
  standalone — not a workspace member — plus `Cargo.lock`).
- A `vendor/` directory produced by `cargo vendor`, and a checked-in
  `.cargo/config.toml` pointing at it. This makes runtime builds fully
  offline and removes the git dependency on `mlir-sync` from user machines.

Stop shipping: `rustc`, `cargo`, `rustdoc`, `rustlib/`, toolchain `libstd`
dylibs, the prebuilt `deps/` directory, and the prebuilt `libreussir_rt.so`.

The shipped source is version-locked to the compiler: the compiler embeds
templates that call specific rt APIs, so a given `rrc` release only consumes
the rt source it ships with.

### 2. Toolchain policy: stable Rust with an MSRV

To genuinely use "whatever rust is installed", the runtime and the polyffi
templates must build on **stable**:

- Declare `rust-version` (MSRV) in the rt `Cargo.toml`; the driver checks
  `rustc --version` against it with a clear diagnostic
  ("Reussir requires Rust >= 1.NN; found ...").
- Default the `mlir_sync` dependency's `nightly` feature off; re-enable it
  automatically when the resolved toolchain is a nightly (it only gates
  optimizations).
- Remove `#![feature(linkage)]` from templates. `weak_odr` dedup moves into
  the compiler: `gatherCompiledModules` already owns an `llvm::Module` per
  polyffi blob, so after import it rewrites the linkage of the generated
  `__reussir_polyffi_*` definitions (and the record-type `Drop`/`Clone` glue)
  to `weak_odr`. Same effect, no nightly attribute. (For the object-file
  fallback in §5, dedup is instead guaranteed by hashing monomorphized
  sources and compiling each distinct instantiation once per link unit.)
- Pass `--edition 2021` explicitly to the polyffi rustc invocation
  (`extern crate` remains valid; implicit edition-2015 goes away), and pass
  `--extern reussir_rt=<exact rlib>` + `-L dependency=<deps dir>` instead of
  bare `-L` directory hunting.
- Sanitized runtime variants stay a *developer* feature that requires a
  nightly toolchain; they are not part of the shipped product.

`rust-toolchain.toml` remains for developing Reussir itself, but CI gains a
job that builds `reussir-rt` and runs the polyffi integration tests against
current **stable**, so stable compatibility cannot rot.

### 3. Toolchain resolution

Replace `RUSTC_HINTS`/`RUSTC_DEPS_HINTS` with a resolution module used by all
consumers (driver, tests, REPL):

1. Explicit flag (`rrc --rustc/--cargo`) or config file entry.
2. `REUSSIR_RUSTC` / `REUSSIR_CARGO` environment (kept for compatibility;
   lit tests keep using it to pin the in-repo toolchain).
3. `CARGO`/`RUSTC` env, then `PATH` lookup (respects rustup shims, so a
   `rust-toolchain.toml` in the user's project still works naturally).

At resolution time run `rustc -vV` once and record: release version, commit
hash, host triple, and **LLVM version**. This record feeds the cache key (§4)
and the bitcode/object mode decision (§5). A `rrc doctor` subcommand prints
the resolved toolchain, cache state, and chosen polyffi mode.

### 4. On-demand runtime build with a shared cache

The driver guarantees, before any polyffi compile or link-flag query, that
the runtime is built for the resolved toolchain:

- Cache location: `${XDG_CACHE_HOME:-~/.cache}/reussir/rt/<key>/` where
  `<key> = hash(rustc commit hash, rt source hash, target triple, profile,
  feature set)`.
- Build: run `cargo build --locked --offline --release
  --message-format=json` in the shipped source dir with
  `CARGO_TARGET_DIR=<cache>/target`. Cargo's own fingerprinting handles
  staleness and concurrent invocations (file locking is built in).
- Parse the JSON messages to locate artifacts exactly (no directory
  guessing) and write a small `manifest.json` in the cache root recording:
  rlib path, staticlib path, cdylib path, `deps/` dir, rustc version, rustc
  LLVM version.
- First-use UX mirrors Koka's first `kklib` build: a one-time
  "Building Reussir runtime (once per Rust toolchain)..." message.
  `rrc rt build` allows explicit prebuild in CI images/containers.

Everything downstream reads the manifest:

- **Polyffi compiles** pass `--extern reussir_rt=<rlib>` and
  `-L dependency=<deps>` from the manifest. Because the rlib was built by
  the same rustc now compiling the template, couplings (1) and (2) hold by
  construction.
- **Final linking** stays with the user's C toolchain, but the flags come
  from the compiler: `rrc --print link-flags` (analogous to `llvm-config`)
  emits `-L <cache> -lreussir_rt <platform sys libs>`. Tests and build
  systems consume this instead of hardcoding `%library_path`.

### 5. Runtime linkage: staticlib by default

Switch the advertised link artifact from the Rust *dylib* to the
**staticlib** (`crate-type` already includes it). This removes the
`libstd-*.so` runtime dependency entirely — the other half of why the
toolchain had to be shipped. The cdylib/dylib remains available for
embedders and the REPL (§7). `--print link-flags` adds the platform system
libraries a staticlib needs (`-lpthread -ldl -lm`, the existing Windows and
macOS lists).

### 6. Polyffi output modes: bitcode with an object fallback

Coupling (3) — LLVM bitcode version — is handled by capability detection:

- **Bitcode mode (preferred).** If the resolved rustc's LLVM major version
  ≤ Reussir's LLVM major version, keep today's flow:
  `--emit=llvm-bc`, embed, parse, `llvm::Linker` merge. Cross-language
  inlining and whole-module optimization are preserved. Since Reussir tracks
  recent LLVM (22) and stable rustc lags LLVM releases, this is the common
  case.
- **Object mode (fallback).** Otherwise, compile the monomorphized module
  with `--emit=obj` and carry the bytes in a new `compiledObject` variant on
  `ReussirPolyFFIOp`. At emission, objects are written next to the main
  object and included in `--print link-flags` output (and for the JIT, added
  via `LLVMOrcLLJITAddObjectFile`). This sacrifices cross-language inlining
  but works with any rustc. The mode is reported by `rrc doctor` and
  overridable by flag for testing.

In both modes, distinct instantiations are deduplicated before invoking
rustc by hashing the monomorphized source, so each template instantiation is
compiled at most once per compilation.

### 7. REPL / JIT

Today the REPL links `reussir-rt` in-process (a cargo path dependency built
by whatever toolchain built the compiler distribution) and registers a fixed
C-symbol list with ORC; polyffi is not exercised in the REPL. Under the new
model this keeps working unchanged for non-polyffi use.

Polyffi-in-REPL, when it lands, cannot resolve Rust-mangled internals
against the embedded rt (different crate hashes than the user-cached build).
The plan is to unify: the REPL resolves runtime symbols by loading the
cached toolchain-matched cdylib through the existing
`OrcJit::add_library` / `runtime_library_path` path (pointing at the §4
cache), instead of embedding a second copy — one runtime instance, same
artifacts as AOT. This is deliberately a separate, later phase; mixing an
embedded rt and a dlopened rt in one process (two allocators, two sets of
globals) is the failure mode to avoid.

### 8. Build system, tests, CI

- `crates/reussir-rt/CMakeLists.txt`: delete the rustup fetch and
  toolchain-copy machinery (`reussir-rust-toolchain-fetch`,
  `reussir-rust-install`, ~150 lines). Keep the cargo build of rt into the
  build tree for the in-repo dev/test flow, and add install rules that stage
  the rt source + `vendor/` into `share/reussir/runtime/`.
- lit: keep exporting `REUSSIR_RUSTC`/`REUSSIR_RUSTC_DEPS` pinned at the
  developer toolchain and build-tree artifacts (hermetic in-repo tests), and
  add integration tests for the resolution/caching path itself plus a
  stable-toolchain polyffi test.
- `flake.nix`: drop the `build/.rustup` symlink hack once CMake no longer
  checks for a cached toolchain.
- `nightly-release.yml`: stop packaging toolchain binaries and `rustlib`;
  package the rt source + vendor tree instead (large artifact-size win).
  Release smoke test: install the artifact into a container with only
  system rust (both a rustup stable and a distro rust), compile and run a
  polyffi program.

### 9. Migration phases

1. **Stable-ize the contract.** Templates lose `#![feature(linkage)]`
   (linkage rewrite lands in `gatherCompiledModules`); explicit `--edition`
   and `--extern`; `mlir_sync/nightly` off by default; CI job building rt +
   polyffi tests on stable.
2. **Toolchain resolution + rt cache.** New resolution module, cargo-driven
   cache with manifest, `rrc rt build` / `rrc doctor` /
   `--print link-flags`; polyffi and docs rewired to the manifest;
   staticlib becomes the documented default link artifact.
3. **Object-mode fallback** for LLVM version mismatch, including the
   `compiledObject` op variant and JIT object loading.
4. **Ship it.** Release/CI/packaging changes; delete toolchain-shipping
   paths and `RUSTC_*_HINTS`; docs (www chapter) updated.
5. **(Later) REPL unification** on the cached runtime for polyffi-in-JIT.

Phases 1–2 are independently landable and immediately de-risk the current
setup; the shipped-toolchain path can remain as a deprecated fallback until
phase 4.

## Open questions / decisions

- **MSRV choice**: pick the oldest stable that supports edition 2024 and
  the rt's dependency tree (≥ 1.85); verify `stacker`, `libmimalloc-sys`,
  and `mlir_sync` (sans `nightly`) build there.
- **Cache vs per-project builds**: this design uses a shared per-user cache
  keyed by toolchain+source hash (Koka builds per project). Cargo's target
  dir gives us safe sharing for free; a `--rt-target-dir` escape hatch can
  serve hermetic build systems (Nix, Bazel) that want project-local builds.
- **Cross-compilation**: the cache key already includes the target triple;
  polyffi invocations need `--target` propagation and the final link needs
  a target-appropriate staticlib. Deferred until cross-compilation is a
  supported flow generally.
- **Windows**: staticlib-by-default sidesteps the `reussir_rt.dll` +
  `libstd` DLL story; MSVC vs GNU toolchain flavor must match the user's
  C toolchain — detect from the rustc host triple and warn on mismatch.
