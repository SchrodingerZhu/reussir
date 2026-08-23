<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="misc/logos/reussir-horizontal-dark.svg">
    <img src="misc/logos/reussir-horizontal-color.svg" alt="Reussir" width="600">
  </picture>
</p>

<p align="center">
  <strong>An MLIR-based compiler framework for token-based memory reuse in RC-managed functional programs</strong>
</p>

<p align="center">
  <em>Frontend in Rust, backend in MLIR/C++, runtime in Rust</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#design-center">Design Center</a> •
  <a href="#repository-layout">Repository Layout</a> •
  <a href="#compilation-guide">Compilation Guide</a> •
  <a href="#testing">Testing</a>
</p>

---

## Overview

Reussir is a research compiler project built around one claim: memory reuse in
reference-counted functional programs should be represented and optimized
directly at the IR level.

The core idea is to make reusable storage explicit as SSA values called
**tokens**, carry those tokens through MLIR structured control flow, and perform
reuse analysis as an ordinary compiler problem rather than as a language-local
trick. In the current design, Reussir includes:

- an ownership-aware functional frontend,
- an MLIR dialect and pass pipeline for RC-managed objects,
- token-based reuse analysis over branches, regions, and loops,
- LLVM lowering and backend-oriented cleanups,
- a Rust runtime for RC objects and related support code,
- extensions for region-local mutable objects and polymorphic FFI.

The frontend language exists to drive and validate this compilation model, but
the main project identity is the compiler framework and object model, not a
surface syntax.

## Design Center

The current implementation is centered on two connected ideas:

- RC-based memory reuse should be explicit in the IR rather than hidden inside a
  language-specific compiler trick.
- region-based local mutation should fit into the same overall object model
  instead of being treated as an unrelated subsystem.

### Core object model

The backend is organized around a small set of pointer and storage concepts:

- `!reussir.rc<T>` for owning reference-counted objects,
- `!reussir.ref<T>` for non-owning inspection references,
- `!reussir.token<align: A, size: S>` for reusable storage,
- `!reussir.nullable<...>` for values that may be null on non-unique paths.

This makes reuse explicit: `rc.dec` may yield reusable storage, consumers can
accept tokens, and later passes decide whether that storage should be reused,
reallocated, or freed.

### What the compiler is trying to do

Reussir is aimed at RC-based functional compilation with strong low-level
optimization, especially for programs that repeatedly destruct and rebuild
algebraic data. The current pipeline is centered on:

- ownership insertion in the frontend,
- `rc.inc` / `rc.dec` cancellation and drop expansion,
- one-shot token reuse across structured control flow,
- RC creation sinking and fusion,
- invariant-group propagation for semantically immutable loads,
- TRMC-like rewrites and other lowering-oriented cleanups.

This is why the repository contains both language frontend code and substantial
backend machinery in MLIR and LLVM.

### Regions are an extension, not the whole story

Reussir does support region-local mutable objects, but regions are not the sole
or primary identity of the project. Regional mutable objects extend the same
RC-oriented runtime and compiler model: values can be built inside a local
region, mutated there, then frozen and converted into regular RC-managed
objects when they escape.

### Interoperability matters

Another explicit goal is to make RC-managed functional compilation compose well
with native code and the broader MLIR ecosystem. The current codebase therefore
includes:

- a Rust runtime that matches the RC object model,
- bridge code between the Rust frontend and MLIR/C++ backend,
- a polymorphic FFI direction instead of forcing everything through one boxed
  representation.

## Repository Layout

The repository is split by responsibility:

- `crates/`:
  the Rust frontend and runtime — `reussir-syntax` (parser), `reussir-core`
  (elaboration, monomorphization, ownership analysis), `reussir-codegen`
  (MLIR lowering), `reussir-backend`/`reussir-jit` (the melior/MLIR bridge and
  JIT), `reussir-compiler` (the `rrc` driver), `reussir-repl` (the `rrepl`
  REPL), `reussir-lsp` (semantic highlighting), `rene` (the package manager
  and build driver), and `reussir-rt` (the RC-object runtime).
- `include/` and `lib/`:
  the Reussir MLIR dialect, analyses, conversions, bridge code, and backend
  support libraries.
- `tool/`:
  command-line MLIR/LLVM tools — `reussir-opt`, `reussir-translate`, and
  `reussir-llvm-opt` — plus the LLDB pretty-printers under `tool/lldb/`.
- `tests/`:
  C++ unit tests and LLVM `lit` integration tests covering both backend passes
  and frontend end-to-end compilation.

## Compilation Guide

### Prerequisites

You need a working development environment with:

- CMake 3.28 or newer,
- Ninja,
- LLVM and MLIR with CMake package files available,
- Rust and Cargo (the pinned toolchain is fetched automatically),
- Python 3 for `lit`.

The exact package source depends on your platform and local toolchain setup.

### 1. Configure the build

```bash
cmake -S . -B build -G Ninja -DREUSSIR_ENABLE_TESTS=ON
```

Useful options:

```bash
# Disable tests
cmake -S . -B build -G Ninja -DREUSSIR_ENABLE_TESTS=OFF

# Enable stricter C++ warnings
cmake -S . -B build -G Ninja -DREUSSIR_ENABLE_TESTS=ON -DREUSSIR_ENABLE_PEDANTIC=ON
```

### 2. Compile everything

```bash
cmake --build build
```

This builds the backend libraries and MLIR tools (`reussir-opt`,
`reussir-translate`), the Rust runtime, and — through explicit CMake targets —
the Rust frontend tools:

- `reussir-opt`
- `reussir-translate`
- `rrc` — the compiler driver
- `reussir-syntax` — the parser (JSON AST emitter)
- `reussir-lsp` — the stdio language server (semantic tokens)
- `rrepl` — the REPL
- `rene` — the package manager and build driver

### 3. Compile specific targets

```bash
cmake --build build --target reussir-opt
cmake --build build --target reussir-translate
cmake --build build --target reussir-rt
cmake --build build --target rrc
cmake --build build --target rrepl
cmake --build build --target rene
cmake --build build --target reussir-lsp
cmake --build build --target reussir-vscode-package
```

Built binaries are placed under `build/bin/`, and runtime libraries are
copied under `build/lib/`.

### 4. Typical local workflows

Inspect elaboration output:

```bash
build/bin/rrc --emit hir path/to/program.rr
```

Compile a Reussir source file:

```bash
build/bin/rrc path/to/program.rr -o program.o
```

Run the REPL:

```bash
cmake --build build --target rrepl
build/bin/rrepl
```

Configure an editor to launch `build/bin/reussir-lsp` as a stdio language
server for Reussir files. The initial server intentionally provides only
whole-document semantic tokens. Reussir syntax is classified from its lossless
cstree; inline MLIR transform schedules and Rust poly-FFI bodies receive
syntax-only Tree-sitter highlighting.

The VS Code extension keeps `reussir-lsp` native while implementing its
bounded client state and LSP wire framing in Rust/WebAssembly. Build a
sideloadable extension with:

```bash
cmake --build build --target reussir-vscode-package
code --install-extension build/reussir-vscode.vsix
```

The extension first uses the `reussir.server.path` setting, then the repository
development build, and finally `reussir-lsp` from `PATH`. Open
`editors/vscode/` in VS Code and press F5 for an Extension Development Host.

Build a package with `rene` (the manifest is `rene.ncl`; artifacts land in
`reussir-build/<profile>/` next to it):

```bash
cd path/to/package
rene build                        # every declared target, dev profile
rene build --profile release      # a built-in or manifest-declared profile
rene build --bin app --lib util   # only the named targets
rene build --target wasm32-wasip1 # cross-compile for a machine target
rene build -j 4                   # cap the compile-process pool
rene inspect --solved --graph     # dependency resolution and graph as JSON
rene clean                        # delete the build directory
```

The example packages under `tests/integration/rene/` (e.g. `calc-project`,
`inventory-project`) show what a manifest looks like.

## Testing

Reussir has both unit tests and integration tests.

### C++ unit tests

```bash
cmake --build build --target reussir-ut
ctest --test-dir build --output-on-failure
```

### Rust crate tests

```bash
cmake --build build --target rrc-test
cmake --build build --target reussir-codegen-test
cmake --build build --target reussir-backend-test
cmake --build build --target reussir-core-test
cmake --build build --target reussir-syntax-test
cmake --build build --target reussir-lsp-test
cmake --build build --target reussir-vscode-test
cmake --build build --target reussir-jit-test
cmake --build build --target rrepl-test
cmake --build build --target rene-test
```

### LLVM `lit` integration tests

```bash
cmake --build build --target check
```

The integration suite covers backend conversions, reuse-related passes, and
frontend end-to-end examples under `tests/integration/`. Build `rrepl` first
(`cmake --build build --target rrepl`) to include the `repl-rs` suite.

Some suites need tools or artifacts the build does not require. Each is a lit
feature, and a test that names a missing one reports `UNSUPPORTED` instead of
failing: `lldb`/`gdb` (the debug-info suite), `openmp` (the multithreaded e2e
drivers), `lto-link`/`rustc-lto-link` (the cross-language LTO links),
`asan`/`lsan`/`msan`/`tsan` (the sanitizer suite — build the instrumented
runtimes with `cmake --build build --target reussir-rt-sanitizers`, then run
`cmake --build build --target check-sanitizer`), and `wasmer` plus
`wasip1`/`wasip1-threads` (the WebAssembly suites, which cross-build the
example packages for WASI and run the modules). To enable the last ones:

```bash
rustup target add wasm32-wasip1 wasm32-wasip1-threads   # their standard libraries
curl https://get.wasmer.io -sSfL | sh     # the engine, or your package manager
# ... then re-run cmake so it finds wasmer (or point at one explicitly):
cmake -S . -B build -G Ninja -DREUSSIR_WASMER=/path/to/wasmer
```

## Status

Reussir is an active research compiler. The design space is still moving, but
the repository already contains an end-to-end implementation of the main
compiler story described above: ownership-aware elaboration, an RC/region-aware
MLIR dialect, token-based reuse analysis, lowering passes, runtime support, and
test infrastructure.

## License

The project is dual-licensed:

- Apache License, Version 2.0
- MIT License

You may use the project under either license. The full license text is in
[LICENSE](LICENSE).
