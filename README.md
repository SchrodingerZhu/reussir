<p align="center">
  <img src="misc/logos/reussir_banner_1600x800.svg" alt="Reussir Logo" width="600">
</p>

<h1 align="center">Reussir</h1>

<p align="center">
  <strong>An MLIR-based compiler framework for token-based memory reuse in RC-managed functional programs</strong>
</p>

<p align="center">
  <em>Frontend in Haskell, backend in MLIR/C++, runtime in Rust</em>
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
- bridge code between the Haskell frontend and MLIR/C++ backend,
- a polymorphic FFI direction instead of forcing everything through one boxed
  representation.

## Repository Layout

The repository is split by responsibility:

- `frontend/`:
  Haskell packages for parsing, diagnostics, elaboration, code generation, the
  REPL, and the language server.
- `include/` and `lib/`:
  the Reussir MLIR dialect, analyses, conversions, bridge code, and backend
  support libraries.
- `crates/reussir-rt/`:
  the Rust runtime for RC objects, nullable helpers, and related runtime
  support.
- `tool/`:
  command-line MLIR tools such as `reussir-opt` and `reussir-translate`.
- `tests/`:
  C++ unit tests and LLVM `lit` integration tests covering both backend passes
  and frontend end-to-end compilation.
- `reussir-vscode/`:
  editor support for the language.

## Compilation Guide

### Prerequisites

You need a working development environment with:

- CMake 3.28 or newer,
- Ninja,
- LLVM and MLIR with CMake package files available,
- Rust and Cargo,
- GHC and Cabal,
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

This builds the backend libraries and tools, the Rust runtime, and the frontend
executables exposed through CMake:

- `reussir-opt`
- `reussir-translate`
- `reussir-elab`
- `reussir-compiler`
- `reussir-parser`
- `reussir-repl`
- `reussir-lsp`

### 3. Compile specific targets

```bash
cmake --build build --target reussir-opt
cmake --build build --target reussir-translate
cmake --build build --target reussir-rt
cmake --build build --target reussir-compiler
cmake --build build --target reussir-repl
```

Built binaries are placed under `build/bin/`, and runtime libraries are copied
under `build/lib/`.

### 4. Frontend-only workflow

The frontend is managed as a Cabal multi-package project:

```bash
cabal build all -j
```

Current frontend packages:

- `reussir-bridge`
- `reussir-codegen`
- `reussir-core`
- `reussir-diagnostic`
- `reussir-lsp`
- `reussir-parser`
- `reussir-repl`

### 5. Typical local workflows

Inspect elaboration output:

```bash
build/bin/reussir-elab path/to/program.rr
```

Compile a Reussir source file:

```bash
build/bin/reussir-compiler path/to/program.rr
```

Run the REPL:

```bash
build/bin/reussir-repl
```

## Testing

Reussir has both unit tests and integration tests.

### C++ unit tests

```bash
cmake --build build --target reussir-ut
ctest --test-dir build --output-on-failure
```

### LLVM `lit` integration tests

```bash
cmake --build build --target check
```

The integration suite covers backend conversions, reuse-related passes, and
frontend end-to-end examples under `tests/integration/`.

### Haskell tests

```bash
cabal test all -j
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
