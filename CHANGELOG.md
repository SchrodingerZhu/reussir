# Changelog

## Unreleased

### Language

- `if` without an `else` branch is now accepted as sugar for an empty
  `else {}`: the whole expression is unit and the then-branch is checked
  against unit. The elaborated HIR carries an explicit empty else-block, so
  everything downstream of elaboration is unchanged.

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
