---
name: compilation
description: Compile Reussir source code.
license: MPL-2.0
---

You can use `rrc` (the Reussir compiler driver, built from
`crates/reussir-compiler`) to compile Reussir source code. Build it with
`cmake --build build --target rrc`; the binary lands at `build/bin/rrc`.

`rrc` is a clang-style pipeline driver. The input's extension (or `--from`)
picks where to enter the chain, and `-o`'s extension (or `--emit`) picks where
to leave it:

```text
.rr ──▶ .hir ──▶ .mir ──▶ .mlir ──▶ mlir-llvm ──▶ .ll ──▶ .s/.o
```

Key options (`rrc --help` for the full list):

```text
Usage: rrc [INPUT] --output <OUTPUT> [OPTIONS]

  -o, --output <OUTPUT>     Output file (`-` writes hir/mir/mlir text dumps to stdout)
  -t, --emit <EMIT>         Stage: hir, mir, mlir, mlir-llvm, llvm-ir, asm, obj
  -x, --from <FROM>         Treat input as: rr, hir, mir, mlir, llvm-ir
  -O, --opt <OPT>           none, default, aggressive, size
      --relocation-mode     default, pic, static, dynamic-no-pic
      --target-triple TRIPLE
      --target-cpu CPU
      --target-features FEATURES
  -v, --verbose             Log lowering/backend tracing events to stderr
```

As an example, to dump the optimized LLVM IR:

```bash
build/bin/rrc tests/integration/frontend/projection.rr -o out.ll --emit llvm-ir -O aggressive
```

The first several lines will look like this:

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%_RIC3BoxIC5RcBoxIC5RcBoxIC5RcBoxIC3BoxIC3BoxmEEEEEE = type { ptr }
```

Note: `llvm-ir`/`asm`/`obj` targets go through the file-based LLVM emitter, so
they need a real `-o` path (not `-o -`). The `hir`/`mir`/`mlir`/`mlir-llvm` text
dumps can stream to stdout with `-o -`.

To cross-compile for a specific target, use the `--target-triple` flag:

```bash
build/bin/rrc tests/integration/frontend/projection.rr -o output.ll --emit llvm-ir --target-triple x86_64-unknown-linux-gnu
```

You can also specify `--target-cpu` and `--target-features` for finer control
over code generation.

Using `-v` (or `RUST_LOG`), you can get more detailed tracing about the
lowering and backend pipeline (it can be quite verbose).
