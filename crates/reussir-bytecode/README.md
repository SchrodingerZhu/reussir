# reussir-bytecode

Direct **MLIR bytecode** emission for the Reussir compiler.

This crate builds an MLIR operation tree in a bump arena and serializes it
straight to the MLIR bytecode container (`.mlirbc`) — the binary encoding —
**without linking against MLIR or LLVM**. It is the Rust successor to the
Haskell `reussir-codegen`, which emitted MLIR *text*.

## Why emit bytecode directly?

Two obvious alternatives both couple the frontend to a specific MLIR build:

* Emit textual MLIR and shell out to `mlir-opt` — couples to a toolchain binary
  and is slow.
* Bind MLIR's C/C++ API (e.g. `melior`/`mlir-sys`) — pins an exact library
  version and ABI.

Writing the bytecode container ourselves avoids both. The output loads in any
`mlir-opt` (or the Reussir-aware `reussir-opt`) of a compatible bytecode
version, which is exactly how the lit tests check it: bytecode in, textual MLIR
out, `FileCheck` the result.

## How it stays small and version-robust

Reimplementing MLIR's binary format in full would mean reimplementing every
dialect's attribute/type codec and every operation's property serialization.
Two format features make that unnecessary:

1. **Target bytecode version 4.** Operation *properties* (inherent attributes
   like `func.func`'s `sym_name` or `arith.constant`'s `value`) were given a
   dedicated binary encoding only in version 5. At version 4 the writer stores
   each operation's *entire* attribute dictionary — inherent merged with
   discardable — and the reader reconstructs properties from it. So the writer
   never serializes properties itself.

2. **Textual fallback for every attribute and type.** The format lets each
   attribute/type entry be stored either with a dialect-specific binary encoding
   or as its plain MLIR text, re-parsed on load. This crate always uses the
   textual form, so it needs no per-dialect codec — a custom `!reussir.rc<...>`
   type is just a string.

Only the operation/region/block skeleton — names, operands, results, successors,
nesting, and the SSA value numbering — is written as true binary structure.

## Coverage

The [`dialects`] constructors cover the operation surface of the Haskell
`reussir-codegen`: the `func`/`scf`/`arith`/`math` operations a frontend leans
on, and the Reussir dialect's reference-counting (`rc.*`), reference (`ref.*`),
record (`record.compound`/`variant`/`extract`/`tag`/`dispatch`), nullable,
region, closure, and string operations, plus `panic`, `scf.index_switch`, and
the FFI ops (`trampoline`, `polyffi`). Record types — including mutually
recursive ones — are modeled in [`records`]. The generic [`Context::op`] builder
covers anything not yet wrapped. Every demo below is verified by `reussir-opt`,
which runs the full MLIR verifier on load.

## Module layout

| Module       | Responsibility                                                |
|--------------|---------------------------------------------------------------|
| `context`    | Bump arena, interning, and the type/attribute model + printers |
| `ir`         | SSA values, operations, blocks, regions, and the op builder    |
| `records`    | (Mutually) recursive `compound`/`variant` record types         |
| `builder`    | Ergonomic module/function builders                             |
| `dialects`   | High-level constructors for `arith`/`math`/`func`/`scf`/`reussir` ops |
| `numbering`  | Assigns the indices the binary format references               |
| `encoder`    | Primitive byte encodings (`PrefixVarInt`, sections, strings)   |
| `writer`     | Assembles the final container                                  |
| `spec`       | Documentation-only: the formal bytecode container grammar      |

`dialects` is split into one submodule per dialect group (`arith`, `math`,
`func`, `scf`, `rc`, `reference`, `record`, `nullable`, `region`, `closure`,
`string`, `ffi`). The example modules used by the lit tests live with the
`reussir-bytecode-demo` binary under `src/bin/`, not in the library.

## Usage

```rust
use reussir_bytecode::{builder::ModuleBuilder, context::Context, writer::write_module};
use stumpalo::Arena;

let arena = Arena::new();
let ctx = Context::new(&arena);
let i32 = ctx.int(32);

let mut module = ModuleBuilder::new(&ctx, "demo");
module.function("add", &[i32, i32], &[i32], |f| {
    let (op, sum) = f.ctx().arith_binary("arith.addi", f.arg(0), f.arg(1), i32);
    f.push(op);
    f.ret(&[sum]);
});
std::fs::write("demo.mlirbc", write_module(module.finish())).unwrap();
```

The bundled `reussir-bytecode-demo` tool emits the example modules used by the
test suite:

```console
$ reussir-bytecode-demo --list-demos
$ reussir-bytecode-demo --demo basic -o out.mlirbc
$ reussir-opt out.mlirbc        # renders the textual MLIR back
```

## Tests

* `cargo test -p reussir-bytecode` — encoder unit tests and doctests.
* `tests/integration/rust-bytecode/` — lit tests that emit each demo as bytecode
  and check that `reussir-opt` decodes it to the expected module.
