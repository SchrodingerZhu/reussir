//! Direct MLIR bytecode emission for the Reussir compiler.
//!
//! This crate builds an MLIR operation tree in a bump arena and serializes it
//! straight to the MLIR **bytecode** container format — the binary `.mlirbc`
//! encoding — without linking against MLIR or LLVM. The output is loadable by
//! any `mlir-opt` (or the Reussir-aware `reussir-opt`) of a compatible bytecode
//! version, which is how the lit tests verify it: bytecode in, textual MLIR out.
//!
//! # Why bytecode, and how it stays version-robust
//!
//! Emitting textual MLIR and shelling out to `mlir-opt` couples the frontend to
//! a specific MLIR build; binding MLIR's C/C++ API couples it to a specific ABI.
//! Writing bytecode directly avoids both. The format is kept tractable by two
//! choices (see [`writer`]): target bytecode version 4, where operation
//! properties are carried inside the ordinary attribute dictionary; and encode
//! every attribute and type via the format's textual fallback, so no per-dialect
//! binary codec has to be reimplemented. Only the operation/region/block
//! skeleton is written as true binary structure.
//!
//! # Layout
//!
//! * [`context`] — the arena, interning, and the type/attribute model.
//! * [`ir`] — SSA values, operations, blocks, regions, and the op builder.
//! * [`numbering`] — assigns the indices the binary format references.
//! * [`encoder`] — the primitive byte encodings and section framing.
//! * [`writer`] — assembles the final container.
//! * [`builder`] — ergonomic helpers for common constructs (modules, functions).
//! * [`records`] — recursive record types; [`dialects`] — per-dialect op constructors.
//!
//! # Format specification
//!
//! The bytecode wire format is documented inline, next to the code that emits
//! each part, rather than in one place. The grammar uses a small EBNF (`*`
//! zero-or-more, `?` optional, `|` alternation, `(...)` grouping; lower-case
//! names are productions). It is distilled from the MLIR sources
//! (`mlir/lib/Bytecode/{Writer,Reader}`, `mlir/include/mlir/Bytecode/Encoding.h`,
//! LLVM 22) and cross-checked byte-for-byte against `mlir-opt`/`reussir-opt`. The
//! map below says where each production is specified:
//!
//! | Production / topic                       | Documented on |
//! |------------------------------------------|---------------|
//! | `varint`, `varint_flag`, `svarint`       | [`encoder::Emitter::var_int`], [`var_int_with_flag`](encoder::Emitter::var_int_with_flag), [`signed_var_int`](encoder::Emitter::signed_var_int) |
//! | `nul_string`                             | [`encoder::Emitter::str_nul`] |
//! | `file` frame (magic, version, producer)  | [`encoder`] module docs, [`encoder::MAGIC`] |
//! | `section` frame, section ids             | [`encoder::Emitter::section`], [`encoder::section`] |
//! | operation encoding mask bits             | [`encoder::op_mask`] |
//! | section write order, version gating      | [`writer`] module docs, [`writer::write_module`] |
//! | String section                           | [`encoder::StringTable`] |
//! | Dialect, AttrType, IR sections           | [`writer::write_module`] and its `build_*`/`write_*` helpers |
//! | value and block numbering                | [`numbering`] module docs |
//!
//! # Example
//!
//! ```
//! use reussir_bytecode::{builder::ModuleBuilder, context::Context, writer::write_module};
//! use stumpalo::Arena;
//!
//! let arena = Arena::new();
//! let ctx = Context::new(&arena);
//! let i32 = ctx.int(32);
//!
//! let mut module = ModuleBuilder::new(&ctx, "demo");
//! module.function("identity", &[i32], &[i32], |f| {
//!     let x = f.arg(0);
//!     f.ret(&[x]);
//! });
//! let bytes = write_module(module.finish());
//! assert_eq!(&bytes[..4], &reussir_bytecode::encoder::MAGIC);
//! ```

pub mod builder;
pub mod context;
pub mod dialects;
pub mod encoder;
pub mod ir;
pub mod numbering;
pub mod records;
pub mod writer;
