//! High-level operation constructors for the dialects a Reussir frontend emits.
//!
//! These are thin, documented wrappers over [`Context::op`](crate::context::Context::op)
//! that encode the *structural* signature of each operation — its name, operand
//! order, result types, and inherent attributes — which is all the bytecode
//! format records. (The custom textual assembly syntax an operation may have is
//! irrelevant to bytecode and is reconstructed by the reader from this
//! structure.)
//!
//! Each constructor returns the built [`Op`](crate::ir::Op) together with any
//! result values, so a frontend can thread results into later operations.
//! Operations are appended to a block by pushing them through the builders in
//! [`crate::builder`].
//!
//! The constructors are inherent methods on [`Context`](crate::context::Context),
//! split across one submodule per dialect group:
//!
//! | Submodule    | Operations                                              |
//! |--------------|---------------------------------------------------------|
//! | [`arith`]    | `arith.*` arithmetic, comparison, conversion, constants |
//! | [`math`]     | `math.*` unary/binary math                              |
//! | [`func`]     | `func.call`                                             |
//! | [`scf`]      | `scf.if`, `scf.yield`, `scf.index_switch`              |
//! | [`rc`]       | `reussir.rc.*` reference counting                       |
//! | [`reference`](mod@reference)| `reussir.ref.*` references                 |
//! | [`record`]   | `reussir.record.*` records and dispatch                 |
//! | [`nullable`] | `reussir.nullable.*` nullable handling                  |
//! | [`region`]   | `reussir.region.*` region scopes                        |
//! | [`closure`]  | `reussir.closure.*` closures                            |
//! | [`string`]   | `reussir.str.*` strings                                 |
//! | [`ffi`]      | `reussir.panic`, `reussir.trampoline`, `reussir.polyffi` |
//!
//! Coverage grows with the frontend migration; the generic
//! [`Context::op`](crate::context::Context::op) builder remains available for any
//! operation not yet wrapped here.

pub mod arith;
pub mod closure;
pub mod ffi;
pub mod func;
pub mod math;
pub mod nullable;
pub mod rc;
pub mod record;
pub mod reference;
pub mod region;
pub mod scf;
pub mod string;

/// Integer comparison predicates for
/// [`Context::arith_cmpi`](crate::context::Context::arith_cmpi), matching MLIR's
/// `arith::CmpIPredicate` enumeration values.
#[derive(Clone, Copy)]
pub enum CmpIPredicate {
    Eq = 0,
    Ne = 1,
    Slt = 2,
    Sle = 3,
    Sgt = 4,
    Sge = 5,
    Ult = 6,
    Ule = 7,
    Ugt = 8,
    Uge = 9,
}
