//! Raw FFI bindings to the Reussir C API.
//!
//! This crate exposes the C entry points provided by `libReussirCAPI` together
//! with a re-export of [`mlir_sys`], the raw MLIR C API the bindings are built
//! on. Safe wrappers live in the `reussir-backend` crate; everything here is
//! `unsafe` and mirrors the C signatures one-to-one.

#![allow(non_snake_case)]

pub use mlir_sys;

use mlir_sys::{MlirContext, MlirDialectHandle};

unsafe extern "C" {
    /// Returns the dialect handle for the Reussir dialect. The handle can be
    /// inserted into a dialect registry or registered into a context.
    pub fn mlirGetDialectHandle__reussir__() -> MlirDialectHandle;

    /// Registers the Reussir dialect together with every upstream dialect,
    /// extension and LLVM/builtin translation it relies on into `context`, then
    /// loads all available dialects.
    pub fn reussirRegisterAllDialects(context: MlirContext);
}
