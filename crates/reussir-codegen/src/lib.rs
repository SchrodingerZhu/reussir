//! Code generation: lowering Reussir's monomorphized Full MIR to MLIR.
//!
//! This crate sits between the frontend ([`reussir_core`], which produces a
//! ground [`reussir_core::full::mir::Program`]) and the MLIR foundation
//! ([`reussir_backend`], which provides [`melior`](reussir_backend::melior), the
//! Reussir-dialect ODS builders, and the lowering [`pipeline`]). It builds MLIR
//! in memory through those ODS builders — the frontend crate itself stays free
//! of any MLIR dependency.

pub mod lower;

// Test-only helpers, shared with the integration tests (which include the same
// file directly via `#[path]`). Not part of the crate's API.
#[cfg(test)]
pub(crate) mod testing;
