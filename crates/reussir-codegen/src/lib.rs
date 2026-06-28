//! Code generation: lowering Reussir's monomorphized Full MIR to MLIR.
//!
//! This crate sits between the frontend ([`reussir_core`], which produces a
//! ground [`reussir_core::full::mir::Program`]) and the MLIR foundation
//! ([`reussir_backend`], which provides [`melior`](reussir_backend::melior), the
//! Reussir-dialect ODS builders, and the lowering [`pipeline`]). It builds MLIR
//! in memory through those ODS builders — the frontend crate itself stays free
//! of any MLIR dependency.

pub mod lower;

#[cfg(test)]
pub(crate) mod test {
    pub(crate) fn init_tracing() {
        static TRACING_INIT: std::sync::Once = std::sync::Once::new();
        TRACING_INIT.call_once(|| {
            tracing_subscriber::fmt().init();
        });
    }
}
