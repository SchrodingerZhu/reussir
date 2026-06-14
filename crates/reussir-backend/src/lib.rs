//! Rust backend for Reussir.
//!
//! This crate provides MLIR access through [`melior`] with the out-of-tree
//! Reussir dialect registered. It re-exports melior so downstream crates can
//! build and manipulate Reussir IR without depending on melior directly, and
//! exposes the dialect-registration entry points the Reussir C API provides.
//!
//! The compilation pipeline that lowers Reussir IR to native code is layered on
//! top of this foundation; see the crate's roadmap for the staged port from the
//! C++ backend.

pub use melior;

use melior::Context;
use melior::dialect::{DialectHandle, DialectRegistry};

use reussir_backend_sys as sys;

pub mod dialect;
pub mod jit;
pub mod pipeline;

/// Returns the [`DialectHandle`] for the Reussir dialect.
///
/// The handle can be inserted into a [`melior::dialect::DialectRegistry`] or
/// registered directly into a [`Context`] via [`register_reussir_dialect`].
pub fn reussir_dialect_handle() -> DialectHandle {
    // SAFETY: the handle returned by the C API points at a static registration
    // hook table that lives for the duration of the program.
    unsafe { DialectHandle::from_raw(sys::mlirGetDialectHandle__reussir__()) }
}

/// Creates a fresh [`Context`] ready to parse and lower Reussir IR.
///
/// The context is constructed with the Reussir dialect, its dependent dialects,
/// and all required extensions and translations already registered, so dialect
/// extensions (such as the `arith`/`func` ConvertToLLVM interfaces the lowering
/// pipeline needs) are applied as dialects load. Prefer this over building a
/// context by hand and calling [`register_all_dialects`].
pub fn context() -> Context {
    let registry = DialectRegistry::new();
    // SAFETY: `registry` is a live melior dialect registry; the C API only
    // inserts dialects, extensions and translations into it.
    unsafe { sys::reussirPopulateRegistry(registry.to_raw()) };
    // Build the context from the populated registry so dialect extensions apply
    // as dialects load. Other dialects load on demand (during parsing and
    // lowering), matching mlir-opt: eagerly loading every available dialect
    // would pull in ones (complex, gpu, ...) whose ConvertToLLVM promise the
    // pass would then check without the conversion ever needing them.
    let context = Context::new_with_registry(&registry, /* threading */ true);
    // Load the dialects used to build IR eagerly so ops and types can be
    // constructed directly through the API (e.g. [`dialect`] op builders and
    // [`dialect::ty`] type constructors): building goes through each dialect's
    // registered op/type tables, which only exist once it is loaded. Parsing
    // loads dialects on demand, but programmatic construction does not.
    //
    // Loading Reussir also pulls its dependent dialects (arith, scf, math, LLVM,
    // ub); func and cf are needed to build the enclosing functions and control
    // flow but are not Reussir dependents. Other registered dialects still load
    // lazily, matching mlir-opt — eagerly loading everything would have the
    // ConvertToLLVM pass check promised interfaces the conversion never needs.
    context.get_or_load_dialect("reussir");
    context.get_or_load_dialect("func");
    context.get_or_load_dialect("cf");
    context
}

/// Registers only the Reussir dialect into `context`.
///
/// Use [`register_all_dialects`] instead when the context also needs the
/// upstream dialects, extensions and translations that Reussir IR depends on.
pub fn register_reussir_dialect(context: &Context) {
    reussir_dialect_handle().register_dialect(context);
}

/// Registers the Reussir dialect together with every upstream dialect,
/// extension and LLVM/builtin translation it relies on into `context`, then
/// loads all available dialects.
///
/// This produces a context ready to parse and lower Reussir IR, matching the
/// dialect set the C++ backend builds its compilation context from.
pub fn register_all_dialects(context: &Context) {
    // SAFETY: `context` is a live melior context; the C API only appends to its
    // registry and loads dialects.
    unsafe { sys::reussirRegisterAllDialects(context.to_raw()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::Module;
    use melior::ir::operation::OperationLike;

    #[test]
    fn reussir_dialect_namespace_is_registered() {
        assert_eq!(
            reussir_dialect_handle().namespace().as_str().unwrap(),
            "reussir"
        );
    }

    #[test]
    fn parses_module_using_reussir_type() {
        let context = Context::new();
        register_all_dialects(&context);

        let source = r#"
            module {
              func.func @identity(%arg0: !reussir.rc<i32>) -> !reussir.rc<i32> {
                func.return %arg0 : !reussir.rc<i32>
              }
            }
        "#;

        let module = Module::parse(&context, source).expect("module should parse");
        assert!(module.as_operation().verify());
    }
}
