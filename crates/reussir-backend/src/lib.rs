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
use melior::dialect::DialectHandle;

use reussir_backend_sys as sys;

/// Returns the [`DialectHandle`] for the Reussir dialect.
///
/// The handle can be inserted into a [`melior::dialect::DialectRegistry`] or
/// registered directly into a [`Context`] via [`register_reussir_dialect`].
pub fn reussir_dialect_handle() -> DialectHandle {
    // SAFETY: the handle returned by the C API points at a static registration
    // hook table that lives for the duration of the program.
    unsafe { DialectHandle::from_raw(sys::mlirGetDialectHandle__reussir__()) }
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
