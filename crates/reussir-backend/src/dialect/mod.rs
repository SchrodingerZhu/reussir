//! High-level bindings for the Reussir dialect.
//!
//! This module mirrors melior's own dialect modules (`arith`, `func`, `llvm`,
//! ...): typed builders for the dialect's operations plus constructors for its
//! types.
//!
//! * The operation builders are generated from `ReussirOps.td` by melior's
//!   [`dialect!`](melior::dialect) macro, so they track the ODS definitions
//!   automatically. They are re-exported here directly, so an op is reachable as
//!   `reussir_backend::dialect::<op>` (e.g. [`rc_create`]).
//! * The [`ty`] submodule constructs the dialect's types (`!reussir.rc`,
//!   `!reussir.ref`, ...) through the Reussir C API, since they use custom
//!   printers/parsers and cannot be built with the generic MLIR type APIs.
//!
//! The macro locates the dialect's TableGen sources through the
//! `REUSSIR_INCLUDE_DIR` environment variable, which the crate's build script
//! sets (see `build.rs`).

pub mod dbg;
pub mod ty;

// melior's `dialect!` macro emits a `pub mod reussir { ... }` containing the
// generated op builders. Wrapping it keeps the inner module name an
// implementation detail; the builders are re-exported below.
#[allow(non_snake_case, clippy::all)]
mod generated {
    melior::dialect! {
        name: "reussir",
        files: ["Reussir/IR/ReussirOps.td"],
        include_directory_env_vars: ["REUSSIR_INCLUDE_DIR"],
    }
}

pub use generated::reussir::*;

#[cfg(test)]
mod tests {
    use super::ty;
    use melior::ir::ValueLike;
    use melior::ir::operation::OperationLike;
    use melior::ir::{Location, Type};

    #[test]
    fn type_constructors_round_trip_and_are_well_formed() {
        let context = crate::context();
        let i32_ty = Type::parse(&context, "i32").unwrap();

        let cases = [
            (
                ty::rc(
                    i32_ty,
                    ty::ReussirCapability::Shared,
                    ty::ReussirAtomicKind::Normal,
                ),
                "!reussir.rc<",
            ),
            (
                ty::r#ref(
                    i32_ty,
                    ty::ReussirCapability::Value,
                    ty::ReussirAtomicKind::Normal,
                ),
                "!reussir.ref<",
            ),
            (ty::token(&context, 8, 16), "!reussir.token<"),
            (
                ty::nullable(ty::token(&context, 8, 16)),
                "!reussir.nullable<",
            ),
            (ty::region(&context), "!reussir.region"),
            (ty::raw_ptr(i32_ty), "!reussir.raw_ptr<"),
            (ty::cell(i32_ty, false), "!reussir.cell<"),
            (ty::cell(i32_ty, true), "!reussir.cell<i32 exclusive>"),
            (ty::atomic_cell(i32_ty), "!reussir.cell<i32 atomic>"),
            (ty::array(&[2, 3], i32_ty), "!reussir.array<"),
            (
                ty::str(&context, ty::ReussirLifeScope::Global),
                "!reussir.str<",
            ),
        ];

        for (built, prefix) in cases {
            // Printing a type walks its storage, so a well-formed prefix proves
            // the C API constructor produced a real, correctly tagged type.
            let printed = built.to_string();
            assert!(
                printed.starts_with(prefix),
                "expected {printed} to start with {prefix}"
            );
        }
    }

    #[test]
    fn generated_op_builder_constructs_reussir_op() {
        let context = crate::context();
        let location = Location::unknown(&context);

        let token_type = ty::token(&context, 8, 16);
        let operation: melior::ir::Operation =
            super::token_alloc(&context, token_type, location).into();

        assert!(operation.verify());
        assert_eq!(operation.result(0).unwrap().r#type(), token_type);
        assert!(operation.to_string().contains("reussir.token.alloc"));
    }
}
