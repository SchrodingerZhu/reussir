//! Lowering the monomorphized Full MIR to MLIR, built in memory.
//!
//! The frontend produces a ground [`mir::Program`]; this module builds the
//! corresponding MLIR module through the `func`/`arith`/`scf` ODS builders
//! (melior) and the Reussir-dialect / hand-written builders re-exported by
//! [`reussir_backend`]. [`reussir_backend::pipeline`] then lowers the result to
//! the LLVM dialect.
//!
//! The work is split across submodules:
//! * [`ty`] — ground scalar Reussir types → MLIR types, and the numeric
//!   classification used to pick cast ops;
//! * [`expr`] — the per-function recursive tree-walk that emits the ops.
//!
//! # Scope
//!
//! Currently lowered:
//! * the **scalar / control-flow subset** — integer and floating-point values,
//!   arithmetic and comparison, `if`, `let`/sequencing, direct calls, exported
//!   trampolines;
//! * **by-value (`[value]`) records** — construction via
//!   `reussir.record.compound`, field projection via `reussir.record.extract`;
//! * **shared (`[shared]`) records** — heap-allocated and reference-counted
//!   (`reussir.rc.create`, borrow/project/load), with the ownership analysis
//!   placing the `dup`/`drop` reference-count ops (see [`expr`]).
//!
//! Regional records, enums/`match`, closures, regions, and strings are not yet
//! lowered and surface as an explicit [`LoweringError`] rather than wrong code.
//!
//! Every callee/trampoline target in the MIR is already resolved to an interned
//! [`mir::Symbol`](reussir_core::full::mir::Symbol) (mono did the mangling), so
//! lowering only looks names up in the program's symbol table — it needs no
//! `DefTable`/resolver/`Mangler`.

mod expr;
mod ty;

use std::borrow::Cow;

use reussir_backend::builders;
use reussir_backend::melior::Context;
use reussir_backend::melior::ir::{BlockLike, Location, Module};

use reussir_core::full::mir;
use reussir_core::semi::ty::TyCtxt;

use expr::Lowerer;

/// A construct the current lowering subset does not handle.
#[derive(Debug, Clone)]
pub struct LoweringError(pub Cow<'static, str>);

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lowering error: {}", self.0)
    }
}

impl std::error::Error for LoweringError {}

type Result<T> = std::result::Result<T, LoweringError>;

fn err<T>(msg: impl Into<Cow<'static, str>>) -> Result<T> {
    Err(LoweringError(msg.into()))
}

/// Build the MLIR module for a whole program in `context`.
///
/// `tcx` is the type arena the program was monomorphized in; the ownership
/// analysis that places reference-count ops is run per function against it.
pub fn lower_program<'c, 'tcx>(
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    program: &mir::Program<'tcx>,
) -> Result<Module<'c>> {
    let module = Module::new(Location::unknown(context));
    let body = module.body();
    let lowerer = Lowerer::new(context, tcx, program);
    for func in &program.functions {
        body.append_operation(lowerer.function(func)?);
    }
    for t in &program.trampolines {
        body.append_operation(builders::trampoline_export(
            context,
            &t.abi,
            program.symbol(t.export),
            program.symbol(t.target),
            Location::unknown(context),
        ));
    }
    tracing::debug!(mlir = %module.as_operation(), "lowered program to MLIR");
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reussir_backend::melior::ir::operation::OperationLike;
    use reussir_core::full::mono::monomorphize;
    use reussir_core::{in_arena, semi::elaborate, surface};

    /// Elaborate + monomorphize + lower `source`, checking the module verifies
    /// and returning its printed text for assertions.
    fn lower_source(source: &str) -> String {
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let module = lower_program(&context, tcx, &full).expect("lowering succeeds");
            assert!(
                module.as_operation().verify(),
                "module verifies:\n{}",
                module.as_operation()
            );
            module.as_operation().to_string()
        })
    }

    #[test]
    fn lowers_fibonacci_to_verifiable_mlir() {
        let src = r#"
            pub fn fibonacci(n: u64) -> u64 {
                if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
            }
            extern "C" trampoline "fibonacci_ffi" = fibonacci;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("func.func @_RC9fibonacci"), "{mlir}");
        assert!(mlir.contains("arith.cmpi ule"), "{mlir}");
        assert!(mlir.contains("scf.if"), "{mlir}");
        assert!(mlir.contains("call @_RC9fibonacci"), "{mlir}");
        assert!(mlir.contains("reussir.trampoline"), "{mlir}");
    }

    #[test]
    fn lowers_value_records_to_verifiable_mlir() {
        // A `[value]` record is a by-value aggregate: construction lowers to
        // `reussir.record.compound` and field access to `reussir.record.extract`.
        let src = r#"
            struct [value] Point { x: i64, y: i64 }
            pub fn dot(a: Point, b: Point) -> i64 { a.x * b.x + a.y * b.y }
            pub fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }
            extern "C" trampoline "dot_ffi" = dot;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.record.compound"), "{mlir}");
        assert!(mlir.contains("reussir.record.extract"), "{mlir}");
        assert!(mlir.contains("!reussir.record<"), "{mlir}");
    }

    #[test]
    fn lowers_shared_records_to_reference_counted_mlir() {
        // A `[shared]` record is heap-allocated and reference-counted. Building
        // one lowers to `record.compound` + `rc.create`; reading a field borrows
        // the box (`rc.borrow`) and navigates it (`ref.project` / `ref.load`); and
        // the ownership analysis surrounds the uses with `rc.inc` / `rc.dec`.
        //
        // `Outer` nests a shared `Inner`, so its field is stored as an `rc` link:
        // projecting it out loads the inner pointer and retains it (`rc.inc`),
        // while the consumed outer box is released (`rc.dec`).
        let src = r#"
            struct [shared] Inner { n: i64 }
            struct [shared] Outer { inner: Inner }
            fn mk_inner(n: i64) -> Inner { Inner { n: n } }
            fn mk_outer(i: Inner) -> Outer { Outer { inner: i } }
            fn inner_of(o: Outer) -> Inner { o.inner }
            fn n_of(i: Inner) -> i64 { i.n }
            pub fn build_and_read(n: i64) -> i64 { n_of(inner_of(mk_outer(mk_inner(n)))) }
            extern "C" trampoline "build_and_read_ffi" = build_and_read;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("reussir.rc.borrow"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.load"), "{mlir}");
        assert!(mlir.contains("reussir.rc.inc"), "{mlir}");
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
    }

    #[test]
    fn lowers_a_recursive_shared_record() {
        // A `[shared]` record whose field is the record itself is finite — the
        // field is an `rc` pointer, not an inline copy. Lowering its type must
        // terminate: the identified record is published on the construction stack
        // before its members, so the self-reference on the field resolves to the
        // in-progress handle instead of recursing forever.
        let src = r#"
            struct [shared] Node { next: Node }
            pub fn forward(n: Node) -> Node { n }
        "#;
        let mlir = lower_source(src);
        // The record's own field is an `rc` link back to itself.
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("Node"), "{mlir}");
    }

    #[test]
    fn drops_an_unused_shared_parameter() {
        // An owned `[shared]` parameter that the body never uses is dead on entry,
        // so the ownership analysis releases it immediately with `rc.dec`.
        let src = r#"
            struct [shared] Box { v: i64 }
            pub fn ignore(b: Box, fallback: i64) -> i64 { fallback }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
    }

    #[test]
    fn lowers_and_runs_through_the_pipeline() {
        let src = r#"
            pub fn add3(a: i32, b: i32, c: i32) -> i32 { a + b + c }
            extern "C" trampoline "add3_ffi" = add3;
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module = lower_program(&context, tcx, &full).expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the scalar module to LLVM");
        });
    }
}
