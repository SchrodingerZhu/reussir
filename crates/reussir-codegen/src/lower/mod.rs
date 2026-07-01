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
//!   placing the `dup`/`drop` reference-count ops (see [`expr`]);
//! * **regional (`[regional]`) records** — construction of region-allocated
//!   `flex` boxes (`reussir.rc.create … region`), `region-run` scopes
//!   (`reussir.region.run`/`region.yield`), calls into `regional` functions
//!   (threading the implicit region handle), and projection of their scalar /
//!   by-value / `[shared]` members (capability-aware borrow/project/load).
//!
//! Regional `[field]` links (projection and assignment), a non-`[field]` member
//! that is itself a regional record, enums/`match`, closures, and strings are not
//! yet lowered and surface as an explicit [`LoweringError`] rather than wrong code.
//!
//! Every callee/trampoline target in the MIR is already resolved to an interned
//! [`mir::Symbol`](reussir_core::full::mir::Symbol) (mono did the mangling), so
//! lowering only looks names up in the program's symbol table — it needs no
//! `DefTable`/resolver/`Mangler`.

mod debug;
mod expr;
mod ty;

use std::borrow::Cow;

use reussir_backend::builders;
use reussir_backend::melior::Context;
use reussir_backend::melior::ir::{BlockLike, Location, Module};

use reussir_core::full::mir;
use reussir_core::semi::ty::TyCtxt;
use reussir_syntax::kind::{Resolver, TokenKey};

use crate::source::SourceMap;
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
/// `source`, when given, labels each lowered op with a `FileLineColLoc` resolved
/// from the MIR's byte spans (and carries the file name into the module's debug
/// attributes); without it, ops get `unknown` locations.
///
/// `names`, when given *together with* `source`, additionally emits DWARF debug
/// info: it resolves the interned source names of functions, parameters, and
/// locals so the debug-info conversion pass can describe each variable and its
/// (precise) type. Without `names`, only line locations are emitted.
pub fn lower_program<'c, 'tcx>(
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    program: &mir::Program<'tcx>,
    source: Option<&SourceMap<'_>>,
    names: Option<&dyn Resolver<TokenKey>>,
) -> Result<Module<'c>> {
    let mut module = Module::new(Location::unknown(context));
    let lowerer = Lowerer::new(context, tcx, program, source, names);
    lowerer.set_module_debug_attrs(&mut module);
    let body = module.body();
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
            let module =
                lower_program(&context, tcx, &full, None, None).expect("lowering succeeds");
            assert!(
                module.as_operation().verify(),
                "module verifies:\n{}",
                module.as_operation()
            );
            module.as_operation().to_string()
        })
    }

    #[test]
    fn attaches_source_locations() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        let src = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n";
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let path = std::path::Path::new("add.rr");
            let map = crate::source::SourceMap::new(path, src);
            let module =
                lower_program(&context, tcx, &full, Some(&map), None).expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with locations");
            assert!(printed.contains("loc(\"add.rr\":1:"), "{printed}");
        });
    }

    #[test]
    fn emits_debug_info_for_a_function() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        // Exercises every kind of debug type emitted: scalars, a `[value]` record
        // (with named members), a `[shared]` record (boxed), and a `let` local.
        let src = r#"
            struct [value] Point { x: i64, y: f64 }
            struct [shared] Boxed { n: i64 }
            fn proj(p: Point) -> i64 { let q = p.x; q }
            fn unbox(b: Boxed) -> i64 { b.n }
            pub fn entry(a: i64, b: f64) -> i64 {
                let p = Point { x: a, y: b };
                let bx = Boxed { n: a };
                proj(p) + unbox(bx)
            }
            extern "C" trampoline "entry" = entry;
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let path = std::path::Path::new("pt.rr");
            let map = crate::source::SourceMap::new(path, src);
            let module = lower_program(&context, tcx, &full, Some(&map), Some(parse.resolver()))
                .expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with debug info");
            // Module file attributes, the subprogram + parameter array, and every
            // debug-type form including precise member names and the boxed type.
            assert!(printed.contains("reussir.dbg.file_basename"), "{printed}");
            assert!(printed.contains("dbg_subprogram"), "{printed}");
            assert!(printed.contains("dbg_func_args"), "{printed}");
            assert!(printed.contains("dbg_recordtype"), "{printed}");
            assert!(printed.contains("dbg_inttype"), "{printed}");
            assert!(printed.contains("dbg_fptype"), "{printed}");
            assert!(printed.contains("dbg_boxedtype"), "{printed}");
            assert!(printed.contains("dbg_localvar"), "{printed}");
            // Field names are preserved (not positional).
            assert!(printed.contains("name : \"x\""), "{printed}");
            assert!(printed.contains("name : \"y\""), "{printed}");
        });
    }

    #[test]
    fn emits_debug_info_for_variants() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        // A `[value]` enum (inline tagged union) and a managed (`[shared]`) enum
        // (a boxed variant). Each enum-typed parameter gets a variant debug type;
        // the shared one is additionally a boxed type. Case names ride along as
        // member names.
        let src = r#"
            enum [value] Tag { A, B(i64) }
            enum List { Nil, Cons(i64, List) }
            fn use_tag(t: Tag) -> i64 { 0 }
            fn use_list(l: List) -> i64 { 0 }
            pub fn entry(n: i64) -> i64 { use_tag(Tag::B{n}) + use_list(List::Nil) }
            extern "C" trampoline "entry" = entry;
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let path = std::path::Path::new("variant.rr");
            let map = crate::source::SourceMap::new(path, src);
            let module = lower_program(&context, tcx, &full, Some(&map), Some(parse.resolver()))
                .expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with debug info");
            // A variant debug type is emitted, and the shared variant is boxed.
            assert!(printed.contains("dbg_recordtype"), "{printed}");
            assert!(printed.contains("is_variant : true"), "{printed}");
            assert!(printed.contains("dbg_boxedtype"), "{printed}");
            // Case names ride along as the variant's member names.
            assert!(printed.contains("\"A\""), "{printed}");
            assert!(printed.contains("\"B\""), "{printed}");
            assert!(printed.contains("\"Nil\""), "{printed}");
            assert!(printed.contains("\"Cons\""), "{printed}");
        });
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
    fn lowers_a_chained_projection_through_shared_records() {
        // `o.inner.n` is a single projection with a two-element path crossing two
        // shared records: borrow the outer box, project + load the inner `rc` link,
        // borrow that, then project the scalar — so the walk emits two `rc.borrow`s.
        let src = r#"
            struct [shared] Inner { n: i64 }
            struct [shared] Outer { inner: Inner }
            pub fn deep(o: Outer) -> i64 { o.inner.n }
        "#;
        let mlir = lower_source(src);
        assert_eq!(
            mlir.matches("reussir.rc.borrow").count(),
            2,
            "expected two borrows for a two-level shared chain:\n{mlir}"
        );
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
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
    fn lowers_value_variant_construction() {
        // A `[value]` enum is an inline tagged union. Each case's fields are packed
        // into its `{enum}::{case}` payload compound with `record.compound`, which
        // `record.variant [tag]` then tags into the variant record. A fieldless
        // case builds an empty payload compound. No `rc` is involved.
        let src = r#"
            enum [value] Shape { Dot, Segment(i64, i64) }
            pub fn dot() -> Shape { Shape::Dot }
            pub fn segment(x: i64, y: i64) -> Shape { Shape::Segment{x, y} }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.record<variant"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[0]"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[1]"), "{mlir}");
        assert!(mlir.contains("reussir.record.compound"), "{mlir}");
        // No heap allocation for a by-value enum.
        assert!(!mlir.contains("reussir.rc.create"), "{mlir}");
    }

    #[test]
    fn lowers_shared_variant_construction() {
        // A managed (default `[shared]`) enum boxes the tagged value: build the
        // case payload, tag it with `record.variant`, then `rc.create` the box. The
        // recursive `Cons` field is itself an `rc` link back to the enum, so the
        // record type lowers finitely (the self-reference resolves to the
        // in-progress handle).
        let src = r#"
            enum IntList { Nil, Cons(i64, IntList) }
            pub fn one(x: i64) -> IntList { IntList::Cons{x, IntList::Nil} }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.record<variant"), "{mlir}");
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[0]"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[1]"), "{mlir}");
        assert_eq!(
            mlir.matches("reussir.rc.create").count(),
            2,
            "one box per constructed case (Nil and Cons):\n{mlir}"
        );
    }

    #[test]
    fn lowers_regional_record_construction_and_region_run() {
        // A `[regional]` record is region-allocated with a `flex` (region-local,
        // mutable) box. A `regional` function takes the region it allocates into
        // as an implicit `!reussir.region` parameter and builds into it
        // (`rc.create … region`); a `region-run` scope (`reussir.region.run`)
        // establishes that region and threads it into the regional call.
        let src = r#"
            struct [regional] Cell { v: i64 }
            regional fn make(x: i64) -> [flex] Cell { Cell { v: x } }
            pub fn run(n: i64) -> Cell { regional { make(n) } }
        "#;
        let mlir = lower_source(src);
        // The regional function carries the implicit region parameter and builds
        // a `flex` box into it.
        assert!(mlir.contains("!reussir.region"), "{mlir}");
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("flex"), "{mlir}");
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("region("), "{mlir}");
        // The value escapes the region, so `run` yields it as a `rigid` box.
        assert!(mlir.contains("rigid"), "{mlir}");
        // The region scope and its terminator.
        assert!(mlir.contains("reussir.region.run"), "{mlir}");
        assert!(mlir.contains("reussir.region.yield"), "{mlir}");
    }

    #[test]
    fn lowers_regional_record_through_the_pipeline() {
        // The regional construction subset lowers all the way to the LLVM
        // dialect: token instantiation and the region patterns pass turn
        // `region.run` into an allocation scope, attach the box vtable, and
        // freeze the flex result on the way out.
        let src = r#"
            struct [regional] Cell { v: i64 }
            regional fn make(x: i64) -> [flex] Cell { Cell { v: x } }
            pub fn run(n: i64) -> Cell { regional { make(n) } }
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
            let mut module =
                lower_program(&context, tcx, &full, None, None).expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the regional module to LLVM");
        });
    }

    #[test]
    fn lowers_regional_record_field_projection() {
        // Projecting a field out of a `flex` regional record borrows it at its
        // `flex` capability (`rc.borrow`), navigates by reference
        // (`reussir.ref.project`), and loads the scalar field (`reussir.ref.load`)
        // — the result leaves the region as a plain `i64`, so nothing escapes.
        let src = r#"
            struct [regional] Cell { v: i64 }
            regional fn make(x: i64) -> [flex] Cell { Cell { v: x } }
            pub fn run(n: i64) -> i64 { regional { make(n).v } }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.borrow"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.load"), "{mlir}");
        // The reference into the region-local record carries `flex` capability.
        assert!(mlir.contains("ref<") && mlir.contains("flex"), "{mlir}");
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
            let mut module =
                lower_program(&context, tcx, &full, None, None).expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the scalar module to LLVM");
        });
    }
}
