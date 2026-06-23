//! Semi elaboration: bidirectional type-checking of the surface AST into a
//! typed, still-polymorphic HIR.
//!
//! "Semi" is the intermediate form between the surface syntax and the
//! monomorphized "Full" form. A Semi program keeps generics open
//! ([`ty::TyKind::Generic`]) and may still carry inference holes that the later
//! flow/monomorphization phase resolves; crucially, the *modality* (capability)
//! of a generic is not yet known here, so nothing is mangled. What Semi *does*
//! fix is the bidirectional types of every expression, the resolved call
//! targets with their (possibly hole) type arguments, the capability "coloring"
//! of concrete regional types, and the compiled pattern-match decision trees.
//!
//! This module owns the whole Semi-phase representation: [`ty`] (the interned
//! Semi type), [`infer`] (its unification solver), and [`traits`] (the trait
//! system) all sit here, alongside the elaboration passes. The monomorphized
//! `full::*` representation is a separate, future thing.

// The Semi-phase type machinery.
pub mod infer;
pub mod traits;
pub mod ty;

// The elaboration passes.
pub mod check;
pub mod ctxt;
pub mod fulfill;
pub mod hir;
pub mod hir_build;
pub mod hir_print;
pub mod hir_raw;
pub mod pattern;
pub mod resolve;
pub mod ty_eval;

// The textual HIR parser, generated from `semi/hir_ir.lalrpop` at build time.
lalrpop_util::lalrpop_mod!(
    #[allow(clippy::all, dead_code, unused_imports)]
    pub hir_ir,
    "/semi/hir_ir.rs"
);

pub use ctxt::{DefaultCap, Elaborator, Report, Severity};

use reussir_syntax::kind::{Resolver, TokenKey};

use crate::surface;
use ty::TyCtxt;

/// Elaborate a whole surface program. Returns the elaborator holding the
/// collected, type-checked items and any diagnostics. `resolver` turns the
/// surface AST's interned token keys back into source text.
pub fn elaborate<'a, 'tcx>(
    tcx: &'a TyCtxt<'tcx>,
    program: &surface::Program,
    resolver: &'a dyn Resolver<TokenKey>,
) -> Elaborator<'a, 'tcx> {
    let mut elab = Elaborator::new(tcx, resolver);
    elab.run(program);
    elab
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::ty::{Capability, IntTy};
    use crate::with_tcx;

    /// Parse + lower + elaborate, asserting there are no errors, then run `f`
    /// on the resulting elaborator.
    fn check(source: &str, f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>, &TyCtxt<'tcx>)) {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            f(&elab, tcx);
        });
    }

    fn function<'a, 'tcx>(elab: &'a Elaborator<'_, 'tcx>, name: &str) -> &'a hir::Function<'tcx> {
        elab.elaborated
            .iter()
            .find(|f| elab.sym(f.name) == name)
            .expect("function not found")
    }

    #[test]
    fn checks_arithmetic_over_params() {
        check("fn add(x: i32, y: i32) -> i32 { x + y }", |elab, tcx| {
            let f = function(elab, "add");
            assert_eq!(f.params.len(), 2);
            let body = f.body.as_ref().unwrap();
            assert_eq!(body.ty, tcx.mk_int(IntTy::Signed(32)));
        });
    }

    #[test]
    fn infers_numeric_literals_from_return_type() {
        check("fn two() -> i32 { 1 + 1 }", |elab, tcx| {
            let body = function(elab, "two").body.as_ref().unwrap();
            // The literal holes were solved to i32 via the return type.
            assert_eq!(body.ty, tcx.mk_int(IntTy::Signed(32)));
        });
    }

    #[test]
    fn instantiates_a_generic_call() {
        check(
            "fn id<T>(x: T) -> T { x }\nfn use_id() -> i32 { id<i32>(1) }",
            |elab, tcx| {
                let body = function(elab, "use_id").body.as_ref().unwrap();
                assert_eq!(body.ty, tcx.mk_int(IntTy::Signed(32)));
            },
        );
    }

    #[test]
    fn elaborates_enum_and_match() {
        check(
            "enum List<T> { Nil, Cons(T, List<T>) }\n\
             fn first_or<T>(xs: List<T>, d: T) -> T {\n\
                 match xs {\n\
                     List::Nil => d,\n\
                     List::Cons(x, rest) => x\n\
                 }\n\
             }",
            |elab, _tcx| {
                let f = function(elab, "first_or");
                let body = f.body.as_ref().unwrap();
                // The body is a sequence wrapping a match expression.
                assert!(matches!(&body.kind, hir::ExprKind::Seq(_)));
            },
        );
    }

    #[test]
    fn colors_regional_struct_fields() {
        check(
            "struct [regional] Cell<T> { v: T, next: [field] Cell<T> }\n\
             regional fn make<T>(x: T) -> [flex] Cell<T> { Cell { v: x, next: Nullable::Null } }",
            |elab, _tcx| {
                let f = function(elab, "make");
                // The return type was colored Flex by the `[flex]` annotation.
                let crate::semi::ty::TyKind::Record { flex, .. } = f.return_ty.kind() else {
                    panic!("expected a record return type, got {:?}", f.return_ty);
                };
                assert_eq!(*flex, Capability::Flex);
            },
        );
    }

    #[test]
    fn fresh_regional_construction_is_flex_and_assignable() {
        // A regional record constructed inside a region is born `Flex`, so it can
        // be assigned into directly — no `[flex]` binding annotation required.
        // Before the fix the construction was `Regional`, and the assignment was
        // rejected with "assignment target must be a flex record".
        check(
            "struct [regional] Cell<T> { v: T, next: [field] Cell<T> }\n\
             regional fn build(seed: i32) -> i32 {\n\
                 let c = Cell { v: seed, next: Nullable::Null };\n\
                 c->next := Nullable::NonNull{c};\n\
                 0\n\
             }",
            |elab, _tcx| {
                let f = function(elab, "build");
                let body = f.body.as_ref().expect("build has a body");
                let crate::semi::hir::ExprKind::Seq(stmts) = &body.kind else {
                    panic!("expected a Seq body, got {:?}", body.kind);
                };
                let crate::semi::hir::ExprKind::Let { value, .. } = &stmts[0].kind else {
                    panic!("expected a `let` binding first, got {:?}", stmts[0].kind);
                };
                assert_eq!(
                    value.ty.capability(),
                    Some(Capability::Flex),
                    "a fresh regional construction should be flex"
                );
            },
        );
    }

    #[test]
    fn value_record_construction_is_not_flex() {
        // The fix is scoped to regional records: a value/shared record is never
        // flex (it carries no regional capability).
        check(
            "struct Pair { a: i32 }\n\
             fn build(n: i32) -> i32 { let p = Pair { a: n }; 0 }",
            |elab, _tcx| {
                let f = function(elab, "build");
                let body = f.body.as_ref().expect("build has a body");
                let crate::semi::hir::ExprKind::Seq(stmts) = &body.kind else {
                    panic!("expected a Seq body, got {:?}", body.kind);
                };
                let crate::semi::hir::ExprKind::Let { value, .. } = &stmts[0].kind else {
                    panic!("expected a `let` binding first, got {:?}", stmts[0].kind);
                };
                assert_eq!(value.ty.capability(), Some(Capability::Irrelevant));
            },
        );
    }

    #[test]
    fn reports_type_mismatch() {
        with_tcx(|tcx| {
            let source = "fn bad() -> bool { 1 }";
            let parse = reussir_syntax::parse(source);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(elab.has_errors(), "expected a type mismatch error");
        });
    }

    #[test]
    fn rejects_regional_call_outside_region() {
        with_tcx(|tcx| {
            let source = "regional fn r() -> i32 { 0 }\nfn caller() -> i32 { r() }";
            let parse = reussir_syntax::parse(source);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                elab.reports.iter().any(|r| r.message.contains("regional")),
                "expected a regional-call error: {:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn rejects_field_link_to_non_regional_record() {
        with_tcx(|tcx| {
            // A `[field]` link's element must be a regional record; a concrete
            // value record is rejected right at the declaration.
            let source = "struct Pair { a: i32 }\n\
                          struct [regional] Holder { item: [field] Pair }";
            let parse = reussir_syntax::parse(source);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("`[field]` link element")),
                "expected a `[field]` element error: {:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn rejects_capturing_a_flex_value() {
        with_tcx(|tcx| {
            // `c` is a flex regional value; a closure cannot capture it (a flex
            // value cannot be materialized, so it cannot escape its region).
            let source = "struct [regional] Cell<T> { v: T, next: [field] Cell<T> }\n\
                          regional fn f(c: [flex] Cell<i32>) -> i32 { let g = || c; 0 }";
            let parse = reussir_syntax::parse(source);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("flex value cannot escape")),
                "expected a flex-capture error: {:#?}",
                elab.reports
            );
        });
    }

    /// Elaborate a source string and return its diagnostics (for negative tests).
    fn reports_of(source: &str) -> Vec<crate::semi::Report> {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            elaborate(tcx, &prog, parse.resolver()).reports.clone()
        })
    }

    fn has_error(source: &str, needle: &str) -> bool {
        reports_of(source)
            .iter()
            .any(|r| r.message.contains(needle))
    }

    // ----- nullable-pointer-inner -----

    #[test]
    fn rejects_nullable_of_scalar() {
        let src = "fn f(x: Nullable<i32>) -> i32 { 0 }";
        assert!(
            has_error(src, "not a pointer-like"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_nullable_of_nullable() {
        let src = "struct Pair { a: i32 }\nfn f(x: Nullable<Nullable<Pair>>) -> i32 { 0 }";
        assert!(
            has_error(src, "not a pointer-like"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn accepts_nullable_of_record() {
        // A record inner is the intended case and must not be rejected.
        let src = "struct Pair { a: i32 }\nfn f(x: Nullable<Pair>) -> i32 { 0 }";
        assert!(
            !has_error(src, "not a pointer-like"),
            "{:#?}",
            reports_of(src)
        );
    }

    // ----- region-flex-checks -----

    #[test]
    fn rejects_regional_construction_outside_region() {
        let src = "struct [regional] C { v: i32, next: [field] C }\n\
                   fn f() -> i32 { C { v: 1, next: Nullable::Null }; 0 }";
        assert!(
            has_error(
                src,
                "cannot construct a regional record outside of a region"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_closure_returning_flex() {
        // The closure body is a captured flex value; returning it would let a
        // non-materializable value escape the region.
        let src = "struct [regional] Cell<T> { v: T, next: [field] Cell<T> }\n\
                   regional fn f(c: [flex] Cell<i32>) -> i32 { let g = || c; 0 }";
        assert!(
            has_error(src, "closure cannot return a flex value"),
            "{:#?}",
            reports_of(src)
        );
    }

    // ----- cast-legality-and-pin -----

    #[test]
    fn rejects_non_numeric_cast() {
        // `bool` does not satisfy `Num`, so casting it is rejected rather than
        // crashing at lowering.
        let src = "fn f(b: bool) -> i32 { b as i32 }";
        assert!(
            has_error(src, "does not implement"),
            "{:#?}",
            reports_of(src)
        );
    }

    // ----- closure-arity-and-convention -----

    #[test]
    fn rejects_closure_over_application() {
        // Immediately-invoked closure: a non-path callee is the only form the
        // parser emits as a closure application (a bare `g(..)` is a func call).
        let src = "fn f() -> i32 { (|x: i32, y: i32| x)(1, 2, 3) }";
        assert!(has_error(src, "takes at most 2"), "{:#?}", reports_of(src));
    }

    #[test]
    fn partial_application_typechecks() {
        // The inner application supplies one of two arguments, yielding a residual
        // closure; the outer application completes it.
        let src = "fn f() -> i32 { ((|x: i32, y: i32| x)(1))(2) }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    // ----- collect-trampoline-roots -----

    #[test]
    fn collects_trampoline_root() {
        check(
            "fn target<T>(x: T) -> T { x }\n\
             extern \"C\" trampoline \"t_ffi\" = target<i32>;",
            |elab, tcx| {
                assert_eq!(elab.trampolines.len(), 1);
                let root = &elab.trampolines[0];
                assert_eq!(root.name, "t_ffi");
                assert_eq!(root.abi, "C");
                assert_eq!(root.ty_args, vec![tcx.mk_int(IntTy::Signed(32))]);
            },
        );
    }

    #[test]
    fn trampoline_arity_mismatch_errors() {
        let src = "fn target<T>(x: T) -> T { x }\n\
                   extern \"C\" trampoline \"t_ffi\" = target;";
        assert!(
            has_error(src, "type argument count mismatch"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn trampoline_unknown_target_errors() {
        let src = "extern \"C\" trampoline \"t_ffi\" = nope<i32>;";
        assert!(has_error(src, "not found"), "{:#?}", reports_of(src));
    }
}
