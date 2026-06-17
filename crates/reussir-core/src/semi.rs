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
pub mod pattern;
pub mod ty_eval;

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
            .find(|f| f.name == name)
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
}
