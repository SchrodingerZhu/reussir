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
pub mod lang;
pub mod traits;
pub mod ty;

// The elaboration passes.
pub mod check;
pub mod ctxt;
pub mod externs;
pub mod fulfill;
pub mod hir;
pub mod pattern;
pub mod repl;
pub mod resolve;
pub mod ty_eval;

pub use ctxt::{
    Checkpoint, DefaultCap, Elaborator, PackageFile, Report, Severity, TransformScript,
    render_reports, render_reports_to,
};
pub use externs::ExternPackage;

use reussir_syntax::SessionInterner;

use crate::surface;
use ty::TyCtxt;

/// Elaborate a whole surface program. Returns the elaborator holding the
/// collected, type-checked items and any diagnostics. `interner` is the
/// shared session table the program was parsed with — it resolves the
/// surface AST's token keys back to text *and* interns the elaborator's own
/// names, both through `&self` (the threaded interner is interior-mutable,
/// so sharing it is never `&mut`).
pub fn elaborate<'a, 'tcx>(
    tcx: &'a TyCtxt<'tcx>,
    program: &surface::Program,
    interner: &'a SessionInterner,
) -> Elaborator<'a, 'tcx> {
    let mut elab = Elaborator::new(tcx, interner);
    elab.run(program);
    elab
}

/// Elaborate a whole package — one translation unit spanning many files, each
/// under its own module path. All items are declared before any is resolved,
/// so cross-file references work in any order. Every file must have been
/// parsed with the **one shared session interner** passed here: cross-file
/// resolution compares interned keys.
pub fn elaborate_package<'a, 'tcx>(
    tcx: &'a TyCtxt<'tcx>,
    files: &[PackageFile<'_>],
    interner: &'a SessionInterner,
) -> Elaborator<'a, 'tcx> {
    let mut elab = Elaborator::new(tcx, interner);
    elab.run_package(files);
    elab
}

/// [`elaborate_package`] against loaded dependency interfaces: every extern
/// package's items are declared first (the declare-only twin of the
/// in-package scan, see [`externs`]), so consumer references `dep::item`
/// resolve during the same run — `pub` items only. Extern paths re-intern
/// into the session table.
pub fn elaborate_package_with_externs<'a, 'tcx>(
    tcx: &'a TyCtxt<'tcx>,
    files: &[PackageFile<'_>],
    interner: &'a SessionInterner,
    externs: &[ExternPackage<'_, 'tcx>],
) -> Elaborator<'a, 'tcx> {
    let mut elab = Elaborator::new(tcx, interner);
    for ext in externs {
        // The extern declare pass is generic over `&mut impl Interner` (it
        // also serves owned tables); the session table satisfies that
        // spelling by a handle copy — an `Arc` clone, not a table copy.
        let mut handle = interner.clone();
        elab.declare_extern_package(&mut handle, ext);
    }
    elab.run_package(files);
    elab
}

#[cfg(test)]
mod tests {
    use reussir_syntax::SharedInterner;

    use super::*;
    use crate::semi::ty::{Flexivity, IntTy, TyKind};
    use crate::with_tcx;

    /// Every session gets the comparison tower: the four lang traits
    /// (method-less compiler stand-ins at the crate root), the
    /// `Num ⊴ PartialOrd` edge, and the method-less scalar impls —
    /// `PartialEq`/`PartialOrd` on every scalar directly, `Ord` on the
    /// totally ordered ones, `Eq` answered purely by super-projection.
    #[test]
    fn cmp_fallback_tower_shape() {
        use crate::semi::lang::LangItem;
        check("pub fn f(x: i64) -> i64 { x }", |elab, _| {
            let id_of = |item| {
                let def = elab.lang.get(item).expect("ensured");
                elab.traits.trait_by_def(def).expect("trait-kinded")
            };
            let (pe, eq, po, ord) = (
                id_of(LangItem::PartialEq),
                id_of(LangItem::Eq),
                id_of(LangItem::PartialOrd),
                id_of(LangItem::Ord),
            );
            for id in [pe, eq, po, ord] {
                let t = elab.traits.trait_def(id);
                assert!(!t.sealed && t.methods.is_empty() && t.compiler_provided());
            }
            let num_supers: Vec<_> = elab
                .traits
                .trait_def(elab.lang.num)
                .supertraits
                .iter()
                .map(|s| s.trait_id)
                .collect();
            assert_eq!(num_supers, [po], "Num gains exactly the PartialOrd edge");
            let count = |id| {
                elab.traits
                    .impls()
                    .filter(|i| i.trait_ref.trait_id == id)
                    .count()
            };
            assert_eq!(
                (count(pe), count(po), count(ord), count(eq)),
                (15, 15, 10, 0),
                "impl completion: every scalar for PartialEq/PartialOrd, the ordered ten for Ord, none for Eq"
            );
            assert!(
                elab.traits
                    .impls()
                    .filter(|i| [pe, po, ord].contains(&i.trait_ref.trait_id))
                    .all(|i| i.methods.is_empty() && i.compiler_provided()),
                "tower impls are method-less and compiler-provided"
            );
        });
    }

    /// Scalars and bounded generics keep the intrinsic path; the
    /// `Num ⊴ PartialOrd` edge is what lets `T: Num` code compare.
    #[test]
    fn scalar_and_generic_comparisons_ground_intrinsically() {
        check(
            "pub fn c(a: char, b: char) -> bool { a < b }
             pub fn b(a: bool, b: bool) -> bool { a == b }
             fn m<T: Ord>(a: T, b: T) -> T { if a < b { a } else { b } }
             fn e<T: Num>(a: T, b: T) -> bool { a != b }
             pub fn go(x: i64, y: i64) -> bool { e(m(x, y), y) }",
            |_, _| {},
        );
    }

    /// Without `core`, comparing anything the compiler does not order
    /// itself is a clean diagnostic, not a mislowering.
    #[test]
    fn non_scalar_comparisons_need_core() {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let src = "pub struct P { pub x: i64 }
                       pub fn s(a: str, b: str) -> bool { a == b }
                       pub fn r(a: P, b: P) -> bool { a < b }";
            let parse = reussir_syntax::parse_with_interner(src, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            let msgs: Vec<_> = elab.reports.iter().map(|r| r.message.as_str()).collect();
            assert!(
                msgs.iter()
                    .any(|m| m.contains("`==` on `str` needs `core`'s `PartialEq`")),
                "{msgs:#?}"
            );
            assert!(
                msgs.iter()
                    .any(|m| m.contains("`<` on `P` needs `core`'s `PartialOrd`")),
                "{msgs:#?}"
            );
        });
    }

    /// The method-less stand-ins cannot be implemented: an impl would let
    /// a non-scalar satisfy a comparison bound and reach the intrinsic
    /// lowering. (`core`'s method-carrying declarations are implementable.)
    #[test]
    fn fallback_operator_traits_reject_impls() {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let src = "pub struct P { pub x: i64 }
                       impl PartialEq for P { }";
            let parse = reussir_syntax::parse_with_interner(src, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports.iter().any(|r| r.message.contains(
                    "cannot implement `PartialEq`: without `core` it is a method-less compiler fallback"
                )),
                "{:#?}",
                elab.reports
            );
        });
    }

    /// A batch that declares the tower itself — `core` compiling itself —
    /// suppresses the fallback: the `#[lang]` declarations bind (with
    /// source spans), operators dispatch to their methods, `!=` negates
    /// `eq`, and `Ordering` construction and matching work.
    #[test]
    fn declared_tower_binds_and_dispatches() {
        use crate::semi::lang::LangItem;
        check(
            r#"
            #[lang("partial_eq")]
            pub trait PartialEq { fn eq(self: Self, other: Self) -> bool; }
            #[lang("eq")]
            pub trait Eq: PartialEq {}
            #[lang("partial_ord")]
            pub trait PartialOrd: PartialEq {
                fn lt(self: Self, other: Self) -> bool;
                fn le(self: Self, other: Self) -> bool;
                fn gt(self: Self, other: Self) -> bool;
                fn ge(self: Self, other: Self) -> bool;
            }
            #[lang("ordering")]
            pub enum [value] Ordering { Less, Equal, Greater }
            #[lang("ord")]
            pub trait Ord: Eq + PartialOrd { fn cmp(self: Self, other: Self) -> Ordering; }

            pub struct P { pub x: i64 }
            impl PartialEq for P { fn eq(self: Self, other: Self) -> bool { self.x == other.x } }
            pub fn eqs(a: P, b: P) -> bool { a == b }
            pub fn nes(a: P, b: P) -> bool { a != b }
            fn tw<T: Ord>(a: T, b: T) -> Ordering { a.cmp(b) }
            pub fn go(x: i64, y: i64) -> i64 {
                match tw(x, y) { Ordering::Less => 0, Ordering::Equal => 1, Ordering::Greater => 2 }
            }
            "#,
            |elab, _| {
                let def = elab.lang.get(LangItem::PartialEq).expect("declared");
                let id = elab.traits.trait_by_def(def).expect("trait-kinded");
                let t = elab.traits.trait_def(id);
                assert!(
                    t.span.is_some() && !t.compiler_provided(),
                    "declared, with a site"
                );
                assert_eq!(t.methods.len(), 1, "the declared trait keeps its method");
                assert!(
                    elab.lang_item_site(LangItem::Ordering).is_some(),
                    "the enum binds with a site too"
                );
            },
        );
    }

    /// The `Num ⊴ PartialOrd` edge and the fallback tower are checkpoint
    /// state: a rejected batch retracts both, and the next batch re-ensures
    /// them at the same dense ids.
    #[test]
    fn cmp_tower_is_checkpoint_covered() {
        use crate::semi::lang::LangItem;
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let mut elab = Elaborator::new(tcx, &interner);
            let batch = |src: &str, interner: &std::sync::Arc<_>| {
                let parse =
                    reussir_syntax::parse_with_interner(src, std::sync::Arc::clone(interner));
                assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
                surface::program(&parse.root)
            };
            let broken = batch("pub fn f() -> i64 { missing() }", &interner);
            assert!(elab.try_extend(&broken).is_err());
            assert!(
                elab.lang.get(LangItem::PartialEq).is_none()
                    && elab.traits.trait_def(elab.lang.num).supertraits.is_empty(),
                "a rejected batch retracts the tower and the edge"
            );
            let ok = batch("pub fn g(a: char, b: char) -> bool { a < b }", &interner);
            assert!(elab.try_extend(&ok).is_ok());
            assert!(elab.lang.get(LangItem::Ord).is_some());
            assert_eq!(
                elab.traits.trait_def(elab.lang.num).supertraits.len(),
                1,
                "the next batch re-ensures the edge exactly once"
            );
        });
    }

    #[test]
    fn repl_batch_gets_the_cmp_tower() {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let mut elab = Elaborator::new(tcx, &interner);
            let src = "fn m<T: Ord>(a: T, b: T) -> T { if a < b { a } else { b } }";
            let parse = reussir_syntax::parse_with_interner(src, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let r = elab.try_extend(&prog);
            assert!(r.is_ok(), "batch rejected: {r:#?}");
        });
    }

    /// Parse + lower + elaborate, asserting there are no errors, then run `f`
    /// on the resulting elaborator.
    fn check(source: &str, f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>, &TyCtxt<'tcx>)) {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            f(&elab, tcx);
        });
    }

    /// Parse a set of `(module path, source)` files with one shared interner
    /// and elaborate them as a package rooted at `pkg`.
    fn check_package(
        pkg: &str,
        files: &[(&[&str], &str)],
        f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>, &TyCtxt<'tcx>),
    ) {
        use std::sync::Arc;

        use reussir_syntax::Interner;
        use reussir_syntax::source::FileId;

        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let pkg_key = Interner::get_or_intern(&mut keys, pkg);
            let parses: Vec<_> = files
                .iter()
                .map(|(_, src)| {
                    let p = reussir_syntax::parse_with_interner(src, interner.clone());
                    assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                    p
                })
                .collect();
            let programs: Vec<surface::Program> =
                parses.iter().map(|p| surface::program(&p.root)).collect();
            let pkg_files: Vec<PackageFile> = files
                .iter()
                .zip(&programs)
                .enumerate()
                .map(|(i, ((module, _), program))| {
                    let mut path = vec![pkg_key];
                    path.extend(
                        module
                            .iter()
                            .map(|seg| Interner::get_or_intern(&mut keys, seg)),
                    );
                    PackageFile {
                        file: FileId::from_index(i as u32),
                        module: path,
                        program,
                    }
                })
                .collect();
            let elab = elaborate_package(tcx, &pkg_files, &interner);
            f(&elab, tcx);
        });
    }

    #[test]
    fn collects_transform_anchors_and_scripts() {
        check(
            "#[transform_anchor]\nfn selected() {}\nfn plain() {}\n\
             transform [{\n  transform.yield\n}];",
            |elab, _| {
                assert_eq!(elab.transform_anchors.len(), 1);
                let proto = elab
                    .functions
                    .get(&elab.transform_anchors[0])
                    .expect("anchored function");
                assert_eq!(elab.sym(proto.name), "selected");
                assert_eq!(elab.transform_scripts.len(), 1);
                assert_eq!(elab.transform_scripts[0].body, "{\n  transform.yield\n}");
            },
        );
    }

    /// The private-in-public discipline reaches through every structural
    /// former — closure types, arrays are ruled out by `Plain` elements, but
    /// `Arc`, `Cell`, `Nullable`, and record arguments all wrap a nominal —
    /// and fires once per offender per surface. A fully-`pub` surface and a
    /// private surface over private records both stay silent.
    #[test]
    fn rejects_private_records_in_public_surfaces() {
        let reports = reports_of(
            "struct Hidden { value: i64 }\n\
             pub struct Wrap<T> { inner: T }\n\
             pub fn through_arc(a: Arc<Hidden>) -> i64 { 0 }\n\
             pub fn through_args(w: Wrap<Hidden>) -> i64 { 0 }\n\
             pub fn through_closure(f: Hidden -> i64) -> i64 { 0 }\n\
             pub struct Leaky { h: Hidden }\n\
             fn private_surface(h: Hidden) -> Hidden { h }\n\
             pub fn clean(w: Wrap<i64>) -> i64 { 0 }",
        );
        let offending: Vec<&str> = reports
            .iter()
            .filter(|r| r.message.contains("private record `Hidden`"))
            .map(|r| r.message.as_str())
            .collect();
        for expected in [
            "`pub fn through_arc`",
            "`pub fn through_args`",
            "`pub fn through_closure`",
            "`pub` record `Leaky`",
        ] {
            assert!(
                offending.iter().any(|m| m.contains(expected)),
                "missing {expected:?}: {offending:#?}"
            );
        }
        assert_eq!(offending.len(), 4, "{offending:#?}");
        assert!(
            !reports
                .iter()
                .any(|r| r.message.contains("private_surface") || r.message.contains("clean")),
            "{reports:#?}"
        );
    }

    #[test]
    fn rejects_invalid_transform_anchors() {
        let reports = reports_of(
            "#[transform_anchor(argument)] fn with_arg() {}\n\
             #[transform_anchor] struct S {}\n\
             #[transform_anchor] fn declaration();\n\
             #[transform_anchor] #[transform_anchor] fn duplicate() {}",
        );
        let messages: Vec<&str> = reports
            .iter()
            .map(|report| report.message.as_str())
            .collect();
        for expected in [
            "does not accept arguments",
            "only be attached to a function",
            "requires a function body",
            "only be specified once",
        ] {
            assert!(
                messages.iter().any(|message| message.contains(expected)),
                "missing {expected:?}: {messages:#?}"
            );
        }
    }

    #[test]
    fn package_elaboration_resolves_across_modules() {
        // The reference `module_basic` shape: `lib.rr` calls `math::add`.
        check_package(
            "mylib",
            &[
                (
                    &[],
                    r#"
                        mod math;
                        pub fn entry(n: u64) -> u64 { math::add(n, 10) }
                        extern "C" trampoline "entry_ffi" = entry;
                    "#,
                ),
                (&["math"], r#"pub fn add(a: u64, b: u64) -> u64 { a + b }"#),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
                // Module paths drive the v0 mangling: nested `Nv` wrappers.
                let (full, reports) = crate::full::mono::monomorphize(&elab.mono_input());
                assert!(reports.is_empty(), "mono reports: {reports:#?}");
                let symbols: Vec<&str> = full
                    .functions
                    .iter()
                    .map(|f| full.symbol(f.symbol))
                    .collect();
                assert!(symbols.contains(&"_RNvC5mylib5entry"), "{symbols:?}");
                assert!(symbols.contains(&"_RNvNvC5mylib4math3add"), "{symbols:?}");
            },
        );
    }

    #[test]
    fn package_relative_paths_resolve_root_and_super() {
        // The reference `module_relative_paths` shape, exercising `super::`,
        // `super::super::`, `root::`, and cross-module record types.
        check_package(
            "mypkg",
            &[
                (
                    &[],
                    r#"
                        mod models;
                        mod utils;
                        pub fn entry(n: u64) -> u64 {
                            root::utils::nested::compute(n) + root::models::seed()
                        }
                    "#,
                ),
                (
                    &["models"],
                    r#"
                        pub fn seed() -> u64 { 3 }
                        pub fn scale(x: u64, k: u64) -> u64 { x * k }
                        pub struct Wrap { pub v: u64 }
                    "#,
                ),
                (
                    &["utils"],
                    r#"
                        mod math;
                        mod nested;
                        pub fn offset() -> u64 { 4 }
                    "#,
                ),
                (
                    &["utils", "math"],
                    r#"
                        pub fn scaled_offset(x: u64) -> u64 {
                            root::models::scale(super::offset(), x)
                        }
                    "#,
                ),
                (
                    &["utils", "nested"],
                    r#"
                        pub fn compute(x: u64) -> u64 {
                            let w = super::super::models::Wrap { v: x };
                            super::math::scaled_offset(w.v) + super::super::models::seed()
                        }
                    "#,
                ),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            },
        );
    }

    #[test]
    fn same_item_name_in_two_modules_stays_distinct() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod b;\npub fn go() -> u64 { a::f() + b::f() }"),
                (&["a"], "pub fn f() -> u64 { 1 }\npub struct S { v: u64 }"),
                (&["b"], "pub fn f() -> u64 { 2 }\npub struct S { v: bool }"),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
                let (full, reports) = crate::full::mono::monomorphize(&elab.mono_input());
                assert!(reports.is_empty(), "mono reports: {reports:#?}");
                let symbols: Vec<&str> = full
                    .functions
                    .iter()
                    .map(|f| full.symbol(f.symbol))
                    .collect();
                assert!(symbols.contains(&"_RNvNvC1p1a1f"), "{symbols:?}");
                assert!(symbols.contains(&"_RNvNvC1p1b1f"), "{symbols:?}");
            },
        );
    }

    #[test]
    fn unresolved_cross_module_reference_reports_the_full_path() {
        check_package(
            "p",
            &[(&[], "pub fn go() -> u64 { missing::f() }")],
            |elab, _| {
                assert!(elab.has_errors());
                let msg = &elab.reports[0].message;
                assert!(msg.contains("unknown function `missing::f`"), "{msg}");
            },
        );
    }

    #[test]
    fn math_intrinsics_type_check_per_shape() {
        check(
            r#"
                fn trig(x : f64) -> f64 {
                    core::intrinsic::math::cos(x, 1) + core::intrinsic::math::sin(x, 127)
                }
                fn narrow(x : f32) -> f32 {
                    core::intrinsic::math::sqrt<f32>(x, 0)
                }
                fn classify(x : f64) -> bool {
                    core::intrinsic::math::isnan(core::intrinsic::math::powf(x, 2.0, 0), 0)
                }
                fn ipow(x : f64) -> f64 {
                    core::intrinsic::math::fpowi(x, 3, 0)
                }
                fn fused(a : f64, b : f64, c : f64) -> f64 {
                    core::intrinsic::math::fma(a, b, c, 127)
                }
            "#,
            |elab, _| {
                use crate::intrinsic::{IntrinsicOp, MathFn};
                use crate::semi::hir::ExprKind;
                // The elaborated bodies carry `Intrinsic` nodes with the parsed
                // op + flag (spot-check `trig`).
                let trig = &elab.elaborated[0];
                let ExprKind::Arith(l, _, r) = &trig.body.as_ref().unwrap().kind else {
                    panic!("trig body is an addition");
                };
                let ExprKind::Intrinsic { op, args } = &l.kind else {
                    panic!("lhs is an intrinsic");
                };
                assert_eq!(
                    (*op, args.len()),
                    (
                        IntrinsicOp::Math {
                            func: MathFn::Cos,
                            flag: 1
                        },
                        1
                    )
                );
                let ExprKind::Intrinsic { op, .. } = &r.kind else {
                    panic!("rhs is an intrinsic");
                };
                assert_eq!(
                    *op,
                    IntrinsicOp::Math {
                        func: MathFn::Sin,
                        flag: 127
                    }
                );
            },
        );
    }

    #[test]
    fn math_intrinsics_reject_bad_shapes() {
        // A non-constant fast-math flag.
        let msg = |reports: Vec<Report>| {
            reports
                .iter()
                .map(|r| r.message.clone())
                .collect::<Vec<_>>()
                .join("\n")
        };
        let m = msg(reports_of(
            "pub fn f(x : f64) -> f64 { core::intrinsic::math::cos(x, x) }",
        ));
        assert!(m.contains("must be a constant integer"), "{m}");
        // A flag out of range.
        let m = msg(reports_of(
            "pub fn f(x : f64) -> f64 { core::intrinsic::math::cos(x, 128) }",
        ));
        assert!(m.contains("must be a constant integer"), "{m}");
        // An unknown intrinsic name, and a non-math `core` path.
        let m = msg(reports_of(
            "pub fn f(x : f64) -> f64 { core::intrinsic::math::tanhh(x, 1) }",
        ));
        assert!(
            m.contains("unknown math intrinsic `core::intrinsic::math::tanhh`"),
            "{m}"
        );
        let m = msg(reports_of(
            "pub fn f(x : f64) -> f64 { core::other::cos(x, 1) }",
        ));
        assert!(
            m.contains("the built-in `core` package has no function `core::other::cos`"),
            "{m}"
        );
        // A non-float operand fails the FloatingPoint bound.
        let m = msg(reports_of(
            "pub fn f(x : u64) -> u64 { core::intrinsic::math::cos(x, 1) }",
        ));
        assert!(m.contains("does not implement `FloatingPoint`"), "{m}");
        // Wrong arity (the trailing flag is required).
        let m = msg(reports_of(
            "pub fn f(x : f64) -> f64 { core::intrinsic::math::cos(x) }",
        ));
        assert!(m.contains("expects 2 argument(s)"), "{m}");
    }

    /// Impl members ride the ordinary defs/generics/functions tables, so the
    /// Checkpoint's truncate-and-retain rollback is sufficient: a rejected
    /// batch retracts them and the method name is reusable.
    #[test]
    fn rejected_batch_retracts_impl_members() {
        use std::sync::Arc;

        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);

            let p1 = parse("pub struct P { pub x: i64 }");
            elab.try_extend(&surface::program(&p1.root))
                .expect("record");

            // A batch whose impl member has a type error is rejected whole.
            let p2 = parse("impl P { pub fn get(p: P) -> i64 { true } }");
            elab.try_extend(&surface::program(&p2.root))
                .expect_err("bad member body");
            assert!(!elab.has_errors());

            // The retracted method name is free again and elaborates.
            let p3 = parse("impl P { pub fn get(p: P) -> i64 { p.x } }");
            elab.try_extend(&surface::program(&p3.root))
                .expect("redeclared member");
            let p4 = parse("fn use_it(p: P) -> i64 { P::get(p) }");
            elab.try_extend(&surface::program(&p4.root)).expect("call");
        });
    }

    #[test]
    fn receiver_value_arc_and_flex_forms_accepted() {
        check(
            "pub struct [value] V { pub v: i64 }\n\
             impl V { pub fn get(self: Self) -> i64 { self.v } }\n\
             pub struct S { pub v: i64 }\n\
             impl S { pub fn read(self: Arc<Self>) -> i64 { self.v } }\n\
             pub struct [regional] R { pub link: [field] R }\n\
             impl R { regional fn poke(self: [flex] Self) { } }",
            |elab, tcx| {
                let proto = |name: &str| {
                    elab.functions
                        .values()
                        .find(|f| elab.sym(f.name) == name)
                        .unwrap_or_else(|| panic!("method {name}"))
                };
                // The value receiver is the record itself.
                let get = proto("get");
                assert!(matches!(get.params[0].1.kind(), TyKind::Record { .. }));
                assert_eq!(Some(get.params[0].1), get.self_ty);
                // The shared receiver is one Arc around Self.
                let read = proto("read");
                let TyKind::Arc(inner) = read.params[0].1.kind() else {
                    panic!("arc receiver");
                };
                assert_eq!(Some(*inner), read.self_ty);
                // The flex receiver refines the regional target to Flex.
                let poke = proto("poke");
                assert!(poke.is_regional);
                assert!(matches!(
                    poke.params[0].1.kind(),
                    TyKind::Record {
                        flex: crate::semi::ty::Flexivity::Flex,
                        ..
                    }
                ));
                let _ = tcx;
            },
        );
    }

    #[test]
    fn plain_self_receiver_on_regional_record_is_rigid() {
        check(
            "pub struct [regional] R { pub v: i64 }\n\
             impl R { pub fn read(self: Self) -> i64 { self.v } }",
            |elab, _| {
                let read = elab
                    .functions
                    .values()
                    .find(|f| elab.sym(f.name) == "read")
                    .expect("method");
                // A plain `Self` receiver of a regional record is the frozen
                // (materializable) form.
                assert!(matches!(
                    read.params[0].1.kind(),
                    TyKind::Record {
                        flex: crate::semi::ty::Flexivity::Rigid,
                        ..
                    }
                ));
            },
        );
    }

    #[test]
    fn flex_receiver_on_non_regional_fn_reports() {
        with_tcx(|tcx| {
            let source = "pub struct [regional] R { pub v: i64 }\n\
                          impl R { fn bad(self: [flex] Self) -> i64 { self.v } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("a non-regional function cannot take flex parameter `self`")),
                "{:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn receiver_type_must_match_impl_target() {
        with_tcx(|tcx| {
            let source = "pub struct A { pub v: i64 }\npub struct B { pub v: i64 }\n\
                          impl A { fn bad(self: B) -> i64 { self.v } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("the receiver of a method must be `Self`, `Arc<Self>`")),
                "{:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn explicit_receiver_spelling_accepted() {
        check(
            "pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Box<T> { pub fn get(self: Box<T>) -> T { self.v } }",
            |elab, _| {
                let get = elab
                    .functions
                    .values()
                    .find(|f| elab.sym(f.name) == "get")
                    .expect("method");
                assert_eq!(Some(get.params[0].1), get.self_ty);
            },
        );
    }

    #[test]
    fn self_alias_resolves_in_signature_and_body_annotations() {
        check(
            "pub struct P { pub x: i64 }\n\
             impl P {\n\
                 pub fn dup(self: Self) -> Self {\n\
                     let q: Self = P { x: self.x };\n\
                     q\n\
                 }\n\
             }",
            |elab, _| {
                let dup = elab
                    .functions
                    .values()
                    .find(|f| elab.sym(f.name) == "dup")
                    .expect("method");
                assert_eq!(Some(dup.return_ty), dup.self_ty);
            },
        );
    }

    #[test]
    fn self_param_not_first_reports() {
        with_tcx(|tcx| {
            let source = "pub struct P { pub x: i64 }\n\
                          impl P { fn bad(n: i64, self: Self) -> i64 { n } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("a parameter named `self` must be the method's first parameter")),
                "{:#?}",
                elab.reports
            );
        });
    }

    fn elaborate_source(source: &str, f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>)) {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            f(&elab);
        });
    }

    #[test]
    fn method_call_dispatches_when_field_misses() {
        check(
            "pub struct P { pub x: i64 }\n\
             impl P { pub fn scaled(self: Self, k: i64) -> i64 { self.x * k } }\n\
             fn use_it(p: P) -> i64 { p.scaled(3) }",
            |elab, _| {
                let use_it = function(elab, "use_it");
                let body = use_it.body.as_ref().unwrap();
                // The dot call lowered to an ordinary FuncCall with the
                // receiver prepended.
                assert!(
                    format!("{body:?}").contains("FuncCall"),
                    "method call did not lower to FuncCall"
                );
            },
        );
    }

    /// Rust-style dispatch: `e.f(x)` resolves the method whenever one
    /// exists, even over a closure-valued field of the same name; the
    /// parenthesized callee `(e.f)(x)` forces the field.
    #[test]
    fn method_wins_over_closure_field_and_parens_force_field() {
        check(
            "pub struct S { pub f: i64 -> i64 }\n\
             impl S { pub fn f(self: Self, k: i64) -> i64 { k } }\n\
             fn dot(s: S) -> i64 { s.f(1) }\n\
             fn parens(s: S) -> i64 { (s.f)(1) }",
            |elab, _| {
                let dot = function(elab, "dot");
                assert!(
                    format!("{:?}", dot.body.as_ref().unwrap()).contains("FuncCall"),
                    "the method must win the dot call"
                );
                let parens = function(elab, "parens");
                assert!(
                    format!("{:?}", parens.body.as_ref().unwrap()).contains("ClosureCall"),
                    "the parenthesized callee must force the field"
                );
            },
        );
    }

    /// A non-closure field of the same name never obstructs the method —
    /// dispatch is by method existence, not field type.
    #[test]
    fn method_wins_over_non_closure_field() {
        check(
            "pub struct S { pub f: i64 }\n\
             impl S { pub fn f(self: Self, k: i64) -> i64 { self.f + k } }\n\
             fn use_it(s: S) -> i64 { s.f(2) }",
            |elab, _| {
                let use_it = function(elab, "use_it");
                assert!(
                    format!("{:?}", use_it.body.as_ref().unwrap()).contains("FuncCall"),
                    "the method must win over the scalar field"
                );
            },
        );
    }

    /// Visibility never redirects dispatch: a private field cannot block a
    /// `pub` method of the same name from another module.
    #[test]
    fn private_field_does_not_block_pub_method() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (
                    &["a"],
                    "pub struct S { f: i64 }\n\
                     impl S {\n\
                         pub fn make() -> S { S { f: 40 } }\n\
                         pub fn f(self: Self, k: i64) -> i64 { self.f + k }\n\
                     }",
                ),
                (&["c"], "pub fn go() -> i64 { super::a::S::make().f(2) }"),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn method_call_through_arc_receiver() {
        check(
            "pub struct S { pub v: i64 }\n\
             impl S { pub fn read(self: Arc<Self>) -> i64 { self.v } }\n\
             fn use_it(a: Arc<S>) -> i64 { a.read() }",
            |elab, _| {
                let _ = elab;
            },
        );
    }

    #[test]
    fn bare_base_against_arc_receiver_reports() {
        elaborate_source(
            "pub struct S { pub v: i64 }\n\
             impl S { pub fn read(self: Arc<Self>) -> i64 { self.v } }\n\
             fn use_it(s: S) -> i64 { s.read() }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains("Arc")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn flex_receiver_method_calls_inside_region_only() {
        check(
            "pub struct [regional] R { pub link: [field] R }\n\
             impl R { regional fn poke(self: [flex] Self) { { } } }\n\
             regional fn go(r: [flex] R) { r.poke() }",
            |elab, _| {
                let _ = elab;
            },
        );
        elaborate_source(
            "pub struct [regional] R { pub link: [field] R }\n\
             impl R { regional fn poke(self: [flex] Self) { { } } }\n\
             fn bad(r: R) { r.poke() }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("cannot call a regional function outside of a region")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// An impl-level bound is enforced at the method call, dot and path
    /// spellings alike.
    #[test]
    fn impl_bound_violation_reports_at_method_call() {
        elaborate_source(
            "pub struct Box<T> { pub v: T }
             impl<T: Num> Box<T> { pub fn scaled(self: Box<T>, k: T) -> T { self.v * k } }
             fn bad(b: Box<bool>) -> bool { b.scaled(true) }
             fn also_bad(b: Box<bool>) -> bool { Box::scaled(b, true) }",
            |elab| {
                let violations = elab
                    .reports
                    .iter()
                    .filter(|r| r.message.contains("Num"))
                    .count();
                assert!(violations >= 2, "{:#?}", elab.reports);
            },
        );
    }

    #[test]
    fn generic_method_infers_type_args_from_receiver_and_args() {
        check(
            "pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Box<T> { pub fn scaled(self: Box<T>, k: T) -> T { self.v * k } }\n\
             fn use_it(b: Box<i64>) -> i64 { b.scaled(3) }",
            |elab, _| {
                let _ = elab;
            },
        );
    }

    #[test]
    fn assoc_fn_without_receiver_is_not_dot_callable() {
        elaborate_source(
            "pub struct P { pub x: i64 }\n\
             impl P { pub fn make(x: i64) -> P { P { x: x } } }\n\
             fn bad(p: P) -> P { p.make(1) }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("takes no receiver and is not callable with `.`")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn no_field_or_method_diagnostic_in_call_position() {
        elaborate_source(
            "pub struct P { pub x: i64 }\n\
             fn bad(p: P) -> i64 { p.nope(1) }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("no field or method `nope` on `P`")),
                    "{:#?}",
                    elab.reports
                );
                // The plain access path keeps its own wording.
                assert!(
                    !elab.reports.iter().any(|r| r.message == "no such field"),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn method_partial_application_rejected_with_hint() {
        elaborate_source(
            "pub struct P { pub x: i64 }\n\
             impl P { pub fn add(self: Self, a: i64, b: i64) -> i64 { self.x + a + b } }\n\
             fn bad(p: P) -> i64 { p.add(1) }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("partial application of a method call is not supported")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn try_extend_accumulates_and_rolls_back_atomically() {
        use std::sync::Arc;

        with_tcx(|tcx| {
            // The REPL flow: many small parses sharing one interner, one
            // persistent elaborator.
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);

            // Input 1: a definition is accepted.
            let p1 = parse("fn double(x: i32) -> i32 { x + x }");
            elab.try_extend(&surface::program(&p1.root))
                .expect("first definition");
            assert_eq!(elab.elaborated.len(), 1);

            // Input 2: a batch containing a duplicate is rejected wholesale —
            // including its error-free items.
            let p2 = parse(
                "#[transform_anchor]\n\
                 fn triple(x: i32) -> i32 { 3 * x }\n\
                 transform [{ transform.yield }];\n\
                 fn double(x: i32) -> i32 { x }",
            );
            let errs = elab
                .try_extend(&surface::program(&p2.root))
                .expect_err("duplicate must be rejected");
            assert!(
                errs.iter()
                    .any(|r| r.message.contains("defined more than once")),
                "{errs:#?}"
            );
            assert_eq!(elab.elaborated.len(), 1, "batch rolled back");
            assert!(elab.transform_anchors.is_empty(), "anchors rolled back");
            assert!(elab.transform_scripts.is_empty(), "scripts rolled back");
            assert!(!elab.has_errors(), "rejected reports don't persist");

            // Input 3: `triple` was rolled back, so its name is free again,
            // and cross-input references (`double`) resolve.
            let p3 = parse("fn triple(x: i32) -> i32 { double(x) + x }");
            elab.try_extend(&surface::program(&p3.root))
                .expect("retracted name is reusable");
            assert_eq!(elab.elaborated.len(), 2);

            // Input 4: a type error rolls back the whole batch too.
            let p4 = parse("fn bad() -> i32 { true }");
            elab.try_extend(&surface::program(&p4.root))
                .expect_err("type mismatch");
            assert_eq!(elab.elaborated.len(), 2);

            // A record checkpointed away frees its fields and name as well.
            let p5 = parse("struct P { x: i32 }\nstruct P { y: i32 }");
            elab.try_extend(&surface::program(&p5.root))
                .expect_err("duplicate record in one batch");
            assert!(elab.records.values().all(|r| elab.sym(r.name) != "P"));
            let p6 = parse("struct P { x: i32, y: i32 }");
            elab.try_extend(&surface::program(&p6.root))
                .expect("name free after rollback");
        });
    }

    #[test]
    fn residual_holes_are_ambiguity_errors_in_batch_mode() {
        with_tcx(|tcx| {
            // Nothing constrains the element type of a bare `Nullable::Null`
            // (no obligation is registered, so there is no bound to fail
            // either): a legitimate inference outcome, not an ICE. Zonking
            // must report it instead of handing monomorphization a type
            // `subst_ty` would panic on.
            let source = "fn f() -> i64 { let x = Nullable::Null; 42 }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            let ambiguous: Vec<_> = elab
                .reports
                .iter()
                .filter(|r| r.message.contains("cannot infer"))
                .collect();
            // Exactly one report, at the offending expression — not one per
            // enclosing node that shares the hole.
            assert_eq!(ambiguous.len(), 1, "reports: {:#?}", elab.reports);
        });
    }

    #[test]
    fn duplicate_definitions_are_rejected_without_clobbering() {
        with_tcx(|tcx| {
            let source = "struct P { x: i32, y: i32 }\n\
                          struct P { z: f64 }\n\
                          fn f(a: i32) -> i32 { a }\n\
                          fn f(b: f64, c: f64) -> f64 { b }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);

            let duplicates = elab
                .reports
                .iter()
                .filter(|r| r.message.contains("defined more than once"))
                .count();
            assert_eq!(duplicates, 2, "reports: {:#?}", elab.reports);

            // The first `P` keeps its own fields — the duplicate declaration
            // must not repopulate the surviving record.
            let p = elab
                .records
                .values()
                .find(|r| elab.sym(r.name) == "P")
                .expect("record P");
            let crate::semi::ctxt::RecordFields::Named(fields) =
                p.fields.as_ref().expect("populated")
            else {
                panic!("named fields");
            };
            assert_eq!(fields.len(), 2);

            // Only the first `f` was checked; the duplicate's body was not
            // elaborated against the surviving prototype.
            let fs: Vec<_> = elab
                .elaborated
                .iter()
                .filter(|f| elab.sym(f.name) == "f")
                .collect();
            assert_eq!(fs.len(), 1);
            assert_eq!(fs[0].params.len(), 1);
        });
    }

    #[test]
    fn field_visibility_lands_in_record_tables() {
        use crate::surface::Visibility::{Private, Public};

        check(
            "pub struct S { pub a: i64, b: i64 }\nstruct P(pub i64, bool)",
            |elab, _| {
                let fields_of = |name: &str| {
                    elab.records
                        .values()
                        .find(|r| elab.sym(r.name) == name)
                        .expect("record")
                        .fields
                        .clone()
                        .expect("populated")
                };
                let crate::semi::ctxt::RecordFields::Named(fs) = fields_of("S") else {
                    panic!("named fields");
                };
                assert_eq!(
                    fs.iter().map(|&(_, _, m, v)| (m, v)).collect::<Vec<_>>(),
                    [(false, Public), (false, Private)]
                );
                let crate::semi::ctxt::RecordFields::Unnamed(fs) = fields_of("P") else {
                    panic!("unnamed fields");
                };
                assert_eq!(
                    fs.iter().map(|&(_, m, v)| (m, v)).collect::<Vec<_>>(),
                    [(false, Public), (false, Private)]
                );
            },
        );
    }

    #[test]
    fn trait_impl_conformance_accepted_and_bodies_check() {
        check(
            "pub trait Show { fn show(self: Self, k: i64) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl Show for P { fn show(self: Self, k: i64) -> i64 { self.x * k } }",
            |elab, _| {
                // The member declared under the trait-impl path and its body
                // checked through the common flow.
                let show = elab
                    .functions
                    .values()
                    .find(|f| elab.sym(f.name) == "show")
                    .expect("declared");
                assert_eq!(
                    elab.defs.path(show.def).display(elab.resolver),
                    "Show::P::show"
                );
                assert!(elab.elaborated.iter().any(|f| f.def == show.def));
            },
        );
    }

    #[test]
    fn trait_impl_conformance_matrix() {
        let hdr = "pub trait Show { fn show(self: Self, k: i64) -> i64; }\n\
                   pub struct P { pub x: i64 }\n";
        for (src, needle) in [
            ("impl Show for P { }", "is missing method(s) `show`"),
            (
                "impl Show for P { fn show(self: Self, k: i64) -> i64 { k } \
                 fn extra(self: Self) -> i64 { 0 } }",
                "`extra` is not a member of trait `Show`",
            ),
            (
                "impl Show for P { fn show(self: Self, k: bool) -> i64 { 1 } }",
                "parameter 1 of method `show` has type `bool` but trait `Show` requires `i64`",
            ),
            (
                "impl Show for P { fn show(self: Self, k: i64) -> bool { true } }",
                "returns `bool` but trait `Show` requires `i64`",
            ),
            (
                "impl Show for P { fn show(self: Arc<Self>, k: i64) -> i64 { k } }",
                "the receiver of `show` must be `self: Self` to match trait `Show`",
            ),
            (
                "impl Show for P { regional fn show(self: Self, k: i64) -> i64 { k } }",
                "is not `regional` in trait `Show` but is in this impl",
            ),
            (
                "impl Show for P { fn show<U: Num>(self: Self, k: i64) -> i64 { k } }",
                "declares 1 generic parameter(s); trait `Show` declares 0",
            ),
            (
                "impl Show for P { pub fn show(self: Self, k: i64) -> i64 { k } }",
                "visibility is not allowed on a trait impl method",
            ),
        ] {
            let source = format!("{hdr}{src}");
            elaborate_source(&source, |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains(needle)),
                    "wanted {needle:?} in {:#?}",
                    elab.reports
                );
            });
        }
    }

    #[test]
    fn sealed_builtins_cannot_be_implemented() {
        for tr in ["Num", "Integral", "FloatingPoint", "PtrLike", "Sync"] {
            let source = format!("pub struct P {{ pub x: i64 }}\nimpl {tr} for P {{ }}");
            elaborate_source(&source, |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("is a built-in trait and cannot be implemented")),
                    "{tr}: {:#?}",
                    elab.reports
                );
            });
        }
    }

    #[test]
    fn trait_and_inherent_methods_coexist() {
        check(
            "pub trait Show { fn norm(self: Self) -> i64; }\n\
             pub struct V { pub x: i64 }\n\
             impl V { pub fn norm(self: Self) -> i64 { self.x } }\n\
             impl Show for V { fn norm(self: Self) -> i64 { 0 - self.x } }",
            |elab, _| {
                // Distinct paths: [V, norm] inherent, [Show, V, norm] trait.
                let paths: Vec<String> = elab
                    .functions
                    .values()
                    .filter(|f| elab.sym(f.name) == "norm")
                    .map(|f| elab.defs.path(f.def).display(elab.resolver))
                    .collect();
                assert_eq!(paths.len(), 2, "{paths:?}");
                assert!(paths.contains(&"V::norm".to_string()));
                assert!(paths.contains(&"Show::V::norm".to_string()));
            },
        );
    }

    #[test]
    fn duplicate_ground_trait_impls_hit_the_guided_clash() {
        // Identical impls are an overlap conflict (coherence speaks first)…
        elaborate_source(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { 1 } }\n\
             impl Show for P { fn show(self: Self) -> i64 { 2 } }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("conflicting implementations of trait")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        // …while DISJOINT impls in one module collide only on the method
        // path, and get steered toward a single parameterized impl.
        elaborate_source(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct Box<T> { pub v: T }\n\
             impl Show for Box<i32> { fn show(self: Self) -> i64 { 1 } }\n\
             impl Show for Box<bool> { fn show(self: Self) -> i64 { 2 } }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("already declared by another impl; merge the impls")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn impl_where_clauses_from_inline_bounds() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Show for Box<T> { fn show(self: Self) -> i64 { 1 } }",
            |elab, _| {
                let imp = (0..elab.traits.impls_len() as u32)
                    .map(crate::semi::traits::ImplId)
                    .map(|i| elab.traits.impl_def(i))
                    .find(|i| !i.generics.is_empty())
                    .expect("generic impl registered");
                assert_eq!(imp.where_clauses.len(), 1, "T: Num recorded");
                assert_eq!(imp.methods.len(), 1);
            },
        );
    }

    #[test]
    fn impl_generic_must_be_constrained() {
        elaborate_source(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl<T: Num> Show for P { fn show(self: Self) -> i64 { 1 } }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("impl generic `T` is not constrained")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn cross_module_duplicate_impls_hit_overlap() {
        // Different modules dodge the method-path clash; the (trait, head)
        // bucket catches the conflict.
        check_package(
            "p",
            &[
                (
                    &[],
                    "mod a; mod b;\n\
                     pub trait Show { fn show(self: Self) -> i64; }\n\
                     pub struct P { pub x: i64 }",
                ),
                (
                    &["a"],
                    "impl super::Show for super::P { fn show(self: Self) -> i64 { 1 } }",
                ),
                (
                    &["b"],
                    "impl super::Show for super::P { fn show(self: Self) -> i64 { 2 } }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("conflicting implementations of trait")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn crossing_generic_impls_overlap() {
        check_package(
            "p",
            &[
                (
                    &[],
                    "mod a; mod b;\n\
                     pub trait Show { fn show(self: Self) -> i64; }\n\
                     pub struct Pair<A, B> { pub a: A, pub b: B }",
                ),
                (
                    &["a"],
                    "impl<T: Num> super::Show for super::Pair<T, i32> {\n\
                         fn show(self: Self) -> i64 { 1 }\n\
                     }",
                ),
                (
                    &["b"],
                    "impl<U: Num> super::Show for super::Pair<u8, U> {\n\
                         fn show(self: Self) -> i64 { 2 }\n\
                     }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("conflicting implementations of trait")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn disjoint_ground_impls_coexist_across_modules() {
        check_package(
            "p",
            &[
                (
                    &[],
                    "mod a; mod b;\n\
                     pub trait Show { fn show(self: Self) -> i64; }\n\
                     pub struct Box<T> { pub v: T }",
                ),
                (
                    &["a"],
                    "impl super::Show for super::Box<i32> { fn show(self: Self) -> i64 { 1 } }",
                ),
                (
                    &["b"],
                    "impl super::Show for super::Box<bool> { fn show(self: Self) -> i64 { 2 } }",
                ),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn orphan_rule_package_granularity() {
        use std::sync::Arc;
        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);
            let p1 = parse(
                "pub trait Ext { fn e(self: Self) -> i64; }\n\
                 pub struct Far { pub x: i64 }",
            );
            elab.try_extend(&surface::program(&p1.root)).expect("decls");
            // Mark both as extern-owned, simulating .rri provenance (trait
            // serialization lands in a later stack; the gate reads only the
            // provenance map).
            let head = interner.intern("dep");
            let trait_def = (0..elab.defs.len() as u32)
                .map(crate::semi::ty::DefId)
                .find(|d| {
                    elab.defs.info(*d).kind == crate::semi::resolve::DefKind::Trait
                        && elab.sym(elab.defs.path(*d).name()) == "Ext"
                })
                .expect("trait def");
            let far_def = (0..elab.defs.len() as u32)
                .map(crate::semi::ty::DefId)
                .find(|d| {
                    elab.defs.info(*d).kind == crate::semi::resolve::DefKind::Record
                        && elab.sym(elab.defs.path(*d).name()) == "Far"
                })
                .expect("record def");
            elab.extern_defs.insert(trait_def, head);
            elab.extern_defs.insert(far_def, head);

            // Both extern: rejected by the orphan rule.
            let p2 = parse("impl Ext for Far { fn e(self: Self) -> i64 { 1 } }");
            let errs = elab
                .try_extend(&surface::program(&p2.root))
                .expect_err("orphan");
            assert!(
                errs.iter().any(|r| r
                    .message
                    .contains("the trait or the self type's head must be local")),
                "{errs:#?}"
            );

            // A local trait for the extern type is admissible.
            let p3 = parse(
                "pub trait Local { fn l(self: Self) -> i64; }\n\
                 impl Local for Far { fn l(self: Self) -> i64 { self.x } }",
            );
            elab.try_extend(&surface::program(&p3.root))
                .expect("local trait, extern head");

            // The extern trait for a local type is admissible too.
            let p4 = parse(
                "pub struct Near { pub y: i64 }\n\
                 impl Ext for Near { fn e(self: Self) -> i64 { self.y } }",
            );
            elab.try_extend(&surface::program(&p4.root))
                .expect("extern trait, local head");
        });
    }

    #[test]
    fn generic_impl_overlaps_its_ground_instance() {
        check_package(
            "p",
            &[
                (
                    &[],
                    "mod a; mod b;\n\
                     pub trait Show { fn show(self: Self) -> i64; }\n\
                     pub struct Box<T> { pub v: T }",
                ),
                (
                    &["a"],
                    "impl<T: Num> super::Show for super::Box<T> {\n\
                         fn show(self: Self) -> i64 { 1 }\n\
                     }",
                ),
                (
                    &["b"],
                    "impl super::Show for super::Box<i32> { fn show(self: Self) -> i64 { 2 } }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("conflicting implementations of trait")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn generic_impls_are_selected_through_bounds() {
        // The one-way matcher at work: `use_show` needs `U: Show`, the call
        // instantiates `U = Box<T>`, and the generic impl applies with its
        // `T: Num` clause discharged by `go`'s own assumption.
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Show for Box<T> { fn show(self: Self) -> i64 { 1 } }\n\
             pub fn use_show<U: Show>(u: U) -> i64 { 2 }\n\
             pub fn go<T: Num>(b: Box<T>) -> i64 { use_show(b) }",
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn where_clause_failure_names_the_root_cause() {
        // `Box<bool>` matches the impl head, but its `T: Num` clause has no
        // proof — the diagnostic must surface the inner goal, not just the
        // outer one.
        elaborate_source(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Show for Box<T> { fn show(self: Self) -> i64 { 1 } }\n\
             pub fn use_show<U: Show>(u: U) -> i64 { 2 }\n\
             pub fn go(b: Box<bool>) -> i64 { use_show(b) }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains(
                        "`Box<bool>` does not implement `Show`: \
                         `bool` does not implement `Num`"
                    )),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// The function named `name` in the elaborated set, with its body.
    #[cfg(test)]
    fn body_of<'a, 'tcx>(
        elab: &'a Elaborator<'_, 'tcx>,
        name: &str,
    ) -> &'a crate::semi::hir::Expr<'tcx> {
        elab.elaborated
            .iter()
            .find(|f| elab.sym(f.name) == name)
            .and_then(|f| f.body.as_ref())
            .expect("function with a body")
    }

    /// Collect `(is_trait_call, target_path)` for every call in `body` —
    /// `target_path` rendered as `::`-joined segments for `FuncCall`s.
    #[cfg(test)]
    fn call_targets(elab: &Elaborator<'_, '_>, name: &str) -> Vec<(bool, String)> {
        let mut out = Vec::new();
        crate::full::mono::for_each_expr(body_of(elab, name), &mut |e| match &e.kind {
            crate::semi::hir::ExprKind::FuncCall { target, .. } => {
                let path = elab
                    .defs
                    .path(*target)
                    .0
                    .iter()
                    .map(|k| elab.sym(*k).to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                out.push((false, path));
            }
            crate::semi::hir::ExprKind::TraitCall { trait_def, .. } => {
                let path = elab
                    .defs
                    .path(*trait_def)
                    .0
                    .iter()
                    .map(|k| elab.sym(*k).to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                out.push((true, path));
            }
            _ => {}
        });
        out
    }

    #[test]
    fn trait_methods_dispatch_on_dot_for_ground_receivers() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
             pub fn go(p: P) -> i64 { p.show() }",
            |elab, _| {
                let calls = call_targets(elab, "go");
                // Ground receiver: committed to the impl method at check
                // time — an ordinary FuncCall through the impl's path.
                assert_eq!(calls.len(), 1, "{calls:?}");
                assert!(!calls[0].0, "no TraitCall for a ground receiver");
                assert_eq!(calls[0].1, "Show::P::show");
            },
        );
    }

    #[test]
    fn trait_methods_dispatch_on_dot_for_generic_receivers() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
             pub fn go<T: Show>(x: T) -> i64 { x.show() }",
            |elab, _| {
                let calls = call_targets(elab, "go");
                assert_eq!(calls.len(), 1, "{calls:?}");
                assert!(calls[0].0, "generic receiver defers to a TraitCall");
                assert_eq!(calls[0].1, "Show");
            },
        );
    }

    #[test]
    fn inherent_methods_win_over_trait_methods() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl P { pub fn show(self: P) -> i64 { 1 } }\n\
             impl Show for P { fn show(self: Self) -> i64 { 2 } }\n\
             pub fn go(p: P) -> i64 { p.show() }",
            |elab, _| {
                let calls = call_targets(elab, "go");
                assert_eq!(calls.len(), 1, "{calls:?}");
                // The inherent path is `P::show`; the trait impl's method
                // lives under `Show::P::show`.
                assert_eq!(calls[0], (false, "P::show".to_string()), "{calls:?}");
            },
        );
    }

    #[test]
    fn trait_method_wins_over_field_and_parens_force_the_field() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub show: (i64) -> i64, pub x: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
             pub fn method(p: P) -> i64 { p.show() }\n\
             pub fn field(p: P) -> i64 { (p.show)(3) }",
            |elab, _| {
                let m = call_targets(elab, "method");
                assert_eq!(m.len(), 1, "{m:?}");
                assert_eq!(m[0].1, "Show::P::show", "method-first, ahead of the field");
                assert!(
                    call_targets(elab, "field").is_empty(),
                    "paren form is a closure call"
                );
            },
        );
    }

    #[test]
    fn ambiguous_trait_methods_are_reported() {
        elaborate_source(
            "pub trait A { fn m(self: Self) -> i64; }\n\
             pub trait B { fn m(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl A for P { fn m(self: Self) -> i64 { 1 } }\n\
             impl B for P { fn m(self: Self) -> i64 { 2 } }\n\
             pub fn go(p: P) -> i64 { p.m() }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("multiple traits provide method `m`")
                            && r.message.contains("`A` and `B`")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn trait_path_calls_dispatch_both_ways() {
        check(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
             pub fn ground(p: P) -> i64 { Show::show(p) }\n\
             pub fn generic<T: Show>(x: T) -> i64 { Show::show(x) }",
            |elab, _| {
                let g = call_targets(elab, "ground");
                assert_eq!(g, vec![(false, "Show::P::show".to_string())], "{g:?}");
                let t = call_targets(elab, "generic");
                assert_eq!(t, vec![(true, "Show".to_string())], "{t:?}");
            },
        );
    }

    #[test]
    fn trait_dispatch_failure_diagnostics() {
        elaborate_source(
            "pub trait Show { fn show(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             pub struct Q { pub y: i64 }\n\
             impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
             pub fn miss(q: Q) -> i64 { q.show() }\n\
             pub fn unbounded<T>(x: T) -> i64 { x.show() }\n\
             pub fn wrong(p: P) -> i64 { Show::nope(p) }",
            |elab| {
                let has = |m: &str| elab.reports.iter().any(|r| r.message.contains(m));
                assert!(
                    has("no field or method `show` on `Q`; trait `Show` declares it"),
                    "{:#?}",
                    elab.reports
                );
                assert!(
                    has("no method `show` on `T`; its bounds do not declare one"),
                    "{:#?}",
                    elab.reports
                );
                assert!(
                    has("trait `Show` has no method `nope`"),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn supertrait_methods_reach_through_bounds() {
        check(
            "pub trait A { fn m(self: Self) -> i64; }\n\
             pub trait B : A { fn extra(self: Self) -> i64; }\n\
             pub struct P { pub x: i64 }\n\
             impl A for P { fn m(self: Self) -> i64 { self.x } }\n\
             pub fn go<T: B>(x: T) -> i64 { x.m() }",
            |elab, _| {
                let calls = call_targets(elab, "go");
                // The `B` bound reaches `A::m` through the super edge; the
                // TraitCall names the declaring trait.
                assert_eq!(calls, vec![(true, "A".to_string())], "{calls:?}");
            },
        );
    }

    #[test]
    fn arc_receiver_trait_methods_dispatch() {
        check(
            "pub trait Shared { fn get(self: Arc<Self>) -> i64; }\n\
             pub struct [shared] P { pub x: i64 }\n\
             impl Shared for P { fn get(self: Arc<Self>) -> i64 { self.x } }\n\
             pub fn go(a: Arc<P>) -> i64 { a.get() }",
            |elab, _| {
                let calls = call_targets(elab, "go");
                assert_eq!(calls.len(), 1, "{calls:?}");
                assert_eq!(calls[0].1, "Shared::P::get");
            },
        );
    }

    #[test]
    fn lang_items_declare_resolve_and_locate() {
        use crate::semi::lang::LangItem;
        check(
            "#[lang(\"partial_eq\")]\n\
             pub trait PartialEq { fn eq(self: Self, other: Self) -> bool; }\n\
             #[lang(\"ordering\")]\n\
             pub enum Ordering { Less(), Equal(), Greater() }",
            |elab, _| {
                let pe = elab.lang.get(LangItem::PartialEq).expect("declared");
                assert_eq!(elab.sym(elab.defs.path(pe).name()), "PartialEq");
                let (_, span) = elab.lang_item_site(LangItem::PartialEq).expect("site");
                assert!(span.is_some(), "source declarations carry locations");
                assert!(elab.lang.get(LangItem::Ordering).is_some());
                // The fallback tower is compiler-provided: no declared site.
                assert!(elab.lang_item_site(LangItem::Num).is_none());
            },
        );
    }

    #[test]
    fn lang_item_diagnostics() {
        elaborate_source(
            "#[lang(\"nope\")] pub trait A { fn a(self: Self) -> i64; }\n\
             #[lang(\"eq\")] pub struct B { pub x: i64 }\n\
             #[lang(\"ordering\")] pub trait C { fn c(self: Self) -> i64; }\n\
             #[lang(\"partial_eq\")] pub trait D { fn d(self: Self) -> i64; }\n\
             #[lang(\"partial_eq\")] pub trait E { fn e(self: Self) -> i64; }\n\
             #[lang(\"ord\")] pub fn f() -> i64 { 1 }",
            |elab| {
                let has = |m: &str| elab.reports.iter().any(|r| r.message.contains(m));
                assert!(has("unknown lang item `nope`"), "{:#?}", elab.reports);
                assert!(has("lang item `eq` must be a trait"), "{:#?}", elab.reports);
                assert!(
                    has("lang item `ordering` must be a struct or enum"),
                    "{:#?}",
                    elab.reports
                );
                assert!(
                    has("lang item `partial_eq` is already declared by `D`"),
                    "{:#?}",
                    elab.reports
                );
                assert!(
                    has("lang item `ord` must be a trait"),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn rejected_batch_retracts_lang_items() {
        use std::sync::Arc;

        use crate::semi::lang::LangItem;
        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);
            // A batch that declares a lang item but fails elsewhere retracts
            // the declaration with everything else…
            let p1 = parse(
                "#[lang(\"partial_eq\")]\npub trait P { fn e(self: Self) -> bool; }\n\
                 fn bad() -> i64 { nonexistent() }",
            );
            elab.try_extend(&surface::program(&p1.root))
                .expect_err("batch fails");
            assert!(elab.lang.get(LangItem::PartialEq).is_none(), "retracted");
            // …so the clean re-declaration is fresh, not a duplicate.
            let p2 = parse("#[lang(\"partial_eq\")]\npub trait P { fn e(self: Self) -> bool; }");
            elab.try_extend(&surface::program(&p2.root)).expect("clean");
            assert!(elab.lang.get(LangItem::PartialEq).is_some());
        });
    }

    #[test]
    fn intrinsic_lang_items_bind_prototypes() {
        use crate::intrinsic::MathFn;
        use crate::semi::lang::{IntrinsicItem, LangItem};
        check(
            "#[lang(\"core::intrinsic::math::sqrt\")]\n\
             pub fn my_sqrt<F: FloatingPoint>(x: F, fastmath: i64) -> F;",
            |elab, _| {
                let item = LangItem::Intrinsic(IntrinsicItem::Math(MathFn::Sqrt));
                let def = elab.lang.get(item).expect("bound");
                assert!(elab.lang.is_intrinsic_def(def));
                assert!(elab.lang_item_site(item).is_some());
            },
        );
    }

    #[test]
    fn intrinsic_prototypes_reject_bodies_and_value_uses() {
        elaborate_source(
            "#[lang(\"core::intrinsic::math::sqrt\")]\n\
             pub fn bad<F: FloatingPoint>(x: F, fastmath: i64) -> F { x }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("an intrinsic prototype must not have a body")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        elaborate_source(
            "#[lang(\"core::intrinsic::math::sqrt\")]\n\
             pub fn my_sqrt<F: FloatingPoint>(x: F, fastmath: i64) -> F;\n\
             pub fn use_it() -> i64 { let g = my_sqrt; 0 }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("is an intrinsic and cannot be used as a value")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// The bundled `library/core` sources stay honest: they elaborate
    /// cleanly and bind a prototype for *every* math intrinsic the
    /// compiler surfaces — a new `MathFn` without a core declaration (or
    /// vice versa) fails here, in `cargo test`, before any rene build.
    #[test]
    fn bundled_core_covers_every_math_intrinsic() {
        use crate::intrinsic::MathFn;
        use crate::semi::lang::{IntrinsicItem, LangItem};
        check_package(
            "core",
            &[
                (&[], include_str!("../../../library/core/src/lib.rr")),
                (&["cmp"], include_str!("../../../library/core/src/cmp.rr")),
                (&["num"], include_str!("../../../library/core/src/num.rr")),
                (
                    &["marker"],
                    include_str!("../../../library/core/src/marker.rr"),
                ),
                (
                    &["intrinsic"],
                    include_str!("../../../library/core/src/intrinsic/mod.rr"),
                ),
                (
                    &["intrinsic", "math"],
                    include_str!("../../../library/core/src/intrinsic/math.rr"),
                ),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
                for &f in MathFn::ALL {
                    let item = LangItem::Intrinsic(IntrinsicItem::Math(f));
                    assert!(
                        elab.lang.get(item).is_some(),
                        "core declares no prototype for `{}`",
                        f.as_str()
                    );
                }
                // `core::cmp` binds the whole comparison tower, with sites.
                for item in LangItem::CMP_TRAITS.into_iter().chain([LangItem::Ordering]) {
                    let (_, span) = elab
                        .lang_item_site(item)
                        .unwrap_or_else(|| panic!("core does not declare `{}`", item.name()));
                    assert!(span.is_some(), "`{}` has no source span", item.name());
                }
                // Suppression held: nothing compiler-provided was declared
                // for the tower, and the declared traits carry methods.
                let po = elab
                    .traits
                    .trait_by_def(elab.lang.get(LangItem::PartialOrd).expect("declared"))
                    .expect("trait-kinded");
                assert_eq!(elab.traits.trait_def(po).methods.len(), 4);
                // The numeric/marker tower: declared, sealed, sited, and
                // the wired fields repointed onto the declarations.
                for (item, field) in LangItem::TOWER.into_iter().zip([
                    elab.lang.num,
                    elab.lang.integral,
                    elab.lang.floating_point,
                    elab.lang.ptr_like,
                    elab.lang.sync,
                ]) {
                    let def = elab
                        .lang
                        .get(item)
                        .unwrap_or_else(|| panic!("core does not declare `{}`", item.name()));
                    assert_eq!(
                        elab.traits.trait_by_def(def),
                        Some(field),
                        "`{}`'s wired field must follow the declaration",
                        item.name()
                    );
                    let t = elab.traits.trait_def(field);
                    assert!(t.sealed && t.span.is_some(), "`{}`", item.name());
                }
                // `Num ⊴ PartialOrd` comes from source, targeting core's
                // own PartialOrd.
                assert!(
                    elab.traits
                        .trait_def(elab.lang.num)
                        .supertraits
                        .iter()
                        .any(|s| s.trait_id == po)
                );
            },
        );
    }

    /// `#[sealed]`: declarable on traits only, argument-free, and an
    /// impl of a sealed trait is rejected wherever it was declared.
    #[test]
    fn sealed_attr_matrix() {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let src = r#"
                #[sealed]
                pub trait Machine { }
                pub struct P { pub x: i64 }
                impl Machine for P { }
                #[sealed]
                pub struct Q { pub x: i64 }
                #[sealed("x")]
                pub trait Noisy { }
            "#;
            let parse = reussir_syntax::parse_with_interner(src, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            let msgs: Vec<_> = elab.reports.iter().map(|r| r.message.as_str()).collect();
            assert!(
                msgs.iter()
                    .any(|m| m.contains("`Machine` is a built-in trait and cannot be implemented")),
                "{msgs:#?}"
            );
            assert!(
                msgs.iter()
                    .any(|m| m.contains("only a trait can be `#[sealed]`")),
                "{msgs:#?}"
            );
            assert!(
                msgs.iter()
                    .any(|m| m.contains("`#[sealed]` takes no arguments")),
                "{msgs:#?}"
            );
        });
    }

    /// Declaring a numeric/marker-tower lang item repoints its wired field
    /// — everything the checker routes through `lang.num`/…: literal
    /// bounds, arithmetic, defaulting, the `Num ⊴ PartialOrd` edge — and a
    /// rejected batch restores the builtins.
    #[test]
    fn declared_tower_repoints_and_rolls_back() {
        use crate::semi::lang::LangItem;
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let mut elab = Elaborator::new(tcx, &interner);
            let builtin_num = elab.lang.num;
            let builtin_integral = elab.lang.integral;
            let batch = |src: &str, interner: &std::sync::Arc<_>| {
                let parse =
                    reussir_syntax::parse_with_interner(src, std::sync::Arc::clone(interner));
                assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
                surface::program(&parse.root)
            };
            let declare = batch(
                "#[lang(\"num\")] #[sealed] pub trait MyNum: PartialOrd { }\n\
                 fn d<T: Num>(a: T, b: T) -> bool { a + a < b }",
                &interner,
            );
            let r = elab.try_extend(&declare);
            assert!(r.is_ok(), "batch rejected: {r:#?}");
            let declared = elab
                .traits
                .trait_by_def(elab.lang.get(LangItem::Num).expect("declared"))
                .expect("trait-kinded");
            assert_ne!(declared, builtin_num, "the wired field repoints");
            assert_eq!(elab.lang.num, declared);
            assert!(elab.traits.trait_def(declared).span.is_some());

            let broken = batch(
                "#[lang(\"integral\")] pub trait MyIntegral { }\n\
                 fn nope() -> i64 { missing() }",
                &interner,
            );
            assert!(elab.try_extend(&broken).is_err());
            assert_eq!(
                elab.lang.integral, builtin_integral,
                "a rejected batch restores the builtin field"
            );
            assert_eq!(elab.lang.num, declared, "accepted repoints survive");
        });
    }

    /// Bare names resolve through the prelude: with the tower declared in
    /// a module (exactly as `core` ships it), unqualified `PartialEq`,
    /// `Ord`, and `Ordering` still resolve everywhere they can appear —
    /// bounds, impl heads, signature types, constructors, and patterns.
    #[test]
    fn prelude_resolves_lang_names_across_modules() {
        check_package(
            "p",
            &[
                (
                    &[],
                    "pub struct P { pub x: i64 }\n\
                     impl PartialEq for P { fn eq(self: Self, other: Self) -> bool { self.x == other.x } }\n\
                     pub fn eqs(a: P, b: P) -> bool { a == b }\n\
                     fn tw<T: Ord>(a: T, b: T) -> Ordering { a.cmp(b) }\n\
                     pub fn go(x: i64, y: i64) -> i64 {\n\
                         match tw(x, y) { Ordering::Less => 0, Ordering::Equal => 1, Ordering::Greater => 2 }\n\
                     }",
                ),
                (&["cmp"], include_str!("../../../library/core/src/cmp.rr")),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn rejected_batch_retracts_trait_impls() {
        use std::sync::Arc;
        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);
            let p1 = parse(
                "pub trait Show { fn show(self: Self) -> i64; }\n\
                 pub struct P { pub x: i64 }",
            );
            elab.try_extend(&surface::program(&p1.root)).expect("decl");
            let impls_before = elab.traits.impls_len();

            // A batch whose impl member body fails is rejected wholesale …
            let p2 = parse("impl Show for P { fn show(self: Self) -> i64 { true } }");
            elab.try_extend(&surface::program(&p2.root))
                .expect_err("bad body");
            assert_eq!(elab.traits.impls_len(), impls_before, "impl retracted");

            // … and the corrected impl registers cleanly (no phantom impl,
            // no phantom method defs).
            let p3 = parse("impl Show for P { fn show(self: Self) -> i64 { self.x } }");
            elab.try_extend(&surface::program(&p3.root)).expect("fixed");
            assert_eq!(elab.traits.impls_len(), impls_before + 1);
        });
    }

    #[test]
    fn user_trait_bounds_resolve_and_imply_through_supertraits() {
        // A user trait resolves as a bound; a user supertrait chain onto a
        // builtin discharges the builtin obligation on the generic.
        check(
            "trait Show { fn norm(self: Self) -> f64; }
             fn f<T: Show>(x: T) -> T { x }",
            |elab, _| {
                let f = elab
                    .functions
                    .values()
                    .find(|p| elab.sym(p.name) == "f")
                    .expect("f");
                let (_, g) = f.generics[0];
                assert_eq!(elab.generic_bounds(g).len(), 1);
            },
        );
        check(
            "trait Croppable: Num { fn crop(self: Self) -> Self; }
             fn double<T: Croppable>(x: T) -> T { x + x }",
            |elab, _| {
                let _ = elab;
            },
        );
    }

    #[test]
    fn parameterized_supertrait_references_are_rejected() {
        // The surface shape carries the arguments; admitting them is the
        // deferred half, so the reference is refused, not silently bared.
        elaborate_source(
            "pub trait A { fn a(self: Self) -> i64; }\n\
             pub trait B : A<i64> { fn b(self: Self) -> i64; }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("parameterized supertrait references are not supported yet")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn supertrait_cycle_is_reported_and_severed() {
        elaborate_source(
            "trait A: B { fn a(self: Self); }
             trait B: A { fn b(self: Self); }
             fn f<T: A>(x: T) -> T { x }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("super-trait cycle")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn trait_redeclaration_diagnostics() {
        elaborate_source(
            "trait A { fn m(self: Self); }
trait A { fn n(self: Self); }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("trait `A` is defined more than once")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        elaborate_source("trait Num { fn m(self: Self); }", |elab| {
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("`Num` is a built-in trait and cannot be redeclared")),
                "{:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn trait_member_rejections() {
        elaborate_source("trait T { fn m(x: i64) -> i64; }", |elab| {
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("a trait method must take `self` as its first parameter")),
                "{:#?}",
                elab.reports
            );
        });
        elaborate_source(
            "trait T { fn m(self: Self); fn m(self: Self) -> i64; }",
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("declares method `m` more than once")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        elaborate_source("trait T<Self> { fn m(self: Self); }", |elab| {
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("`Self` is implicit in a trait")),
                "{:#?}",
                elab.reports
            );
        });
        elaborate_source(
            "trait Convert<T> { fn conv(self: Self) -> T; }
             trait Bad: Convert { fn b(self: Self); }",
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("parameterized bounds are not supported here")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        elaborate_source("trait T<A> { fn m<A: Num>(self: Self, a: A); }", |elab| {
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("shadows the trait's generic")),
                "{:#?}",
                elab.reports
            );
        });
        elaborate_source("trait T { fn m(self: [flex] Self); }", |elab| {
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("requires a `regional fn`")),
                "{:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn trait_method_receiver_forms_recorded() {
        use crate::semi::traits::def::ReceiverForm;
        check(
            "trait Multi {
                 fn v(self: Self) -> i64;
                 fn a(self: Arc<Self>) -> i64;
                 regional fn f(self: [flex] Self);
             }",
            |elab, _| {
                let id = elab
                    .traits
                    .trait_by_def(
                        (0..elab.defs.len() as u32)
                            .map(crate::semi::ty::DefId)
                            .find(|d| {
                                elab.defs.info(*d).kind == crate::semi::resolve::DefKind::Trait
                                    && elab.sym(elab.defs.path(*d).name()) == "Multi"
                            })
                            .expect("Multi def"),
                    )
                    .expect("registered");
                let def = elab.traits.trait_def(id);
                let forms: Vec<ReceiverForm> = def.methods.iter().map(|m| m.receiver).collect();
                assert_eq!(
                    forms,
                    [ReceiverForm::Value, ReceiverForm::Arc, ReceiverForm::Flex]
                );
                assert!(def.methods[2].is_regional);
            },
        );
    }

    #[test]
    fn rejected_batch_retracts_trait_decls() {
        use std::sync::Arc;
        with_tcx(|tcx| {
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let parse = |src: &str| {
                let p = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(p.ok(), "parse errors for {src:?}: {:#?}", p.errors);
                p
            };
            let mut elab = Elaborator::new(tcx, &interner);

            // A batch with an unknown supertrait is rejected wholesale …
            let p1 = parse("trait Broken: Nope { fn m(self: Self); }");
            elab.try_extend(&surface::program(&p1.root))
                .expect_err("unknown supertrait");
            assert!(!elab.has_errors());

            // … and the same trait name declares cleanly next batch: the
            // DefTable entry, the TraitDb stub, and its generics all rolled
            // back.
            let p2 = parse("trait Broken { fn m(self: Self); }");
            elab.try_extend(&surface::program(&p2.root))
                .expect("retracted name is reusable");
        });
    }

    #[test]
    fn impl_members_declare_under_type_path_and_resolve_cross_file() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (
                    &["a"],
                    "pub struct Rect { pub w: i64, h: i64 }\n\
                     impl Rect {\n\
                         pub fn area(r: Rect) -> i64 { r.w * r.h }\n\
                         regional fn noop() { }\n\
                     }",
                ),
                (
                    &["c"],
                    "pub fn use_area(r: super::a::Rect) -> i64 { super::a::Rect::area(r) }",
                ),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
                let area = elab
                    .functions
                    .values()
                    .find(|f| elab.defs.path(f.def).display(elab.resolver) == "p::a::Rect::area")
                    .expect("method declared under the type path");
                assert_eq!(area.visibility, surface::Visibility::Public);
                let noop = elab
                    .functions
                    .values()
                    .find(|f| elab.defs.path(f.def).display(elab.resolver) == "p::a::Rect::noop")
                    .expect("regional member declared");
                assert!(noop.is_regional);
            },
        );
    }

    /// A method body's scope is the type's defining module, so it reads the
    /// target's private fields — the visibility model methods encapsulate.
    #[test]
    fn method_body_accesses_private_field_of_target() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (
                    &["a"],
                    "pub struct Rect { pub w: i64, h: i64 }\n\
                     impl Rect { pub fn height(r: Rect) -> i64 { r.h } }",
                ),
                (
                    &["c"],
                    "pub fn peek(r: super::a::Rect) -> i64 { super::a::Rect::height(r) }",
                ),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn impl_member_clashes_with_module_function_reports() {
        check_package(
            "p",
            &[
                (&[], "mod a;"),
                (
                    &["a"],
                    "mod Point;\npub struct Point { pub x: i64 }\n\
                     impl Point { pub fn scale(p: Point) -> i64 { p.x } }",
                ),
                (&["a", "Point"], "pub fn scale(x: i64) -> i64 { x }"),
            ],
            |elab, _| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains(
                        "method `scale` conflicts with an existing function `Point::scale`"
                    )),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// Rust-style package-level impls: a block may appear in any module of
    /// the defining package; its members declare under the type's path and
    /// dispatch from anywhere.
    #[test]
    fn impl_allowed_across_modules_in_package() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (
                    &["a"],
                    "mod sub;
pub struct Rect { pub w: i64 }",
                ),
                (
                    &["a", "sub"],
                    "impl super::Rect { pub fn from_child(r: super::Rect) -> i64 { r.w } }",
                ),
                (
                    &["c"],
                    "impl super::a::Rect { pub fn twice(r: super::a::Rect) -> i64 { r.w * 2 } }
                     pub fn go(r: super::a::Rect) -> i64 {
                         super::a::Rect::from_child(r) + super::a::Rect::twice(r)
                     }",
                ),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    /// An out-of-module impl sees exactly what its module sees: the target's
    /// private fields stay private (Rust's rule — placement grants nothing).
    #[test]
    fn out_of_module_impl_cannot_read_private_fields() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (&["a"], "pub struct Rect { pub w: i64, h: i64 }"),
                (
                    &["c"],
                    "impl super::a::Rect { pub fn peek(r: super::a::Rect) -> i64 { r.h } }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message == "field `h` of record `p::a::Rect` is private"),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// Coherence is by declared path, so two impls in different modules
    /// declaring the same member still clash deterministically.
    #[test]
    fn cross_module_impl_clash_reports() {
        check_package(
            "p",
            &[
                (&[], "mod a; mod c;"),
                (
                    &["a"],
                    "pub struct Rect { pub w: i64 }
                     impl Rect { pub fn area(r: Rect) -> i64 { r.w } }",
                ),
                (
                    &["c"],
                    "impl super::a::Rect { pub fn area(r: super::a::Rect) -> i64 { 0 } }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains(
                        "method `area` conflicts with an existing function `Rect::area`"
                    )),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn pub_impl_block_rejected() {
        with_tcx(|tcx| {
            let source = "struct P { x: i64 }\npub impl P { fn get(p: P) -> i64 { p.x } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports.iter().any(|r| r
                    .message
                    .contains("`pub` is not allowed on an `impl` block")),
                "{:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn impl_generics_concatenate_with_method_generics() {
        check(
            "pub struct Box<T> { pub v: T }\n\
             impl<T: Num> Box<T> {\n\
                 pub fn paired<U: Num>(b: Box<T>, u: U) -> U { u + u }\n\
             }\n\
             fn use_it(b: Box<i32>) -> i64 { Box::paired(b, 2) }",
            |elab, _| {
                let paired = elab
                    .functions
                    .values()
                    .find(|f| elab.sym(f.name) == "paired")
                    .expect("method declared");
                // Impl generics precede the method's own in the binder.
                assert_eq!(paired.generics.len(), 2);
            },
        );
    }

    #[test]
    fn method_generic_shadowing_impl_generic_reports() {
        with_tcx(|tcx| {
            let source = "pub struct Box<T> { pub v: T }\n\
                          impl<T: Num> Box<T> { fn bad<T: Num>(b: Box<T>) -> i64 { 0 } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("shadows the impl block's generic")),
                "{:#?}",
                elab.reports
            );
        });
    }

    /// The shared two-module package for the visibility tests: module `a`
    /// defines the records, module `c` uses them from outside.
    fn vis_package(consumer: &str, f: impl Fn(&[crate::semi::ctxt::Report])) {
        let defs = r#"
            pub struct S { pub a: i64, b: i64 }
            pub struct P(pub i64, i64)
            pub struct [regional] R { hidden: [field] R }
        "#;
        check_package(
            "p",
            &[(&[], "mod a; mod c;"), (&["a"], defs), (&["c"], consumer)],
            |elab, _| f(&elab.reports),
        );
    }

    #[test]
    fn private_field_access_rejected_across_modules() {
        vis_package("pub fn read(s: super::a::S) -> i64 { s.b }", |reports| {
            assert!(
                reports
                    .iter()
                    .any(|r| r.message == "field `b` of record `p::a::S` is private"),
                "{reports:#?}"
            );
        });
        // Assignment shares the resolver: the private diagnostic fires before
        // mutability or source checking.
        vis_package(
            "regional fn poke(r: [flex] super::a::R) { r->hidden := 1 }",
            |reports| {
                assert!(
                    reports
                        .iter()
                        .any(|r| r.message == "field `hidden` of record `p::a::R` is private"),
                    "{reports:#?}"
                );
            },
        );
    }

    #[test]
    fn pub_field_access_allowed_across_modules() {
        vis_package(
            "pub fn read(s: super::a::S, t: super::a::P) -> i64 { s.a + t.0 }",
            |reports| assert!(reports.is_empty(), "{reports:#?}"),
        );
    }

    #[test]
    fn private_field_access_allowed_in_child_module() {
        check_package(
            "p",
            &[
                (&[], "mod a;"),
                (&["a"], "mod sub;\npub struct S { pub a: i64, b: i64 }"),
                (
                    &["a", "sub"],
                    "pub fn read(s: super::S) -> i64 { s.b }\n\
                     pub fn make() -> super::S { super::S { a: 1, b: 2 } }",
                ),
            ],
            |elab, _| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
    }

    #[test]
    fn private_field_ctor_rejected_across_modules() {
        vis_package(
            "pub fn make() -> super::a::S { super::a::S { a: 1, b: 2 } }",
            |reports| {
                assert!(
                    reports.iter().any(|r| r.message
                        == "cannot construct `p::a::S` outside its module: field(s) `b` are private"),
                    "{reports:#?}"
                );
            },
        );
        vis_package(
            "pub fn make() -> super::a::P { super::a::P { 1, 2 } }",
            |reports| {
                assert!(
                    reports.iter().any(|r| r.message
                        == "cannot construct `p::a::P` outside its module: field(s) `1` are private"),
                    "{reports:#?}"
                );
            },
        );
    }

    #[test]
    fn tuple_index_respects_named_field_visibility() {
        // `s.1` into a named record resolves the named field, so the private
        // diagnostic names it.
        vis_package("pub fn read(s: super::a::S) -> i64 { s.1 }", |reports| {
            assert!(
                reports
                    .iter()
                    .any(|r| r.message == "field `b` of record `p::a::S` is private"),
                "{reports:#?}"
            );
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
                // A one-statement block collapses to the statement itself, so the
                // body is the `match` directly (not a single-element `Seq`).
                assert!(matches!(&body.kind, hir::ExprKind::Match(..)));
            },
        );
    }

    #[test]
    fn colors_regional_struct_fields() {
        check(
            "struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }\n\
             regional fn make<T>(x: T) -> [flex] TestCell<T> { TestCell { v: x, next: Nullable::Null } }",
            |elab, _tcx| {
                let f = function(elab, "make");
                // The return type was colored Flex by the `[flex]` annotation.
                let crate::semi::ty::TyKind::Record { flex, .. } = f.return_ty.kind() else {
                    panic!("expected a record return type, got {:?}", f.return_ty);
                };
                assert_eq!(*flex, Flexivity::Flex);
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
            "struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }\n\
             regional fn build(seed: i32) -> i32 {\n\
                 let c = TestCell { v: seed, next: Nullable::Null };\n\
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
                    value.ty.flexivity(),
                    Some(Flexivity::Flex),
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
                assert_eq!(value.ty.flexivity(), Some(Flexivity::Irrelevant));
            },
        );
    }

    #[test]
    fn reports_type_mismatch() {
        with_tcx(|tcx| {
            let source = "fn bad() -> bool { 1 }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(elab.has_errors(), "expected a type mismatch error");
        });
    }

    /// An `if` with no `else` is unit, so a non-unit then-branch is a type
    /// error reported at the branch itself.
    #[test]
    fn if_without_else_requires_a_unit_then_branch() {
        with_tcx(|tcx| {
            let source = "fn bad(c: bool) -> i32 { if c { true } }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(elab.has_errors(), "expected a unit mismatch error");
            let messages = elab
                .reports
                .iter()
                .map(|r| r.message.clone())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                messages.contains("expected `unit`, found `bool`"),
                "then-branch should be checked against unit: {messages}"
            );
        });
    }

    /// Diagnostics spell types the way the surface does (`i32`, `bool`,
    /// `Nullable<TestCell<u64>>`, `[rigid] …`), never the internal `Debug` form
    /// (`Int(Signed(32))`).
    #[test]
    fn diagnostics_render_types_in_surface_syntax() {
        let messages = |source: &str| {
            with_tcx(|tcx| {
                let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
                let parse = reussir_syntax::parse_with_interner(source, interner.clone());
                assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
                let prog = surface::program(&parse.root);
                let elab = elaborate(tcx, &prog, &interner);
                assert!(elab.has_errors(), "expected an error");
                elab.reports
                    .iter()
                    .map(|r| r.message.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        };

        let m = messages("fn bad() -> i32 { true }");
        assert!(
            m.contains("expected `i32`, found `bool`"),
            "surface spelling expected: {m}"
        );

        let m = messages(
            r#"
            struct [regional] TestCell<T> { v: T }
            fn f(c: TestCell<u64>) -> bool { c }
            "#,
        );
        assert!(
            m.contains("expected `bool`, found `[rigid] TestCell<u64>`"),
            "record spelling expected: {m}"
        );

        let m = messages(
            r#"
            struct [regional] Matrix<T> { m00: [field] T }
            fn test() -> Matrix<u64> { regional { Matrix { m00: 0 } } }
            "#,
        );
        assert!(
            m.contains("`Nullable<u64>` does not implement `Integral`"),
            "nullable spelling expected: {m}"
        );
        assert!(
            !m.contains("Int(") && !m.contains("Unsigned("),
            "no Debug leakage: {m}"
        );
    }

    /// A non-`[field]` regional-record member holds an already-*frozen* value:
    /// the member's expected type is `Rigid`-refined (`rigid_member_ty`), so a
    /// projection reads a `Rigid` value out, and construction accepts a frozen
    /// argument.
    #[test]
    fn plain_regional_member_is_rigid_on_both_sides() {
        check(
            r#"
            struct [regional] Inner { v: u64 }
            struct [regional] Outer { inner: Inner, tag: u64 }

            fn mk_inner(v: u64) -> Inner {
                regional { Inner { v: v } }
            }

            fn take(v: u64) -> Inner {
                let o = regional { Outer { inner: mk_inner(v), tag: v } };
                o.inner
            }
            "#,
            |elab, _| {
                let body = function(elab, "take").body.as_ref().unwrap();
                // The projected member is a frozen view.
                assert_eq!(body.ty.flexivity(), Some(Flexivity::Rigid));
            },
        );
    }

    /// A nullable match dispatches on the two built-in constructors; the
    /// `NonNull` payload binder gets the element type, and the whole match
    /// unifies as usual.
    #[test]
    fn nullable_patterns_type_and_dispatch() {
        check(
            r#"
            struct RcBox<T> { value: T }

            fn unwrap_or(n: Nullable<RcBox<i32>>, d: i32) -> i32 {
                match n {
                    Nullable::NonNull(b) => b.value,
                    Nullable::Null => d
                }
            }
            "#,
            |elab, tcx| {
                let body = function(elab, "unwrap_or").body.as_ref().unwrap();
                assert_eq!(body.ty, tcx.mk_int(IntTy::Signed(32)));
                // The match compiled to a nullable switch.
                let hir::ExprKind::Match(_, tree) = &body.kind else {
                    panic!("expected a match body, got {:?}", body.kind);
                };
                let hir::DecisionTree::Switch { cases, .. } = tree else {
                    panic!("expected a switch, got {tree:?}");
                };
                assert!(
                    matches!(cases, hir::SwitchCases::Nullable { .. }),
                    "expected a nullable switch"
                );
            },
        );
    }

    /// Omitting the `Null` arm is a non-exhaustive match, like any other
    /// missing case of a closed family.
    #[test]
    fn nullable_match_missing_null_arm_is_non_exhaustive() {
        with_tcx(|tcx| {
            let source = r#"
                struct RcBox<T> { value: T }

                fn oops(n: Nullable<RcBox<i32>>) -> i32 {
                    match n {
                        Nullable::NonNull(b) => b.value
                    }
                }
            "#;
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("non-exhaustive")),
                "expected a non-exhaustive diagnostic: {:#?}",
                elab.reports
            );
        });
    }

    /// A `[field]` link projects at the *base's* view: `Flex` (writable)
    /// through a flex base, `Rigid` (a frozen view) through a rigid base —
    /// after the region freezes, `x.f` must not hand out a mutable coloring
    /// (mirrors `getProjectedType`, where only a flex reference projects a
    /// flex link).
    #[test]
    fn field_link_takes_the_bases_view() {
        check(
            r#"
            struct [regional] TestCell { v: u64, next: [field] TestCell }

            regional fn mk(v: u64) -> [flex] TestCell {
                TestCell { v: v, next: Nullable::Null }
            }

            fn frozen_view(v: u64) -> Nullable<TestCell> {
                let c = regional { mk(v) };
                c.next
            }

            regional fn flex_view(c: [flex] TestCell) -> [flex] Nullable<TestCell> {
                c.next
            }
            "#,
            |elab, _| {
                let link_flex = |name: &str| {
                    let body = function(elab, name).body.as_ref().unwrap();
                    let TyKind::Nullable(inner) = body.ty.kind() else {
                        panic!("{name}: expected a nullable link, got {:?}", body.ty);
                    };
                    inner.flexivity()
                };
                assert_eq!(link_flex("frozen_view"), Some(Flexivity::Rigid));
                assert_eq!(link_flex("flex_view"), Some(Flexivity::Flex));
            },
        );
    }

    /// Storing a still-live `Flex` value into a plain (non-`[field]`) regional
    /// member is a flexivity mismatch at elaboration — the member requires a
    /// frozen value (the backend types the slot `rc<_, rigid>`); previously
    /// this leaked through and died as an MLIR verifier error.
    #[test]
    fn rejects_flex_value_in_plain_regional_member() {
        with_tcx(|tcx| {
            let source = r#"
                struct [regional] Inner { v: u64 }
                struct [regional] Outer { inner: Inner, tag: u64 }

                regional fn mk(v: u64) -> [flex] Outer {
                    Outer { inner: Inner { v: v }, tag: v }
                }
            "#;
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("flexivity mismatch")),
                "expected a flexivity mismatch: {:#?}",
                elab.reports
            );
        });
    }

    #[test]
    fn rejects_regional_call_outside_region() {
        with_tcx(|tcx| {
            let source = "regional fn r() -> i32 { 0 }\nfn caller() -> i32 { r() }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
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
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
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
            let source = "struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }\n\
                          regional fn f(c: [flex] TestCell<i32>) -> i32 { let g = || c; 0 }";
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("flex value cannot escape")),
                "expected a flex-capture error: {:#?}",
                elab.reports
            );
        });
    }

    // ----- import -----

    /// Elaborate and print the HIR (no locations), to compare an abbreviated
    /// program against its fully-qualified spelling.
    fn printed_hir(source: &str) -> String {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let strings = elab.strings.entries();
            crate::semi::hir::print::Printer::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            )
        })
    }

    #[test]
    fn imports_abbreviate_intrinsics_like_the_full_spelling() {
        let full = printed_hir(
            "fn f(x: f64) -> f64 { core::intrinsic::math::sqrt(x, 0) }\n\
             fn g(a: [f64; 4], i: i64) -> f64 { core::intrinsic::array::get(a, i) }\n\
             fn h(x: f64) -> f64 { core::intrinsic::math::sqrt(x, 0) }",
        );
        // Both binding spellings: `import` (last segment) and `import … as …`,
        // the latter on a whole module and on a single intrinsic function.
        let abbreviated = printed_hir(
            "import core::intrinsic::math;\n\
             import core::intrinsic::array as arr;\n\
             import core::intrinsic::math::sqrt as rt;\n\
             fn f(x: f64) -> f64 { math::sqrt(x, 0) }\n\
             fn g(a: [f64; 4], i: i64) -> f64 { arr::get(a, i) }\n\
             fn h(x: f64) -> f64 { rt(x, 0) }",
        );
        assert_eq!(full, abbreviated);
    }

    #[test]
    fn imports_abbreviate_records_in_every_position() {
        let full = printed_hir(
            "enum List<T> { Nil, Cons(T, List<T>) }\n\
             fn len(l: List<i64>) -> i64 { match l { List::Nil => 0, List::Cons(x, xs) => 1 + len(xs) } }\n\
             fn make() -> List<i64> { List::Cons{1, List::Nil} }",
        );
        // A record import covers type annotations, constructors (incl.
        // nullary), and pattern qualifiers; `LL` also exercises a binding
        // that targets another binding.
        let abbreviated = printed_hir(
            "enum List<T> { Nil, Cons(T, List<T>) }\n\
             import List as L;\n\
             import L as LL;\n\
             fn len(l: L<i64>) -> i64 { match l { LL::Nil => 0, L::Cons(x, xs) => 1 + len(xs) } }\n\
             fn make() -> LL<i64> { L::Cons{1, LL::Nil} }",
        );
        assert_eq!(full, abbreviated);
    }

    #[test]
    fn locals_shadow_imports() {
        // The imported `sqrt` takes two arguments; the local closure takes
        // one. Elaborating without errors proves the local won.
        check(
            "import core::intrinsic::math::sqrt;\n\
             fn f(x: f64) -> f64 { let sqrt = |y: f64| y; sqrt(x) }",
            |_, _| {},
        );
    }

    #[test]
    fn import_bindings_are_file_scoped() {
        check_package(
            "p",
            &[
                (
                    &[],
                    "mod m;\nimport m::helper as h;\npub fn go() -> u64 { h(1) }",
                ),
                (&["m"], "pub fn helper(x: u64) -> u64 { x }"),
            ],
            |elab, _| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            },
        );
        // The binding must not leak into a sibling file.
        check_package(
            "p",
            &[
                (&[], "mod m;\nimport m::helper as h;"),
                (
                    &["m"],
                    "pub fn helper(x: u64) -> u64 { x }\npub fn go() -> u64 { h(1) }",
                ),
            ],
            |elab, _| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("unknown function `h`")),
                    "expected the sibling file to miss the binding: {:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn rejects_invalid_imports() {
        for (src, expected) in [
            ("pub import core::intrinsic::math;", "cannot be `pub`"),
            (
                "import core::intrinsic::math;\nimport core::intrinsic::array as math;",
                "already bound",
            ),
            ("import foo as core;", "reserved path head"),
        ] {
            assert!(has_error(src, expected), "{src:?}: {:#?}", reports_of(src));
        }
        // A self-referential binding chain must fail resolution, not hang.
        assert!(has_error(
            "import b::x as a;\nimport a::y as b;\nfn f() -> i64 { a::z(1) }",
            "unknown function",
        ));
    }

    /// Elaborate a source string and return its diagnostics (for negative tests).
    fn reports_of(source: &str) -> Vec<crate::semi::Report> {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            elaborate(tcx, &prog, &interner).reports.clone()
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

    // ----- arc-shared-inner -----

    #[test]
    fn rejects_arc_of_scalar() {
        let src = "fn f(x: Arc<i32>) -> i32 { 0 }";
        assert!(
            has_error(
                src,
                "is not a `[shared]` record, array, or closure (not an rc box)"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_of_cell() {
        // A cell is a shared box, but synchronization is the cell's own axis;
        // `Arc` must not stack a second discipline on top of it.
        let src = "fn f(x: Arc<Cell<i32>>) -> i32 { 0 }";
        assert!(
            has_error(
                src,
                "is not a `[shared]` record, array, or closure (a cell)"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_of_arc() {
        let src = "struct Pair { a: i32 }\nfn f(x: Arc<Arc<Pair>>) -> i32 { 0 }";
        assert!(
            has_error(
                src,
                "is not a `[shared]` record, array, or closure (already an `Arc`)"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_of_value_record() {
        let src = "struct [value] Pair { a: i32 }\nfn f(x: Arc<Pair>) -> i32 { 0 }";
        assert!(
            has_error(
                src,
                "is not a `[shared]` record, array, or closure (a `[value]` record)"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_of_regional_record() {
        let src = "struct [regional] C { v: i32, next: [field] C }\n\
                   fn f(x: Arc<C>) -> i32 { 0 }";
        assert!(
            has_error(
                src,
                "is not a `[shared]` record, array, or closure (a `[regional]` record)"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_arity() {
        let src = "struct Pair { a: i32 }\nfn f(x: Arc<Pair, Pair>) -> i32 { 0 }";
        assert!(
            has_error(src, "`Arc` takes exactly one type argument"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn arc_record_members_are_allowed() {
        // An `Arc` member spells its own atomic link (§3.3) — legal in
        // shared and `[value]` records; a `[regional]` record rejects it for
        // now (not implemented — the §6 sync-regional design will decide).
        let ok = "struct Pair { a: i32 }\n\
                  struct Holder { p: Arc<Pair> }\n\
                  struct [value] VHolder { p: Arc<Pair> }";
        assert!(reports_of(ok).is_empty(), "{:#?}", reports_of(ok));
        let regional = "struct Pair { a: i32 }\n\
                        struct [regional] R { v: i64, p: Arc<Pair> }";
        assert!(
            has_error(
                regional,
                "an `Arc` member of a `[regional]` record is not implemented yet"
            ),
            "{:#?}",
            reports_of(regional)
        );
    }

    #[test]
    fn rejects_infinite_value_recursion() {
        // A `[value]` record stored inline within itself has no finite
        // layout: mutual and self cycles are rejected at declaration.
        let mutual = "struct [value] A { b: B }\nstruct [value] B { a: A }";
        assert!(
            has_error(mutual, "recursive `[value]` record has infinite size"),
            "{:#?}",
            reports_of(mutual)
        );
        let direct = "struct [value] A { a: A }";
        assert!(
            has_error(direct, "recursive `[value]` record has infinite size"),
            "{:#?}",
            reports_of(direct)
        );
    }

    #[test]
    fn boxed_links_break_value_recursion() {
        // Any pointer member breaks the inline chain: a shared box makes
        // the recursion finite. (A shared record holding a value chain is
        // fine too — edges into a shared record are pointers, so it can
        // never sit on an inline cycle.)
        let src = "struct S { a: A }\n\
                   struct [value] A { s: S }\n\
                   struct Shared2 { v: i64 }\n\
                   struct [value] C { s: Shared2 }\n\
                   struct [value] D { c: C }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn sync_cell_surface_types() {
        // The four sync kinds are surface type constructors with per-kind
        // element bounds. Well-formed shapes elaborate cleanly, including
        // the composition direction: an Arc'd record with a sync-cell
        // member is interior mutability behind an arc.
        let ok = "struct Pair { a: i32 }\n\
                  struct Counter { hits: Atomic<i64>, guard: Mutex<i64> }\n\
                  fn f(m: Mutex<i64>, a: Atomic<f64>, l: FlatLock<i64>, r: RwLock<Arc<Pair>>) -> i32 { 0 }\n\
                  fn g(c: Arc<Counter>) -> i32 { 0 }";
        assert!(reports_of(ok).is_empty(), "{:#?}", reports_of(ok));
    }

    #[test]
    fn atomic_rmw_is_gated() {
        // The atomic CAS-retry region must be effect-free, which a closure
        // call is not — rmw on an Atomic cell is rejected at the checker
        // until dedicated atomic arithmetic intrinsics exist.
        let src = "fn f(a: Atomic<i64>) -> i64 {\n\
                       core::intrinsic::cell::rmw(a, |x| x + 1);\n\
                       core::intrinsic::cell::get(a)\n\
                   }";
        assert!(
            has_error(
                src,
                "`core::intrinsic::cell::rmw` is not supported on `Atomic<i64>`"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn sync_cell_element_bounds() {
        // Atomic demands an arithmetic primitive …
        let bad_atomic = "struct Pair { a: i32 }\nfn f(a: Atomic<Pair>) -> i32 { 0 }";
        assert!(
            has_error(bad_atomic, "`Atomic<Pair>` is ill-formed"),
            "{:#?}",
            reports_of(bad_atomic)
        );
        // … the lock kinds demand a Sync element (Mutex<Arc> yes,
        // Mutex<bare shared> no) …
        let bad_mutex = "struct Pair { a: i32 }\nfn f(m: Mutex<Pair>) -> i32 { 0 }";
        assert!(
            has_error(
                bad_mutex,
                "the element of a lock-guarded cell must be `Sync`"
            ),
            "{:#?}",
            reports_of(bad_mutex)
        );
        // … and a pointer-or-primitive slot shape.
        let bad_slot = "struct [value] V { a: i32 }\nfn f(r: RwLock<V>) -> i32 { 0 }";
        assert!(
            has_error(
                bad_slot,
                "`RwLock<V>` is ill-formed: the element is a `[value]` record"
            ),
            "{:#?}",
            reports_of(bad_slot)
        );
    }

    #[test]
    fn infers_arc_ctors() {
        // The struct and variant arc constructors type at `Arc<R>`.
        let src = "struct Pair { a: i32 }\n\
                   enum Opt<T> { None, Some(T) }\n\
                   fn f(v: i32) -> Arc<Pair> { Arc<Pair> { a: v } }\n\
                   fn g() -> Arc<Opt<i32>> { Arc<Opt<i32>>::Some{1} }\n\
                   fn h() -> Arc<Opt<i32>> { Arc<Opt<i32>>::None }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn rejects_arc_whose_member_is_a_bare_shared_record() {
        // The stratified wf rule: the arc colors exactly one box, so a bare
        // `[shared]` member *outside the recursive group* refutes — it is
        // not part of the spine the coloring floods, so it must already be
        // `Sync` (an `Arc` member) on its own.
        let src = "struct Q { x: i64 }\n\
                   struct S { p: Q }\n\
                   fn f(s: Arc<S>) -> i32 { 0 }";
        assert!(
            has_error(src, "every member of an `Arc` inner must be `Sync`"),
            "{:#?}",
            reports_of(src)
        );
        assert!(
            has_error(src, "member `p` is a plain `[shared]` rc box"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn arc_of_recursive_enum_is_wf_via_scc_promotion() {
        // §3.3: Arc<List> selects the atomic instantiation of List's whole
        // recursive group — the Cons tail promotes to Arc<List<T>>
        // automatically, and the coinductive rule closes the cycle. Mutual
        // recursion promotes the same way.
        let src = "enum List<T> { Nil, Cons(T, List<T>) }\n\
                   enum Tree { Leaf(i64), Node(Forest) }\n\
                   enum Forest { None, More(Tree, Forest) }\n\
                   fn f(l: Arc<List<i32>>) -> i32 { 0 }\n\
                   fn g(t: Arc<Tree>) -> i32 { 0 }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn arc_coloring_mismatch_explains_itself() {
        // A mismatch that is *only* the arc coloring gets the reconciliation
        // diagnostic instead of a bare "type mismatch" — on both sides of
        // the confusion. Creation side: feeding a bare value where the
        // atomic world demands an arc'd one names the SCC promotion and
        // says how to fix it …
        let creation = "enum List<T> { Nil, Cons(T, List<T>) }\n\
                        fn f(x: i32) -> i32 { Arc<List<i32>>::Cons{x, List::Nil}; 0 }";
        assert!(
            has_error(
                creation,
                "cannot reconcile thread-safety: expected `Arc<List<i32>>`, found `List<i32>`"
            ),
            "{:#?}",
            reports_of(creation)
        );
        assert!(
            has_error(
                creation,
                "fields of the recursive group `List` are promoted to `Arc`"
            ),
            "{:#?}",
            reports_of(creation)
        );
        assert!(
            has_error(
                creation,
                "construct it as `Arc<List<i32>>::…` from the start"
            ),
            "{:#?}",
            reports_of(creation)
        );
        // … and projection side: a field matched out of an arc'd box keeps
        // the coloring, which the diagnostic explains rather than leaving
        // the user wondering where the `Arc` came from.
        let projection = "enum List<T> { Nil, Cons(T, List<T>) }\n\
                          fn tail(l: Arc<List<i32>>) -> List<i32> {\n\
                              match l { List::Cons(_, xs) => xs, List::Nil => List::Nil }\n\
                          }";
        assert!(
            has_error(
                projection,
                "a value read out of an arc'd box keeps its `Arc` coloring"
            ),
            "{:#?}",
            reports_of(projection)
        );
        // A non-recursive pair still reconciles, without the SCC note.
        let flat = "struct Pair { a: i32 }\n\
                    fn f(p: Pair) -> Arc<Pair> { p }";
        assert!(
            has_error(
                flat,
                "cannot reconcile thread-safety: expected `Arc<Pair>`, found `Pair`"
            ),
            "{:#?}",
            reports_of(flat)
        );
        assert!(
            !has_error(flat, "recursive group"),
            "{:#?}",
            reports_of(flat)
        );
    }

    #[test]
    fn arc_mixes_explicit_and_promoted_members() {
        // A shared record can mix both member forms (§3.3): an explicit
        // `Arc<Inner>` member keeps its declared type everywhere — readable
        // from a *bare* box too, no arc context needed — while the bare
        // recursive children promote only under the arc'd box. The arc'd
        // ctor demands the explicit member as declared and the recursive
        // ones promoted.
        let src = "struct Inner { v: i64 }\n\
                   enum Tree { Leaf, Node(Arc<Inner>, Tree, Tree) }\n\
                   fn node(h: Arc<Inner>, l: Arc<Tree>, r: Arc<Tree>) -> Arc<Tree> {\n\
                       Arc<Tree>::Node{h, l, r}\n\
                   }\n\
                   fn header(t: Arc<Tree>) -> Arc<Inner> {\n\
                       match t {\n\
                           Tree::Node(h, _, _) => h,\n\
                           Tree::Leaf => Arc<Inner> { v: 0 }\n\
                       }\n\
                   }\n\
                   fn left(t: Arc<Tree>) -> Arc<Tree> {\n\
                       match t {\n\
                           Tree::Node(_, l, _) => l,\n\
                           Tree::Leaf => Arc<Tree>::Leaf\n\
                       }\n\
                   }\n\
                   fn bare_header(t: Tree) -> Arc<Inner> {\n\
                       match t {\n\
                           Tree::Node(h, _, _) => h,\n\
                           Tree::Leaf => Arc<Inner> { v: 0 }\n\
                       }\n\
                   }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn value_intermediate_requires_explicit_arc_member() {
        // A `[value]` record in the recursive group folds *outside* the
        // promotion group: loaded out of the box by value, nothing carries
        // its coloring, so a bare same-group link inside it refutes …
        let bad = "struct [value] V { l: List }\n\
                   enum List { Nil, Cons(V) }\n\
                   fn f(l: Arc<List>) -> i32 { 0 }";
        assert!(
            has_error(bad, "every member of an `Arc` inner must be `Sync`"),
            "{:#?}",
            reports_of(bad)
        );
        // … and the expressible alternative is an explicit `Arc` member,
        // which is `Sync` on its own (closed coinductively through the
        // enclosing arc).
        let good = "struct [value] W { l: Arc<List> }\n\
                    enum List { Nil, Cons(W) }\n\
                    fn f(l: Arc<List>) -> i32 { 0 }";
        assert!(reports_of(good).is_empty(), "{:#?}", reports_of(good));
    }

    #[test]
    fn arc_ctor_and_match_use_promoted_field_types() {
        // The closed colored world: the arc'd Cons demands an arc'd tail,
        // and matching through the arc binds the tail at its promoted type.
        let ok = "enum List<T> { Nil, Cons(T, List<T>) }\n\
                  fn cons(x: i32, xs: Arc<List<i32>>) -> Arc<List<i32>> {\n\
                      Arc<List<i32>>::Cons{x, xs}\n\
                  }\n\
                  fn tail(l: Arc<List<i32>>) -> Arc<List<i32>> {\n\
                      match l {\n\
                          List::Cons(_, xs) => xs,\n\
                          List::Nil => Arc<List<i32>>::Nil\n\
                      }\n\
                  }";
        assert!(reports_of(ok).is_empty(), "{:#?}", reports_of(ok));
        // A bare tail cannot enter the atomic world implicitly.
        let bad = "enum List<T> { Nil, Cons(T, List<T>) }\n\
                   fn f(x: i32) -> i32 { Arc<List<i32>>::Cons{x, List::Nil}; 0 }";
        assert!(!reports_of(bad).is_empty(), "{:#?}", reports_of(bad));
    }

    #[test]
    fn rejects_arc_wf_at_the_ctor_site() {
        // The coloring site checks too: no annotation is needed to trip it.
        let src = "struct Q { x: i64 }\n\
                   struct S { p: Q }\n\
                   fn f() -> i32 { Arc<S> { p: Q { x: 1 } }; 0 }";
        assert!(
            has_error(src, "every member of an `Arc` inner must be `Sync`"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn later_batch_arc_wf_reports_once() {
        // A second `run` batch (the REPL shape) scans its signatures while
        // the flag left by the previous batch's completed sweep would still
        // be set; the annotation-site check must stay silent until this
        // batch's own sweep, or the error doubles.
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let mut elab = Elaborator::new(tcx, &interner);
            for src in [
                "struct Q { x: i64 }\nstruct S { p: Q }",
                "fn f(s: Arc<S>) -> i32 { 0 }",
            ] {
                let parse = reussir_syntax::parse_with_interner(src, interner.clone());
                assert!(parse.ok(), "{src}: {:#?}", parse.errors);
                elab.run(&surface::program(&parse.root));
            }
            let arc_errors = elab
                .reports
                .iter()
                .filter(|r| {
                    r.message
                        .contains("every member of an `Arc` inner must be `Sync`")
                })
                .count();
            assert_eq!(arc_errors, 1, "{:#?}", elab.reports);
        });
    }

    #[test]
    fn arc_wf_witness_names_a_nested_member_path() {
        // The `!Sync` leaf sits two levels down, through a `[value]` record;
        // the diagnostic walks the path instead of blaming the root.
        let src = "struct Leaf { x: i32 }\n\
                   struct [value] Mid { leaf: Leaf }\n\
                   struct Top { m: Mid }\n\
                   fn f(t: Arc<Top>) -> i32 { 0 }";
        assert!(
            has_error(src, "member `m.leaf` is a plain `[shared]` rc box"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_whose_member_is_a_cell() {
        let src = "struct Counter { c: Cell<i32> }\n\
                   fn f(x: Arc<Counter>) -> i32 { 0 }";
        assert!(
            has_error(src, "member `c` is a plain `Cell`"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn arc_ctor_is_not_the_plain_record_type() {
        // The arc coloring is part of the type: an arc'd constructor does not
        // check against the bare record.
        let src = "struct Pair { a: i32 }\nfn f() -> Pair { Arc<Pair> { a: 1 } }";
        assert!(!reports_of(src).is_empty(), "expected a type mismatch");
    }

    #[test]
    fn rejects_arc_ctor_of_non_shared_record() {
        let src = "struct [value] Pair { a: i32 }\nfn f() -> i32 { Arc<Pair> { a: 1 }; 0 }";
        assert!(
            has_error(
                src,
                "`Arc` constructs a `[shared]` record; `Pair` is not one"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_arc_ctor_without_inner_type() {
        let src = "struct Pair { a: i32 }\nfn f() -> i32 { Arc { a: 1 }; 0 }";
        assert!(
            has_error(
                src,
                "`Arc` construction takes exactly one explicit type argument"
            ),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn arc_reads_are_transparent() {
        // Projection and matching see through the arc coloring; the bound
        // fields carry their declared (plain) types.
        let src = "struct Pair { a: i32, b: i32 }\n\
                   enum Opt<T> { None, Some(T) }\n\
                   fn f(p: Arc<Pair>) -> i32 { p.a + p.b }\n\
                   fn g(o: Arc<Opt<i32>>) -> i32 {\n\
                       match o { Opt::Some(x) => x, Opt::None => 0 }\n\
                   }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    // There is no write-through-arc test because the interaction cannot be
    // formed: `Arc` cannot be applied to regional objects (a `[regional]`
    // inner is rejected), and `[field]` mutable links exist only on regional
    // objects — so `Arc` and `[field]` never meet. Interior mutability behind
    // an arc composes via `Cell` instead.

    #[test]
    fn accepts_arc_of_shared_record_and_generic() {
        // A `[shared]` record inner is the intended case; a generic inner is
        // deferred to the instantiation check.
        let src = "struct Pair { a: i32 }\n\
                   fn f(x: Arc<Pair>) -> i32 { 0 }\n\
                   fn g<T>(x: Arc<T>) -> i32 { 0 }\n\
                   fn h(x: Nullable<Arc<Pair>>) -> i32 { 0 }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn accepts_arc_of_array_and_closure() {
        // Arrays and closures are single shared rc boxes like a `[shared]`
        // record, so they take the atomic coloring the same way.
        let src = "fn f(x: Arc<[f64; 8]>) -> i32 { 0 }\n\
                   fn g(x: Arc<(i64) -> i64>) -> i32 { 0 }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
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
        let src = "struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }\n\
                   regional fn f(c: [flex] TestCell<i32>) -> i32 { let g = || c; 0 }";
        assert!(
            has_error(src, "closure cannot return a flex value"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_non_regional_function_returning_flex() {
        // The `frozen()` shape: a plain function cannot return a flex value — the
        // only way its body could produce one is a region-run, which freezes to
        // `rigid` on exit, so a `[flex]` return can never be satisfied.
        let src = "struct [regional] A { v: i32, next: [field] A }\n\
                   fn frozen(x: i32) -> [flex] A { regional { A { v: x, next: Nullable::Null } } }";
        assert!(
            has_error(src, "non-regional function cannot return a flex value"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_non_regional_function_taking_flex_param() {
        // A flex value cannot exist outside a region, so it cannot be handed to a
        // plain function as an argument.
        let src = "struct [regional] A { v: i32, next: [field] A }\n\
                   fn f(a: [flex] A) -> i32 { a.v }";
        assert!(
            has_error(src, "non-regional function cannot take flex parameter"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn accepts_regional_function_with_flex_signature() {
        // A `regional fn` shares its caller's region, so flex params and a flex
        // return are legal there — the boundary check must not over-reject them.
        let src = "struct [regional] A { v: i32, next: [field] A }\n\
                   regional fn thread(a: [flex] A) -> [flex] A { a }";
        assert!(!has_error(src, "cannot"), "{:#?}", reports_of(src));
    }

    #[test]
    fn colors_field_link_null_flex() {
        // `Nullable::Null` stored into a `[field]` slot of a flex record must be
        // colored `flex` (a freshly-built region-local value), not left at the
        // record's default `Regional` — otherwise it reaches the backend uncolored.
        let src = "struct [regional] TestCell { v: i32, next: [field] TestCell }\n\
                   regional fn clear(c: [flex] TestCell) -> i32 { c->next := Nullable::Null; c.v }";
        check(src, |elab, _| {
            // Body is `Seq[ Assign(c, next, Nullable::Null), c.v ]`.
            let body = function(elab, "clear").body.as_ref().unwrap();
            let hir::ExprKind::Seq(items) = &body.kind else {
                panic!("expected a block, got {:?}", body.kind)
            };
            let src_ty = items
                .iter()
                .find_map(|e| match &e.kind {
                    hir::ExprKind::Assign(_, _, src) => Some(src.ty),
                    _ => None,
                })
                .expect("an assignment");
            let TyKind::Nullable(inner) = src_ty.kind() else {
                panic!("assignment source is not Nullable: {src_ty:?}")
            };
            assert_eq!(
                inner.flexivity(),
                Some(Flexivity::Flex),
                "a `[field]` null's element should be flex, got {:?}",
                inner.flexivity()
            );
        });
    }

    #[test]
    fn rejects_storing_a_rigid_value_into_a_field_link() {
        // A `[field]` link on a flex record holds region-local (flex) values. A
        // frozen (`rigid`) value — here the result of a `region { }` that escaped
        // its region — is a flexivity mismatch against that flex slot.
        let src = "struct [regional] TestCell { v: i32, next: [field] TestCell }\n\
                   fn frozen() -> TestCell { regional { TestCell { v: 0, next: Nullable::Null } } }\n\
                   regional fn t(c: [flex] TestCell) -> i32 { c->next := Nullable::NonNull{ frozen() }; c.v }";
        assert!(
            has_error(src, "flexivity mismatch"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn rejects_returning_a_frozen_body_as_flex() {
        // A regional function whose body is a frozen (`rigid`) value but which
        // declares a `[flex]` return is a flexivity mismatch at the return — the
        // residual the boundary check alone (which only guards the non-regional
        // case) would miss.
        let src = "struct [regional] A { v: i32, next: [field] A }\n\
                   fn mk() -> A { regional { A { v: 0, next: Nullable::Null } } }\n\
                   regional fn f() -> [flex] A { mk() }";
        assert!(
            has_error(src, "flexivity mismatch"),
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

    // ----- named-function lifting (a `fn` used as a closure value) -----

    #[test]
    fn lifts_a_named_function_used_as_a_value() {
        // A bare `fn` name in argument position lifts to a closure of the
        // function's type, matching the expected `i32 -> i32` parameter.
        check(
            "fn double(x: i32) -> i32 { x * 2 }\n\
             fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }\n\
             pub fn run() -> i32 { apply(double, 21) }",
            |elab, _| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
                // The lifted argument is a `Closure`, not a bare reference.
                use crate::semi::hir::ExprKind;
                let run = elab
                    .elaborated
                    .iter()
                    .find(|f| elab.resolver.resolve(f.name) == "run")
                    .expect("run is elaborated");
                let ExprKind::FuncCall { args, .. } = &run.body.as_ref().unwrap().kind else {
                    panic!("run body is a call to apply");
                };
                assert!(
                    matches!(args[0].kind, ExprKind::Closure(_)),
                    "the `double` argument lifted to a closure, got {:#?}",
                    args[0].kind
                );
            },
        );
    }

    #[test]
    fn partially_applies_a_named_function() {
        // Calling a two-parameter `fn` with one argument lifts it and applies
        // the argument, yielding a residual `i32 -> i32` closure.
        let src = "fn add(a: i32, b: i32) -> i32 { a + b }\n\
                   fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }\n\
                   pub fn run() -> i32 { let g = add(5); apply(g, 37) }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn lifts_a_generic_function_at_the_expected_type() {
        // Lifting an unapplied generic `fn` leaves holes for its type parameters;
        // the expected closure type (`i32 -> i32`) then solves them.
        let src = "fn id<T>(x: T) -> T { x }\n\
                   fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }\n\
                   pub fn run() -> i32 { apply(id, 42) }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn rejects_regional_function_as_a_closure_value() {
        // A regional function takes an implicit region handle, so it has no
        // plain closure type to lift to.
        let src = "regional fn helper(x: i32) -> i32 { x }\n\
                   fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }\n\
                   pub fn run() -> i32 { apply(helper, 1) }";
        assert!(
            has_error(src, "regional function as a closure value"),
            "{:#?}",
            reports_of(src)
        );
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

    // ----- fuzzy "did you mean" suggestions on a failed name resolution -----

    #[test]
    fn unknown_function_suggests_a_close_name() {
        // `lenght` is `length` with the last two letters transposed — caught by
        // the edit-distance fallback, not nucleo's subsequence match.
        let src = "fn length(x: i32) -> i32 { x }\n\
                   fn use_it() -> i32 { lenght(0) }";
        assert!(
            has_error(src, "did you mean `length`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn unknown_type_suggests_a_close_record() {
        let src = "struct Point { x: i32, y: i32 }\n\
                   fn f(p: Pont) -> i32 { 0 }";
        assert!(
            has_error(src, "did you mean `Point`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn unknown_enum_suggests_a_close_record() {
        let src = "enum Color { Red, Green }\n\
                   fn f() -> i32 { Colr::Red; 0 }";
        assert!(
            has_error(src, "did you mean `Color`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn unknown_variable_suggests_a_close_binding() {
        // `valeu` is `value` with the last two letters transposed.
        let src = "fn f(value: i32) -> i32 { valeu }";
        assert!(
            has_error(src, "did you mean `value`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn unknown_trait_bound_suggests_a_builtin() {
        let src = "fn f<T: Nm>(x: T) -> T { x }";
        assert!(
            has_error(src, "did you mean `Num`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn frecency_steers_an_ambiguous_suggestion() {
        // `Lst` is an equally-close typo of both `List` and `Last`. The body
        // constructs `List` repeatedly before the typo, so the frecency built up
        // during checking must steer the hint toward `List` rather than `Last`.
        let src = "struct List { a: i32 }\n\
                   struct Last { a: i32 }\n\
                   fn f() -> i32 {\n\
                       List { a: 1 };\n\
                       List { a: 2 };\n\
                       List { a: 3 };\n\
                       Lst { a: 0 };\n\
                       0\n\
                   }";
        assert!(
            has_error(src, "did you mean `List`?"),
            "{:#?}",
            reports_of(src)
        );
    }

    #[test]
    fn unrelated_unknown_name_offers_no_suggestion() {
        // Nothing in scope is close to `Frobnicate`, so no hint is appended.
        let src = "struct Point { x: i32 }\n\
                   fn f(p: Frobnicate) -> i32 { 0 }";
        assert!(has_error(src, "unknown type"), "{:#?}", reports_of(src));
        assert!(!has_error(src, "did you mean"), "{:#?}", reports_of(src));
    }
}
