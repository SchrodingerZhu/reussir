//! REPL-specific elaboration: one expression checked as a synthetic nullary
//! function, exposed to the JIT through a C-ABI trampoline root.
//!
//! A REPL driver keeps one persistent [`Elaborator`] for the whole session
//! (see [`Elaborator::try_extend`]) and calls [`Elaborator::try_repl_expr`]
//! for every expression input. On success the accumulated program gains a
//! private `fn __repl_expr_N() -> T { <expr> }` plus a
//! [`TrampolineRoot`](super::ctxt::TrampolineRoot) exporting the same name,
//! so monomorphization roots the wrapper and codegen emits a host-callable
//! `extern "C"` entry for it. On any error the elaborator rolls back
//! atomically, exactly like `try_extend`.

use reussir_syntax::kind::TokenKey;

use crate::semi::ctxt::{Checkpoint, Elaborator, FuncProto, Report, Severity, TrampolineRoot};
use crate::semi::fulfill::{expr_has_hole, ty_has_hole};
use crate::semi::hir::Function;
use crate::semi::ty::Ty;
use crate::surface::{self, Visibility};

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Elaborate one REPL expression as the synthetic nullary function
    /// `name`, registering a C-ABI trampoline root that exports `export`
    /// (conventionally the same text, `__repl_expr_N`; the driver interns it
    /// into the shared session interner).
    ///
    /// Atomic: `Err` carries the input's reports with all state rolled back;
    /// `Ok` carries the expression's resolved (ground) type plus any
    /// warnings. Numeric inference holes are defaulted REPL-style
    /// ([`default_numeric_holes`](Elaborator::default_numeric_holes)); a hole
    /// that survives defaulting — e.g. a bare `Nullable::Null` — is rejected
    /// with an annotation hint rather than left to panic in
    /// monomorphization.
    pub fn try_repl_expr(
        &mut self,
        name: TokenKey,
        export: &str,
        expr: &surface::Expr,
    ) -> Result<(Ty<'tcx>, Vec<Report>), Vec<Report>> {
        let cp = self.checkpoint();
        let span = Some(expr.span());

        let def = match self.defs.declare_function(name) {
            Some(def) => def,
            None => {
                // Defensive: the driver allocates a fresh `__repl_expr_N` per
                // input, so a clash means a driver bug, not user error.
                self.error(span, format!("`{}` is already defined", self.sym(name)));
                return Err(self.take_reports_and_rollback(&cp));
            }
        };

        // Check the expression as a nullary, non-regional function body with
        // a hole return type; inference solves the hole to the expression's
        // type.
        self.enter_function(&[]);
        self.regional_generics = Vec::new();
        let ret = self.infer.new_hole_ty();
        let body = self.check_expr(expr, ret);
        self.default_numeric_holes();
        self.resolve_obligations();
        let body = self.zonk_expr(body);
        let return_ty = self.infer.resolve(ret);

        // A residual hole would panic in monomorphization (`subst_ty`);
        // reject it here with an actionable message instead.
        if ty_has_hole(return_ty) || expr_has_hole(&body) {
            self.error(
                span,
                "cannot infer the type of this expression; add a type annotation",
            );
        }
        // The wrapper is an ordinary (non-regional) function, so its result
        // must not be region-local (mirrors the check_function boundary rule).
        if self.is_flex(return_ty) {
            self.error(
                span,
                "a REPL expression cannot produce a flex value: a flex value \
                 cannot escape its region",
            );
        }

        let new = self.reports.split_off(cp.reports);
        if new.iter().any(|r| r.severity == Severity::Error) {
            self.rollback(&cp);
            return Err(new);
        }

        // Accept: register the prototype (later inputs may call the wrapper),
        // the elaborated body, and the trampoline root that seeds
        // monomorphization and names the exported C entry.
        self.functions.insert(
            def,
            FuncProto {
                def,
                name,
                visibility: Visibility::Private,
                generics: Vec::new(),
                regional_generics: Vec::new(),
                params: Vec::new(),
                return_ty,
                is_regional: false,
                span,
            },
        );
        self.elaborated.push(Function {
            def,
            name,
            visibility: Visibility::Private,
            generics: Vec::new(),
            regional_generics: std::mem::take(&mut self.regional_generics),
            params: Vec::new(),
            return_ty,
            is_regional: false,
            body: Some(body),
            span,
        });
        self.trampolines.push(TrampolineRoot {
            name: export.to_string(),
            abi: "C".to_string(),
            target: def,
            ty_args: Vec::new(),
        });
        Ok((return_ty, new))
    }

    /// Split off the reports added since `cp`, then roll back.
    fn take_reports_and_rollback(&mut self, cp: &Checkpoint) -> Vec<Report> {
        let new = self.reports.split_off(cp.reports);
        self.rollback(cp);
        new
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reussir_syntax::{Interner, MultiThreadedTokenInterner, new_threaded_interner};

    use crate::semi::ty::{FpTy, IntTy, Ty, TyCtxt, TyKind};
    use crate::semi::{Elaborator, Report};
    use crate::surface;
    use crate::with_tcx;

    /// A REPL-session harness: one shared interner, one elaborator, a
    /// counter for `__repl_expr_N` names.
    struct Session<'a, 'tcx> {
        interner: Arc<MultiThreadedTokenInterner>,
        elab: Elaborator<'a, 'tcx>,
        counter: usize,
    }

    impl<'a, 'tcx> Session<'a, 'tcx> {
        fn define(&mut self, source: &str) {
            let p = reussir_syntax::parse_repl(source, self.interner.clone());
            assert_eq!(p.kind, reussir_syntax::ReplInputKind::Items, "{source}");
            assert!(p.parse.ok(), "{source}: {:#?}", p.parse.errors);
            self.elab
                .try_extend(&surface::program(&p.parse.root))
                .unwrap_or_else(|e| panic!("{source}: {e:#?}"));
        }

        fn eval(&mut self, source: &str) -> Result<Ty<'tcx>, Vec<Report>> {
            let p = reussir_syntax::parse_repl(source, self.interner.clone());
            assert_eq!(p.kind, reussir_syntax::ReplInputKind::Expr, "{source}");
            assert!(p.parse.ok(), "{source}: {:#?}", p.parse.errors);
            let expr = surface::repl_expr(&p.parse.root);
            let export = format!("__repl_expr_{}", self.counter);
            self.counter += 1;
            let key = Interner::get_or_intern(&mut self.interner.clone(), &export);
            self.elab
                .try_repl_expr(key, &export, &expr)
                .map(|(ty, _)| ty)
        }
    }

    fn session<R>(f: impl for<'a, 'tcx> FnOnce(&mut Session<'a, 'tcx>, &TyCtxt<'tcx>) -> R) -> R {
        with_tcx(|tcx| {
            let interner = Arc::new(new_threaded_interner());
            // The elaborator borrows the session interner as its resolver —
            // the same shape a real REPL driver uses.
            let resolver: &dyn reussir_syntax::kind::Resolver<reussir_syntax::kind::TokenKey> =
                &interner;
            let mut s = Session {
                interner: interner.clone(),
                elab: Elaborator::new(tcx, resolver),
                counter: 0,
            };
            f(&mut s, tcx)
        })
    }

    #[test]
    fn defaults_numeric_literals() {
        session(|s, tcx| {
            // An unconstrained integer literal defaults to i64...
            assert_eq!(s.eval("1 + 1").unwrap(), tcx.mk_int(IntTy::Signed(64)));
            // ...a float literal to f64, including mixed Num+FloatingPoint
            // obligations on one hole.
            assert_eq!(s.eval("1.5").unwrap(), tcx.mk_fp(FpTy::Ieee(64)));
            assert_eq!(s.eval("1.5 * 2.0").unwrap(), tcx.mk_fp(FpTy::Ieee(64)));
            // Ground types are untouched.
            assert!(matches!(s.eval("true").unwrap().kind(), TyKind::Bool));
            assert!(matches!(s.eval("1 == 2").unwrap().kind(), TyKind::Bool));
        });
    }

    #[test]
    fn defaulting_reaches_generic_instantiations() {
        session(|s, tcx| {
            s.define("fn id<T>(x: T) -> T { x }");
            // The literal's hole flows into `id`'s instantiation; defaulting
            // must solve it there, not just at the root type.
            assert_eq!(s.eval("id(1)").unwrap(), tcx.mk_int(IntTy::Signed(64)));
        });
    }

    #[test]
    fn expression_sequences_get_block_semantics() {
        session(|s, tcx| {
            assert_eq!(
                s.eval("let a = 10; let b = 20; a + b").unwrap(),
                tcx.mk_int(IntTy::Signed(64))
            );
        });
    }

    #[test]
    fn residual_holes_are_rejected_cleanly() {
        session(|s, _| {
            // `Nullable::Null`'s element hole carries no numeric obligation,
            // so defaulting cannot solve it; the guard must reject it instead
            // of letting monomorphization panic.
            let errs = s.eval("Nullable::Null").expect_err("unconstrained hole");
            assert!(
                errs.iter().any(|r| r.message.contains("cannot infer")),
                "{errs:#?}"
            );
            // The failed input rolled back: the next one reuses the def space
            // without a clash and the session stays usable.
            assert_eq!(s.elab.elaborated.len(), 0);
            s.counter = 0;
            s.eval("42").expect("session still usable");
        });
    }

    #[test]
    fn registers_the_wrapper_and_trampoline() {
        session(|s, _| {
            s.eval("40 + 2").unwrap();
            assert_eq!(s.elab.elaborated.len(), 1);
            let f = &s.elab.elaborated[0];
            assert_eq!(s.elab.sym(f.name), "__repl_expr_0");
            assert!(f.generics.is_empty() && f.params.is_empty());
            assert_eq!(s.elab.trampolines.len(), 1);
            let t = &s.elab.trampolines[0];
            assert_eq!(t.name, "__repl_expr_0");
            assert_eq!(t.abi, "C");
            assert_eq!(t.target, f.def);
        });
    }

    #[test]
    fn monomorphizes_on_the_fly_instantiations() {
        use crate::full::mono::monomorphize;

        session(|s, _| {
            s.define("fn poly_add<T : Num>(x: T) -> T { x + x }");
            s.eval("poly_add(21)").unwrap();
            s.eval("poly_add(1.5)").unwrap();

            // The maintained HIR monomorphizes with both on-the-fly
            // instantiations of `poly_add` and both expression trampolines.
            let (program, reports) = monomorphize(&s.elab.mono_input());
            assert!(reports.is_empty(), "{reports:#?}");
            assert_eq!(program.trampolines.len(), 2);
            let names: Vec<&str> = program
                .functions
                .iter()
                .map(|f| program.symbol(f.symbol))
                .collect();
            assert!(
                names.iter().filter(|n| n.contains("poly_add")).count() >= 2,
                "{names:?}"
            );
        });
    }
}
