//! REPL-specific elaboration: one expression checked as a synthetic nullary
//! function, exposed to the JIT through a C-ABI trampoline root.
//!
//! A REPL driver keeps one persistent [`Elaborator`] for the whole session
//! (see [`Elaborator::try_extend`]) and calls [`Elaborator::try_repl_expr`]
//! for every expression input. On success the accumulated program gains a
//! private wrapper `fn __repl_expr_N() -> __ReplBox<T> { __ReplBox { value:
//! <expr> } }` plus a dec companion `fn __repl_expr_N_dec(b: __ReplBox<T>)
//! { }`, each with a C-ABI [`TrampolineRoot`](super::ctxt::TrampolineRoot),
//! so monomorphization roots them and codegen emits host-callable entries.
//! On any error the elaborator rolls back atomically, exactly like
//! `try_extend`.
//!
//! **Why box:** [`Elaborator::install_repl_box`] plants a session prelude
//! `struct [shared] __ReplBox<T> { value: T }` (the name is unspellable —
//! an identifier cannot start with `_` — so user code can never collide
//! with or name it). Wrapping every result in it makes every expression
//! cross the C ABI as *one* pointer to a shared rc box: scalars, value
//! records, enums, closures and nullables all take the same host call
//! shape, value aggregates get a stable heap address for the layout
//! walker, and releasing the *box* releases the whole result — the dec
//! companion is an empty *consuming* function, so the ownership pass
//! synthesizes the type-correct recursive drop; no hand-written drop glue
//! exists anywhere. A unit result skips the box (nothing to show or free).

use reussir_syntax::kind::TokenKey;

use crate::semi::ctxt::{
    Checkpoint, DefaultCap, Elaborator, FuncProto, Record, RecordFields, Report, Severity,
    TrampolineRoot,
};
use crate::semi::hir::{Expr, ExprKind, Function, VarId};
use crate::semi::ty::{DefId, Flexivity, Ty, TyKind};
use crate::surface::{self, Visibility};

/// One REPL expression evaluation request (see
/// [`Elaborator::try_repl_expr`]).
pub struct ReplExprRequest<'a, 'tcx> {
    /// The interned wrapper name (`__repl_expr_N`).
    pub name: TokenKey,
    /// The interned dec-companion name (`__repl_expr_N_dec`).
    pub dec_name: TokenKey,
    /// The exported C symbol (conventionally the wrapper name's text).
    pub export: &'a str,
    /// A `let` ascription to check the expression against, if any.
    pub ascription: Option<&'a (surface::Type, bool)>,
    /// The session's live global bindings (name, *value* type), in the
    /// order the host passes their boxes on every call.
    pub bindings: &'a [(TokenKey, Ty<'tcx>)],
    /// The `__ReplBox` prelude record ([`Elaborator::install_repl_box`]).
    pub repl_box: DefId,
}

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
    /// by the zonk-time ambiguity check with an annotation hint.
    pub fn try_repl_expr(
        &mut self,
        request: &ReplExprRequest<'_, 'tcx>,
        expr: &surface::Expr,
    ) -> Result<(Ty<'tcx>, Vec<Report>), Vec<Report>> {
        let &ReplExprRequest {
            name,
            dec_name,
            export,
            ascription,
            bindings,
            repl_box,
        } = request;
        let cp = self.checkpoint();
        let span = Some(expr.span());

        // Check the expression as a non-regional function body; the return
        // type is the `let` ascription when the driver passes one, else a
        // hole that inference solves to the expression's type.
        self.enter_function(&[]);
        self.regional_generics.clear();

        // Pre-seed the session's global `let` bindings: each becomes a
        // wrapper parameter `g` holding the binding's `__ReplBox<T>`,
        // immediately shadowed (same user-visible name) by the projected
        // payload, so the checked expression resolves the bare name to the
        // value. The host passes every live binding's box on each call —
        // an unused parameter is simply dropped by the ownership pass,
        // balancing the host-side retain.
        struct Seeded<'tcx> {
            name: TokenKey,
            param: VarId,
            box_ty: Ty<'tcx>,
            value: VarId,
            value_ty: Ty<'tcx>,
        }
        let seeded: Vec<Seeded<'tcx>> = bindings
            .iter()
            .map(|&(bname, bty)| {
                let box_ty = self.tcx.mk_record(repl_box, &[bty], Flexivity::Irrelevant);
                let param = self.vars.fresh(bname, box_ty, None);
                let value = self.vars.fresh(bname, bty, None);
                Seeded {
                    name: bname,
                    param,
                    box_ty,
                    value,
                    value_ty: bty,
                }
            })
            .collect();

        let ret = match ascription {
            Some((ty, flex)) => self.eval_type_flex(ty, *flex),
            None => self.infer.new_hole_ty(),
        };
        let body = self.check_expr(expr, ret);
        self.default_numeric_holes();
        self.resolve_obligations();
        let body = self.zonk_expr(body);
        let return_ty = self.infer.resolve(ret);

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

        // Accept: declare the wrapper and register its prototype (later
        // inputs may call it), the elaborated body, and the trampoline root
        // that seeds monomorphization and names the exported C entry.
        //
        // Declaring only now — after the body checked — keeps the "every
        // resolvable def has a prototype" invariant the call paths index by
        // (`self.functions[&def]`). Today no surface input can name the
        // wrapper anyway (a leading `_` cannot start an identifier), but the
        // invariant should hold structurally, not by lexical accident.
        let def = match self.defs.declare_function(name) {
            Some(def) => def,
            None => {
                // Defensive: the driver allocates a fresh `__repl_expr_N` per
                // input, so a clash means a driver bug, not user error.
                self.error(span, format!("`{}` is already defined", self.sym(name)));
                return Err(self.take_reports_and_rollback(&cp));
            }
        };
        // Box the result (module docs): the wrapper returns
        // `__ReplBox<T> { value: <expr> }` so every expression crosses the
        // C ABI as one shared-box pointer. Unit has nothing to show or
        // free and skips the box.
        let boxed = !matches!(return_ty.kind(), TyKind::Unit);
        // The binding prologue: `let x = g.value` per seeded binding, in
        // front of the (boxed) user expression.
        let params: Vec<(TokenKey, VarId, Ty<'tcx>)> =
            seeded.iter().map(|b| (b.name, b.param, b.box_ty)).collect();
        let prologue: Vec<Expr<'tcx>> = seeded
            .iter()
            .map(|b| {
                let base = Expr {
                    kind: ExprKind::Var(b.param),
                    ty: b.box_ty,
                    span,
                    id: self.fresh_expr_id(),
                };
                let projected = Expr {
                    kind: ExprKind::Proj(Box::new(base), vec![0]),
                    ty: b.value_ty,
                    span,
                    id: self.fresh_expr_id(),
                };
                Expr {
                    kind: ExprKind::Let {
                        var: b.value,
                        name: b.name,
                        span,
                        value: Box::new(projected),
                    },
                    ty: self.tcx.mk_unit(),
                    span,
                    id: self.fresh_expr_id(),
                }
            })
            .collect();
        let (body, wrapper_ty) = if boxed {
            // A shared record's coloring is `Irrelevant` (only regional
            // records carry Flex/Rigid), matching `record_ty`.
            let box_ty = self
                .tcx
                .mk_record(repl_box, &[return_ty], Flexivity::Irrelevant);
            let body = Expr {
                kind: ExprKind::CompoundCall {
                    target: repl_box,
                    ty_args: vec![return_ty],
                    args: vec![body],
                },
                ty: box_ty,
                span,
                id: self.fresh_expr_id(),
            };
            (body, box_ty)
        } else {
            (body, return_ty)
        };
        let body = if prologue.is_empty() {
            body
        } else {
            let ty = body.ty;
            let mut stmts = prologue;
            stmts.push(body);
            Expr {
                kind: ExprKind::Seq(stmts),
                ty,
                span,
                id: self.fresh_expr_id(),
            }
        };
        self.functions.insert(
            def,
            FuncProto {
                def,
                name,
                visibility: Visibility::Private,
                generics: Vec::new(),
                regional_generics: Vec::new(),
                params: params.iter().map(|&(n, _, t)| (n, t)).collect(),
                return_ty: wrapper_ty,
                is_regional: false,
                span,
                file: self.current_file,
            },
        );
        self.elaborated.push(Function {
            def,
            name,
            visibility: Visibility::Private,
            generics: Vec::new(),
            regional_generics: std::mem::take(&mut self.regional_generics),
            params,
            return_ty: wrapper_ty,
            is_regional: false,
            body: Some(body),
            span,
            file: self.current_file,
        });
        self.trampolines.push(TrampolineRoot {
            name: export.to_string(),
            abi: "C".to_string(),
            target: def,
            ty_args: Vec::new(),
        });

        // The dec companion: an empty function *consuming* the box. The
        // ownership pass drops the unused parameter, which is exactly the
        // type-correct recursive release of the whole displayed result —
        // the host calls it once the value walker is done.
        if boxed {
            let Some(dec_def) = self.defs.declare_function(dec_name) else {
                self.error(span, format!("`{}` is already defined", self.sym(dec_name)));
                return Err(self.take_reports_and_rollback(&cp));
            };
            self.functions.insert(
                dec_def,
                FuncProto {
                    def: dec_def,
                    name: dec_name,
                    visibility: Visibility::Private,
                    generics: Vec::new(),
                    regional_generics: Vec::new(),
                    params: vec![(dec_name, wrapper_ty)],
                    return_ty: self.tcx.mk_unit(),
                    is_regional: false,
                    span,
                    file: self.current_file,
                },
            );
            let dec_body = Expr {
                kind: ExprKind::Seq(Vec::new()),
                ty: self.tcx.mk_unit(),
                span,
                id: self.fresh_expr_id(),
            };
            self.elaborated.push(Function {
                def: dec_def,
                name: dec_name,
                visibility: Visibility::Private,
                generics: Vec::new(),
                regional_generics: Vec::new(),
                params: vec![(dec_name, VarId(0), wrapper_ty)],
                return_ty: self.tcx.mk_unit(),
                is_regional: false,
                body: Some(dec_body),
                span,
                file: self.current_file,
            });
            self.trampolines.push(TrampolineRoot {
                name: format!("{export}_dec"),
                abi: "C".to_string(),
                target: dec_def,
                ty_args: Vec::new(),
            });
        }
        Ok((return_ty, new))
    }

    /// Install the REPL's boxing prelude — `struct [shared] __ReplBox<T>
    /// { value: T }` (see the module docs) — under the given interned keys.
    /// Call once per session, **before** the first checkpoint, so the record
    /// is never retracted by a rollback. `None` if `name` is taken (a
    /// driver bug: the name is unspellable).
    pub fn install_repl_box(
        &mut self,
        name: TokenKey,
        ty_param: TokenKey,
        field: TokenKey,
    ) -> Option<DefId> {
        let def = self.defs.declare_record(name)?;
        let generic = self.fresh_generic(ty_param, Vec::new());
        self.records.insert(
            def,
            Record {
                def,
                name,
                ty_params: vec![(ty_param, generic)],
                kind: surface::RecordKind::StructKind,
                default_cap: DefaultCap::Shared,
                repr_fixed: false,
                fields: Some(RecordFields::Named(vec![(
                    field,
                    self.tcx.mk_generic(generic),
                    false,
                )])),
                regional_generics: Vec::new(),
                span: None,
                file: self.current_file,
            },
        );
        Some(def)
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

    use crate::semi::repl::ReplExprRequest;
    use crate::semi::ty::{FpTy, IntTy, Ty, TyCtxt, TyKind};
    use crate::semi::{Elaborator, Report};
    use crate::surface;
    use crate::with_tcx;

    /// A REPL-session harness: one shared interner, one elaborator (with
    /// the boxing prelude installed), a counter for `__repl_expr_N` names.
    struct Session<'a, 'tcx> {
        interner: Arc<MultiThreadedTokenInterner>,
        elab: Elaborator<'a, 'tcx>,
        repl_box: crate::semi::ty::DefId,
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
            let dec_key =
                Interner::get_or_intern(&mut self.interner.clone(), &format!("{export}_dec"));
            self.elab
                .try_repl_expr(
                    &ReplExprRequest {
                        name: key,
                        dec_name: dec_key,
                        export: &export,
                        ascription: None,
                        bindings: &[],
                        repl_box: self.repl_box,
                    },
                    &expr,
                )
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
            let mut interning = interner.clone();
            let box_name = Interner::get_or_intern(&mut interning, "__ReplBox");
            let box_param = Interner::get_or_intern(&mut interning, "T");
            let box_field = Interner::get_or_intern(&mut interning, "value");
            let mut elab = Elaborator::new(tcx, resolver);
            let repl_box = elab
                .install_repl_box(box_name, box_param, box_field)
                .expect("fresh session");
            let mut s = Session {
                interner: interner.clone(),
                elab,
                repl_box,
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
    fn clashing_wrapper_names_are_a_clean_error() {
        session(|s, _| {
            s.eval("1").expect("first use of the name");
            // A driver bug reusing an export name must surface as a report
            // (and roll back), not corrupt the session.
            s.counter = 0;
            let errs = s.eval("2").expect_err("name already taken");
            assert!(
                errs.iter().any(|r| r.message.contains("already defined")),
                "{errs:#?}"
            );
            // Wrapper + dec companion from the surviving first input.
            assert_eq!(s.elab.elaborated.len(), 2, "clashing input rolled back");
            s.counter = 1;
            s.eval("2").expect("session still usable");
        });
    }

    #[test]
    fn registers_the_wrapper_and_trampoline() {
        session(|s, _| {
            s.eval("40 + 2").unwrap();
            // The wrapper and its dec companion, each with a trampoline.
            assert_eq!(s.elab.elaborated.len(), 2);
            let f = &s.elab.elaborated[0];
            assert_eq!(s.elab.sym(f.name), "__repl_expr_0");
            assert!(f.generics.is_empty() && f.params.is_empty());
            // The wrapper returns the box, not the bare expression type.
            assert!(matches!(f.return_ty.kind(), TyKind::Record { .. }));
            let dec = &s.elab.elaborated[1];
            assert_eq!(s.elab.sym(dec.name), "__repl_expr_0_dec");
            assert_eq!(dec.params.len(), 1, "dec consumes the box");
            assert_eq!(dec.params[0].2, f.return_ty);
            assert_eq!(s.elab.trampolines.len(), 2);
            let t = &s.elab.trampolines[0];
            assert_eq!(t.name, "__repl_expr_0");
            assert_eq!(t.abi, "C");
            assert_eq!(t.target, f.def);
            assert_eq!(s.elab.trampolines[1].name, "__repl_expr_0_dec");
            assert_eq!(s.elab.trampolines[1].target, dec.def);
        });
    }

    #[test]
    fn global_bindings_become_wrapper_parameters() {
        use crate::full::mono::monomorphize;

        session(|s, tcx| {
            // Bind `x` by evaluating an ordinary expression first (the
            // driver stores its box), then check an expression referencing
            // it: the wrapper takes the binding's box as a parameter and
            // projects the value in a prologue.
            let x_ty = s.eval("41").expect("bind rhs");
            let x_key = Interner::get_or_intern(&mut s.interner.clone(), "x");

            let p = reussir_syntax::parse_repl("x + 1", s.interner.clone());
            assert!(p.parse.ok());
            let expr = surface::repl_expr(&p.parse.root);
            let key = Interner::get_or_intern(&mut s.interner.clone(), "__repl_expr_1");
            let dec = Interner::get_or_intern(&mut s.interner.clone(), "__repl_expr_1_dec");
            let (ty, _) = s
                .elab
                .try_repl_expr(
                    &ReplExprRequest {
                        name: key,
                        dec_name: dec,
                        export: "__repl_expr_1",
                        ascription: None,
                        bindings: &[(x_key, x_ty)],
                        repl_box: s.repl_box,
                    },
                    &expr,
                )
                .expect("binding resolves");
            assert_eq!(ty, tcx.mk_int(IntTy::Signed(64)));

            // The wrapper's sole parameter is the binding's box.
            let wrapper = s
                .elab
                .elaborated
                .iter()
                .find(|f| s.elab.sym(f.name) == "__repl_expr_1")
                .expect("wrapper");
            assert_eq!(wrapper.params.len(), 1);
            assert!(matches!(wrapper.params[0].2.kind(), TyKind::Record { .. }));

            // An unknown name still errors cleanly.
            let p = reussir_syntax::parse_repl("y + 1", s.interner.clone());
            let expr = surface::repl_expr(&p.parse.root);
            let key = Interner::get_or_intern(&mut s.interner.clone(), "__repl_expr_2");
            let dec = Interner::get_or_intern(&mut s.interner.clone(), "__repl_expr_2_dec");
            let errs = s
                .elab
                .try_repl_expr(
                    &ReplExprRequest {
                        name: key,
                        dec_name: dec,
                        export: "__repl_expr_2",
                        ascription: None,
                        bindings: &[(x_key, x_ty)],
                        repl_box: s.repl_box,
                    },
                    &expr,
                )
                .expect_err("unknown name");
            assert!(
                errs.iter().any(|r| r.message.contains("unknown")),
                "{errs:#?}"
            );

            // The whole accumulated program (parameterized wrapper included)
            // monomorphizes.
            let (_, reports) = monomorphize(&s.elab.mono_input());
            assert!(reports.is_empty(), "{reports:#?}");
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
            // Two expressions × (wrapper + dec companion).
            assert_eq!(program.trampolines.len(), 4);
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
