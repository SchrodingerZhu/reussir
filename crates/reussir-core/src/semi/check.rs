//! The bidirectional type checker: surface expressions → typed Semi HIR.

use reussir_syntax::kind::TokenKey;

use crate::semi::infer::Instantiation;
use crate::semi::traits::{Obligation, TraitId, TraitRef};
use crate::semi::ty::{DefId, Flexivity, GenericId, Ty, TyKind};
use crate::surface::{self, BinOp, Const, Span, UnaryOp};

use super::ctxt::{Elaborator, RecordFields};
use super::fulfill::{collect_holes, ty_has_hole};
use super::hir::{ArithOp, ClosureExpr, CmpOp, Expr, ExprKind, Function, VarId};

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Check a function definition: bind its parameters into Γ, check the body
    /// against the declared return type (`Γ ⊢ body ⇐ return_ty`), discharge the
    /// collected trait obligations, then zonk. Drives the checking judgment over a
    /// whole function body.
    pub(super) fn check_function(
        &mut self,
        func: &surface::Function,
        def: DefId,
        span: Option<Span>,
    ) {
        let Some(proto) = self.functions.get(&def).cloned() else {
            return;
        };
        // The body's references resolve (and its reports attribute) in the
        // function's own declaration scope — the package driver checks items
        // from many files/modules in one pass.
        self.enter_item_scope(def, proto.file);
        self.enter_function(&proto.generics);
        // A `[regional]` function body runs inside an implicit region: its
        // parameters may already be flex and it may construct regional records
        // directly, so the body is checked with the region context active.
        self.inside_region = proto.is_regional;
        // Seed the regional-requirement set from the prototype's `[flex]`-position
        // generics; the body may add more (e.g. a generic assigned into a link).
        self.regional_generics = proto.regional_generics.clone();

        // A flex value is region-local and cannot be materialized, so it cannot
        // escape its region across a function boundary. A `regional fn` shares its
        // caller's region (it takes the region as an implicit parameter), so flex
        // may cross its boundary; a plain function has no such region, so a flex
        // parameter or return is rejected. This also rejects declaring a `[flex]`
        // return whose body can only be a frozen (`rigid`) region-run result.
        if !proto.is_regional {
            if self.is_flex(proto.return_ty) {
                self.error(
                    span,
                    "a non-regional function cannot return a flex value: a flex \
                     value cannot escape its region",
                );
            }
            for (name, ty) in &proto.params {
                if self.is_flex(*ty) {
                    let name = self.sym(*name);
                    self.error(
                        span,
                        format!(
                            "a non-regional function cannot take flex parameter \
                             `{name}`: a flex value cannot escape its region"
                        ),
                    );
                }
            }
        }

        let mut params = Vec::new();
        for (name, ty) in &proto.params {
            let var = self.vars.fresh(*name, *ty, None);
            params.push((*name, var, *ty));
        }

        let body = func
            .body
            .as_ref()
            .map(|b| self.check_expr(b, proto.return_ty));
        self.resolve_obligations();
        let body = body.map(|b| self.zonk_expr(b));
        let return_ty = self.infer.resolve(proto.return_ty);

        self.elaborated.push(Function {
            def: proto.def,
            name: proto.name,
            visibility: proto.visibility,
            generics: proto.generics,
            regional_generics: std::mem::take(&mut self.regional_generics),
            params,
            return_ty,
            is_regional: proto.is_regional,
            body,
            span,
            file: proto.file,
        });
    }

    fn mk_expr(&mut self, kind: ExprKind<'tcx>, ty: Ty<'tcx>, span: Option<Span>) -> Expr<'tcx> {
        let id = self.fresh_expr_id();
        Expr { kind, ty, span, id }
    }

    /// (POISON) error recovery: `Γ ⊢ e ⇒ (poison : ⊥)`. `⊥` (Bottom) unifies with
    /// anything and is never stored into a hole, so a poisoned subterm does not
    /// cascade into spurious follow-on diagnostics.
    fn poison(&mut self, span: Option<Span>) -> Expr<'tcx> {
        let ty = self.tcx.mk(TyKind::Bottom);
        self.mk_expr(ExprKind::Poison, ty, span)
    }

    /// (CHECK) the unification-coercion that backs the checking judgment: unify the
    /// synthesized type `found` with `expected`. On failure emit a mismatch
    /// (CHECK-FAIL) but do not abort — checking always yields a node.
    fn expect(&mut self, found: Ty<'tcx>, expected: Ty<'tcx>, span: Option<Span>) {
        if self.infer.unify(expected, found).is_err() {
            let e = self.infer.resolve(expected);
            let f = self.infer.resolve(found);
            self.error(
                span,
                format!(
                    "type mismatch: expected `{}`, found `{}`",
                    self.ty_display(e),
                    self.ty_display(f)
                ),
            );
            return;
        }
        // `unify` reconciles structure but not flexivity coloring, so check it
        // here at the head (peeling `Nullable`, e.g. a `[field]` link). Two
        // colorings are compatible when one refines to the other in the
        // [refinement tree](crate::semi::ty::Flexivity); the incompatible cases
        // are the siblings — `Flex` versus `Rigid`. Rejecting them is what stops a
        // frozen `rigid` value being stored into a flex `[field]` link, or a
        // frozen `region { }` result satisfying a `[flex]` return. An `Unknown`
        // head (a non-record, or a not-yet-solved hole such as a `Nullable::Null`
        // element) is the tree root and compatible with either, so it simply
        // takes the expected coloring.
        if let (Some(f), Some(e)) = (self.head_flexivity(found), self.head_flexivity(expected))
            && !f.compatible(e)
        {
            self.error(
                span,
                format!(
                    "flexivity mismatch: a `{f:?}` regional value cannot be used where \
                     `{e:?}` is required (one does not refine to the other)"
                ),
            );
        }
    }

    /// The flexivity coloring at the head of `ty`, peeling a `Nullable` wrapper
    /// (a `[field]` link is a `Nullable<Record>`). `None` for a non-record head.
    fn head_flexivity(&mut self, ty: Ty<'tcx>) -> Option<crate::semi::ty::Flexivity> {
        match self.infer.shallow_resolve(ty).kind() {
            TyKind::Record { flex, .. } => Some(*flex),
            TyKind::Nullable(inner) => self.head_flexivity(*inner),
            _ => None,
        }
    }

    /// The **checking** judgment `Γ ⊢ e ⇐ T ⇒ h`: synthesize `e`, then [`Self::expect`]
    /// its type against `T` (the CHECK rule). Always produces a node.
    pub(super) fn check_expr(&mut self, e: &surface::Expr, expected: Ty<'tcx>) -> Expr<'tcx> {
        let h = self.infer_expr(e);
        self.expect(h.ty, expected, h.span);
        h
    }

    /// The **synthesis** judgment `Γ ⊢ e ⇒ (h : T)`: dispatch each surface form to
    /// its typing rule, producing typed HIR whose `ty` field is the synthesized `T`.
    /// Notation used by the per-form rules below: `⇒` synthesizes a type, `⇐` checks
    /// against one, and `Trait ⊳ T` is a registered trait obligation on `T`.
    pub(super) fn infer_expr(&mut self, e: &surface::Expr) -> Expr<'tcx> {
        let span = Some(e.span());
        let kind = e.kind();
        match &kind {
            surface::ExprKind::ConstExpr(c) => self.infer_const(c, span),
            surface::ExprKind::Var(path) => self.infer_var(path, span),
            surface::ExprKind::ExprSeq(exprs) => self.infer_seq(exprs, span),
            surface::ExprKind::If(c, t, f) => self.infer_if(c, t, f, span),
            surface::ExprKind::Let(name, ty, value) => {
                self.infer_let(name, ty.as_ref(), value, span)
            }
            surface::ExprKind::BinOpExpr(op, l, r) => self.infer_binop(*op, l, r, span),
            surface::ExprKind::UnaryOpExpr(op, e) => self.infer_unop(*op, e, span),
            surface::ExprKind::Cast(ty, e) => self.infer_cast(ty, e, span),
            surface::ExprKind::FuncCallExpr(fc) => self.infer_func_call(fc, span),
            surface::ExprKind::CtorCallExpr(cc) => self.infer_ctor_call(cc, span),
            surface::ExprKind::CallExpr(callee, args) => {
                self.infer_closure_call(callee, args, span)
            }
            surface::ExprKind::AccessChain(base, accs) => self.infer_access(base, accs, span),
            surface::ExprKind::Assign(dst, acc, src) => self.infer_assign(dst, acc, src, span),
            surface::ExprKind::RegionalExpr(body) => self.infer_region(body, span),
            surface::ExprKind::Lambda(lam) => self.infer_lambda(lam, span),
            surface::ExprKind::Match(scrut, arms) => self.infer_match(scrut, arms, span),
        }
    }

    /// (LIT) literals synthesize: an integer literal is a fresh hole `α` with
    /// `Integral ⊳ α`, a float literal a fresh `α` with `FloatingPoint ⊳ α`; `s ⇒ Str`
    /// and `b ⇒ Bool`. The literal's concrete width is left to inference.
    fn infer_const(&mut self, c: &Const, span: Option<Span>) -> Expr<'tcx> {
        match c {
            Const::ConstInt(i) => {
                let hole = self.infer.new_hole_ty();
                self.register_bound(self.builtins.integral, hole, span);
                self.mk_expr(ExprKind::ConstInt(self.tcx.alloc(i.clone())), hole, span)
            }
            Const::ConstFloat(f) => {
                let hole = self.infer.new_hole_ty();
                self.register_bound(self.builtins.floating_point, hole, span);
                self.mk_expr(ExprKind::ConstFloat(self.tcx.alloc(f.clone())), hole, span)
            }
            Const::ConstString(s) => {
                let token = self.strings.allocate(s);
                let ty = self.tcx.mk_str();
                self.mk_expr(ExprKind::GlobalStr(token), ty, span)
            }
            Const::ConstChar(c) => {
                let ty = self.tcx.mk_char();
                self.mk_expr(ExprKind::ConstChar(*c), ty, span)
            }
            Const::ConstBool(b) => {
                let ty = self.tcx.mk_bool();
                self.mk_expr(ExprKind::ConstBool(*b), ty, span)
            }
        }
    }

    /// Register the obligation `ty : Trait<trait_id>` — a single-parameter bound,
    /// written `Trait ⊳ ty` in the rules — for the fulfillment pass to discharge.
    pub(super) fn register_bound(&mut self, trait_id: TraitId, ty: Ty<'tcx>, span: Option<Span>) {
        self.fulfill.register(
            Obligation::Trait(TraitRef {
                trait_id,
                args: vec![ty],
            }),
            span,
        );
    }

    /// (VAR) `Γ ⊢ x ⇒ (Var v : T)` when `x:T` is in scope; an unknown bare name is
    /// (VAR-ERR) → poison. A qualified path with no arguments is a nullary constructor.
    fn infer_var(&mut self, path: &surface::Path, span: Option<Span>) -> Expr<'tcx> {
        if path.segments.is_empty() {
            if let Some((id, ty)) = self.vars.lookup(path.basename) {
                self.record_use(path.basename);
                return self.mk_expr(ExprKind::Var(id), ty, span);
            }
            let hint = self.variable_suggestion(path.basename);
            self.error(
                span,
                format!("unknown variable `{}`{hint}", self.sym(path.basename)),
            );
            return self.poison(span);
        }
        // A qualified path with no arguments: a nullary constructor.
        self.infer_ctor(path, &[], &[], span)
    }

    /// (SEQ) `Γ ⊢ {e₁;…;eₙ} ⇒ (Seq[hᵢ] : Tₙ)`: synthesize each element in a scope
    /// marked on entry and restored on exit; an empty sequence has type `Unit`.
    fn infer_seq(&mut self, exprs: &[surface::Expr], span: Option<Span>) -> Expr<'tcx> {
        let mark = self.vars.mark();
        let mut out = Vec::with_capacity(exprs.len());
        let mut ty = self.tcx.mk_unit();
        for e in exprs {
            let h = self.infer_expr(e);
            ty = h.ty;
            out.push(h);
        }
        self.vars.restore(mark);
        // A one-statement block *is* that statement: collapse `{ e }` to `e`
        // rather than wrapping it in a single-element `Seq`. The textual HIR
        // grammar does the same (a one-element block parses back to the bare
        // expression), so emitting the `Seq` here would make a printed body fail
        // to round-trip — and lower to different MIR — than its re-parsed form.
        if out.len() == 1 {
            return out.pop().unwrap();
        }
        self.mk_expr(ExprKind::Seq(out), ty, span)
    }

    /// (IF) `Γ ⊢ if c t f ⇒ (If(hc,ht,hf) : T)`: `c ⇐ Bool`, `t ⇒ T`, then `f ⇐ T`
    /// (the then-branch's synthesized type drives the else-branch).
    fn infer_if(
        &mut self,
        c: &surface::Expr,
        t: &surface::Expr,
        f: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let bool_ty = self.tcx.mk_bool();
        let c = self.check_expr(c, bool_ty);
        let t = self.infer_expr(t);
        let result = t.ty;
        let f = self.check_expr(f, result);
        self.mk_expr(
            ExprKind::If(Box::new(c), Box::new(t), Box::new(f)),
            result,
            span,
        )
    }

    /// (LET) `Γ ⊢ let name:ann = v ⇒ (Let{…} : Unit)`: with an annotation `(A, flex)`,
    /// `v ⇐ eval_flex(A)` and `Tv = eval_flex(A)`; without one, `v ⇒ Tv`. Binds
    /// `name : Tv` into Γ for the rest of the enclosing sequence.
    fn infer_let(
        &mut self,
        name: &surface::Spanned<reussir_syntax::kind::TokenKey>,
        ty: Option<&(surface::Type, bool)>,
        value: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let (value, var_ty) = match ty {
            Some((ty, flex)) => {
                let expected = self.eval_type_flex(ty, *flex);
                (self.check_expr(value, expected), expected)
            }
            None => {
                let h = self.infer_expr(value);
                let t = h.ty;
                (h, t)
            }
        };
        let name_span = Some(name.span());
        let var = self.vars.fresh(name.value, var_ty, name_span);
        let unit = self.tcx.mk_unit();
        self.mk_expr(
            ExprKind::Let {
                var,
                name: name.value,
                span: name_span,
                value: Box::new(value),
            },
            unit,
            span,
        )
    }

    /// Binary operators synthesize:
    /// (ARITH) `op∈{+,-,*,/,%}`: `l ⇒ T`, `Num ⊳ T`, `r ⇐ T`, result `T`.
    /// (LOGIC) `op∈{&&,||}`: `l ⇐ Bool`, `r ⇐ Bool`, result `Bool`.
    /// (CMP) comparisons: `l ⇒ T`, `r ⇐ T`, result `Bool`; ordering comparisons also
    /// require `Num ⊳ T` (equality/inequality do not).
    fn infer_binop(
        &mut self,
        op: BinOp,
        l: &surface::Expr,
        r: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                let l = self.infer_expr(l);
                let ty = l.ty;
                self.register_bound(self.builtins.num, ty, span);
                let r = self.check_expr(r, ty);
                let aop = arith_op(op);
                self.mk_expr(ExprKind::Arith(Box::new(l), aop, Box::new(r)), ty, span)
            }
            BinOp::And | BinOp::Or => {
                let bool_ty = self.tcx.mk_bool();
                let l = self.check_expr(l, bool_ty);
                let r = self.check_expr(r, bool_ty);
                let aop = arith_op(op);
                self.mk_expr(
                    ExprKind::Arith(Box::new(l), aop, Box::new(r)),
                    bool_ty,
                    span,
                )
            }
            _ => {
                let l = self.infer_expr(l);
                let r = self.check_expr(r, l.ty);
                if matches!(op, BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte) {
                    self.register_bound(self.builtins.num, l.ty, span);
                }
                let bool_ty = self.tcx.mk_bool();
                let cop = cmp_op(op);
                self.mk_expr(ExprKind::Cmp(Box::new(l), cop, Box::new(r)), bool_ty, span)
            }
        }
    }

    /// Unary operators synthesize: (NEG) `-e ⇒ (Negate he : T)` with `e ⇒ T` and
    /// `Num ⊳ T`; (NOT) `!e ⇒ (Not he : Bool)` with `e ⇐ Bool`.
    fn infer_unop(&mut self, op: UnaryOp, e: &surface::Expr, span: Option<Span>) -> Expr<'tcx> {
        match op {
            UnaryOp::Negate => {
                let e = self.infer_expr(e);
                let ty = e.ty;
                self.register_bound(self.builtins.num, ty, span);
                self.mk_expr(ExprKind::Negate(Box::new(e)), ty, span)
            }
            UnaryOp::Not => {
                let bool_ty = self.tcx.mk_bool();
                let e = self.check_expr(e, bool_ty);
                self.mk_expr(ExprKind::Not(Box::new(e)), bool_ty, span)
            }
        }
    }

    /// (CAST) `Γ ⊢ (e as A) ⇒ (Cast(he, Ttgt) : Ttgt)` where `Ttgt = eval(A)`: `e ⇒ _`
    /// and both source and `Ttgt` carry `Num`; an unconstrained source hole is pinned
    /// to `Ttgt` (the cast doubles as an annotation).
    fn infer_cast(
        &mut self,
        ty: &surface::Type,
        e: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let e = self.infer_expr(e);
        let target = self.eval_type(ty);

        // A cast is only legal between numerics: both the source and the target
        // must satisfy `Num`. Registering a `Num` obligation on each means a
        // concrete non-numeric operand fails in the fulfillment loop with a
        // clear diagnostic, instead of surviving to a lowering-time crash.
        let src_ty = self.infer.shallow_resolve(e.ty);
        self.register_bound(self.builtins.num, target, span);
        self.register_bound(self.builtins.num, src_ty, span);

        // When the source is still an unconstrained hole and the target is a
        // concrete numeric type, treat the cast as a type annotation and pin the
        // source to the target. This both resolves the literal's type and avoids
        // emitting a degenerate same-type conversion.
        if matches!(src_ty.kind(), TyKind::Hole(_)) && !matches!(target.kind(), TyKind::Hole(_)) {
            // Unification of a head hole with a concrete numeric type cannot
            // fail here; ignore the (impossible) mismatch rather than panic.
            let _ = self.infer.unify(src_ty, target);
        }

        self.mk_expr(ExprKind::Cast(Box::new(e), target), target, span)
    }

    /// (CALL) `Γ ⊢ name⟨ty_args⟩(args) ⇒ (FuncCall{…} : R)`: resolve `name` to a
    /// function, instantiate its generics to a substitution θ, check `argᵢ ⇐
    /// ⌈paramᵢ⌉θ`, and return `R = ⌈return_ty⌉θ`. A regional callee outside a region,
    /// an unknown name, or an arity mismatch is a diagnostic (unknown name ⇒ poison).
    /// A bare `name` that binds a local closure value is redirected to a closure
    /// application (`closure_apply`) — the parser cannot tell the two apart.
    fn infer_func_call(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        let fname = self.sym(fc.name.basename);
        // The parser emits a `FuncCall` for every bare-name application `f(x)` —
        // it cannot tell a `fn` from a closure-valued local. A name that binds a
        // local value is a closure application (a local binding shadows a
        // same-named function, matching `infer_var`'s var-first precedence).
        if fc.name.segments.is_empty()
            && let Some((id, ty)) = self.vars.lookup(fc.name.basename)
        {
            self.record_use(fc.name.basename);
            if !fc.ty_args.is_empty() {
                self.error(
                    span,
                    format!("`{fname}` is a local value and cannot take type arguments"),
                );
            }
            let callee = self.mk_expr(ExprKind::Var(id), ty, span);
            return self.closure_apply(callee, &fc.args, span);
        }
        let Some(def) = self.resolve_function_ref(&fc.name) else {
            let hint = if fc.name.segments.is_empty() {
                self.function_suggestion(fc.name.basename)
            } else {
                String::new()
            };
            let shown = self.path_display(&fc.name);
            self.error(span, format!("unknown function `{shown}`{hint}"));
            return self.poison(span);
        };
        self.record_use(fc.name.basename);
        let proto = self.functions[&def].clone();
        if proto.is_regional && !self.inside_region {
            self.error(span, "cannot call a regional function outside of a region");
        }
        let inst = self.instantiate(&proto.generics, &fc.ty_args, span);

        if fc.args.len() != proto.params.len() {
            self.error(
                span,
                format!(
                    "`{fname}` expects {} argument(s), got {}",
                    proto.params.len(),
                    fc.args.len()
                ),
            );
        }
        let mut args = Vec::new();
        for (arg, (_, pty)) in fc.args.iter().zip(&proto.params) {
            let expected = self.infer.instantiate_ty(*pty, &inst);
            args.push(self.check_expr(arg, expected));
        }
        let ty_args = self.inst_args(&proto.generics, &inst);
        let result = self.infer.instantiate_ty(proto.return_ty, &inst);
        self.mk_expr(
            ExprKind::FuncCall {
                target: def,
                ty_args,
                args,
                regional: proto.is_regional,
            },
            result,
            span,
        )
    }

    /// (APP) `Γ ⊢ callee(args) ⇒ (ClosureCall{hc,[hᵢ]} : R)`: synthesize the callee to
    /// a `Closure{⟨Pᵢ⟩, R}` and check `argᵢ ⇐ Pᵢ`. Fewer args than params yields a
    /// residual `Closure` over the remaining params (partial application); more args is
    /// a diagnostic; a non-closure callee is (APP-ERR) → poison.
    fn infer_closure_call(
        &mut self,
        callee: &surface::Expr,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let callee = self.infer_expr(callee);
        self.closure_apply(callee, args, span)
    }

    /// Apply an already-synthesized `callee` (a closure value) to `args`, shared
    /// by [`infer_closure_call`](Self::infer_closure_call) — where the callee is
    /// a first-class expression — and [`infer_func_call`](Self::infer_func_call),
    /// where a bare name turned out to be a local closure binding rather than a
    /// function. See `infer_closure_call` for the partial/full-application rules.
    fn closure_apply(
        &mut self,
        callee: Expr<'tcx>,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let (params, ret) = match callee.ty.kind() {
            TyKind::Closure { params, ret } => (params.to_vec(), *ret),
            _ => {
                self.error(span, "called value is not a closure");
                return self.poison(span);
            }
        };
        // A closure may be applied to fewer arguments than it has parameters:
        // that is a *partial* application, which lowers to one `closure.apply`
        // per supplied argument (no `closure.eval`) and yields a residual closure
        // over the remaining parameters. A *full* application — one argument per
        // parameter — appends the final `closure.eval` and has the return type.
        // Supplying more arguments than parameters is an error: the result of a
        // full application is not itself a closure to apply the extras to.
        let n = args.len();
        if n > params.len() {
            self.error(
                span,
                format!(
                    "closure applied to {} argument(s) but takes at most {}",
                    n,
                    params.len()
                ),
            );
        }
        let mut out = Vec::new();
        for (arg, pty) in args.iter().zip(&params) {
            out.push(self.check_expr(arg, *pty));
        }
        // Full (or over-) application is typed as the return type; a partial
        // application is typed as a closure over the not-yet-supplied parameters.
        let result = if n >= params.len() {
            ret
        } else {
            self.tcx.mk_closure(&params[n..], ret)
        };
        self.mk_expr(
            ExprKind::ClosureCall {
                target: Box::new(callee),
                args: out,
            },
            result,
            span,
        )
    }

    /// (REGION) `Γ ⊢ region { body } ⇒ (RegionRun(hb) : freeze(T))`: set
    /// `inside_region` for the body (`body ⇒ T`), restore it, and stamp the node with
    /// the head-only `freeze_region(T)` (Regional|Flex → Rigid). A nested region is a
    /// diagnostic. This is the only rule that rewrites the synthesized type.
    fn infer_region(&mut self, body: &surface::Expr, span: Option<Span>) -> Expr<'tcx> {
        if self.inside_region {
            self.error(span, "cannot create a nested region");
        }
        let saved = self.inside_region;
        self.inside_region = true;
        let body = self.infer_expr(body);
        self.inside_region = saved;
        let frozen = self.freeze_region(body.ty);
        self.mk_expr(ExprKind::RegionRun(Box::new(body)), frozen, span)
    }

    /// Whether `ty`'s head (peeling `Nullable`) is a flex regional record — a
    /// value that cannot be materialized, hence cannot escape its region.
    pub(super) fn is_flex(&mut self, ty: Ty<'tcx>) -> bool {
        match self.infer.shallow_resolve(ty).kind() {
            TyKind::Record {
                flex: crate::semi::ty::Flexivity::Flex,
                ..
            } => true,
            TyKind::Nullable(inner) => self.is_flex(*inner),
            _ => false,
        }
    }

    /// (LAM) `Γ ⊢ |params…| body ⇒ (Closure{captures,params,hb} : Closure{⟨Pₖ⟩, hb.ty})`:
    /// each param type is its annotation or a fresh hole; the body is checked against an
    /// annotated return type or synthesized; captures are the body's free variables
    /// minus the params. Capturing or returning a flex value is a diagnostic (a flex
    /// value cannot escape its region).
    fn infer_lambda(&mut self, lam: &surface::Lambda, span: Option<Span>) -> Expr<'tcx> {
        let mark = self.vars.mark();
        let mut params = Vec::new();
        for (name, ty) in &lam.args {
            let pty = match ty {
                Some(t) => self.eval_type(t),
                None => self.infer.new_hole_ty(),
            };
            let var = self.vars.fresh(*name, pty, None);
            params.push((var, pty));
        }
        let body = match &lam.ret_ty {
            Some(rt) => {
                let expected = self.eval_type(rt);
                self.check_expr(&lam.body, expected)
            }
            None => self.infer_expr(&lam.body),
        };
        self.vars.restore(mark);

        // Captures are the free variables of the body that are not parameters.
        let param_ids: Vec<VarId> = params.iter().map(|(v, _)| *v).collect();
        let mut free = Vec::new();
        free_vars(&body, &mut free);
        let captures: Vec<(VarId, Ty<'tcx>)> = free
            .into_iter()
            .filter(|v| !param_ids.contains(v))
            .map(|v| (v, self.vars.def(v).ty))
            .collect();

        // A flex value cannot be materialized, so it cannot escape its region by
        // being captured in a closure (the closure may outlive the region).
        for &(v, ty) in &captures {
            if self.is_flex(ty) {
                let name = self.sym(self.vars.def(v).name);
                self.error(
                    span,
                    format!(
                        "closure cannot capture `{name}`: a flex value cannot escape its region"
                    ),
                );
            }
        }

        // A flex value cannot be materialized, so it cannot escape the region by
        // being *returned* from a closure either: the closure is a shared,
        // possibly region-outliving object, so a flex result would hand a
        // non-materializable value out of its region.
        if self.is_flex(body.ty) {
            self.error(
                span,
                "closure cannot return a flex value: a flex value cannot escape its region",
            );
        }

        let param_tys: Vec<Ty<'tcx>> = params.iter().map(|(_, t)| *t).collect();
        let ty = self.tcx.mk_closure(&param_tys, body.ty);
        self.mk_expr(
            ExprKind::Closure(ClosureExpr {
                captures,
                params,
                body: Box::new(body),
            }),
            ty,
            span,
        )
    }

    /// (ASSIGN) `Γ ⊢ dst.acc = src ⇒ (Assign(hd,idx,hs) : Unit)`: `dst ⇒ T` must be a
    /// `Flex` record with a *mutable* field `acc` of type `Tf`; `src ⇐ Tf`. A non-flex
    /// target, missing field, or immutable field is a diagnostic.
    fn infer_assign(
        &mut self,
        dst: &surface::Expr,
        acc: &surface::Access,
        src: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let dst = self.infer_expr(dst);
        let dty = self.infer.shallow_resolve(dst.ty);
        let TyKind::Record {
            def,
            args,
            flex: crate::semi::ty::Flexivity::Flex,
        } = dty.kind()
        else {
            self.error(span, "assignment target must be a flex record");
            return self.poison(span);
        };
        // The target was just checked to be `Flex`, so the link's slot is seen
        // at its flex (writable) coloring.
        let Some((idx, field_ty, mutable)) = self.resolve_field(*def, args, Flexivity::Flex, acc)
        else {
            self.error(span, "no such field on assignment target");
            return self.poison(span);
        };
        if !mutable {
            self.error(span, "cannot assign to an immutable field");
        }
        // A mutable field's type is already the nullable link type.
        let src = self.check_expr(src, field_ty);
        // The assigned value must be a `Nullable<R>` whose element `R` is a
        // regional record — only a regional value belongs in a mutable link. Its
        // *flexivity* (a frozen `rigid` value cannot go into a flex `[field]`
        // link) is enforced by the checking boundary above: `check_expr` unified
        // `src` against the link's flex slot type. If `R` is a generic, record the
        // requirement for the monomorphization call-boundary check (the same
        // `regional_generics` channel as a `[flex] T` parameter).
        let resolved = self.infer.shallow_resolve(src.ty);
        if let TyKind::Nullable(inner) = resolved.kind() {
            let inner = self.infer.shallow_resolve(*inner);
            match inner.kind() {
                TyKind::Generic(g) => {
                    if !self.regional_generics.contains(g) {
                        self.regional_generics.push(*g);
                    }
                }
                _ if matches!(
                    inner.flexivity(),
                    Some(
                        crate::semi::ty::Flexivity::Regional
                            | crate::semi::ty::Flexivity::Flex
                            | crate::semi::ty::Flexivity::Rigid
                    )
                ) => {}
                _ => self.error(
                    span,
                    "assignment source must be a `Nullable` of a regional record",
                ),
            }
        } else {
            self.error(
                span,
                "assignment source must be a `Nullable` of a regional record",
            );
        }
        let unit = self.tcx.mk_unit();
        self.mk_expr(
            ExprKind::Assign(Box::new(dst), idx, Box::new(src)),
            unit,
            span,
        )
    }

    /// (PROJ) `Γ ⊢ base.acc₁.….accₙ ⇒ (Proj(hb,[idxᵢ]) : Tₙ)`: `base ⇒ T₀`, then at each
    /// step the (shallow-resolved) record type's field `accᵢ` gives the next type `Tᵢ`
    /// and its numeric index. A non-record head or missing field poisons.
    fn infer_access(
        &mut self,
        base: &surface::Expr,
        accs: &[surface::Access],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let base = self.infer_expr(base);
        let mut cur = self.infer.shallow_resolve(base.ty);
        let mut indices = Vec::new();
        for acc in accs {
            let TyKind::Record { def, args, flex } = cur.kind() else {
                let shown = self.infer.resolve(cur);
                self.error(
                    span,
                    format!("cannot access a field of `{}`", self.ty_display(shown)),
                );
                return self.poison(span);
            };
            let Some((idx, field_ty, _)) = self.resolve_field(*def, args, *flex, acc) else {
                self.error(span, "no such field");
                return self.poison(span);
            };
            indices.push(idx);
            cur = self.infer.shallow_resolve(field_ty);
        }
        self.mk_expr(ExprKind::Proj(Box::new(base), indices), cur, span)
    }

    /// Resolve an access on a record type to `(field index, field type, mutable)`.
    /// The field type is instantiated with the record's type arguments and
    /// colored as seen through a base of flexivity `base_flex`
    /// ([`Self::field_member_ty`]).
    fn resolve_field(
        &mut self,
        record_def: DefId,
        args: &[Ty<'tcx>],
        base_flex: Flexivity,
        acc: &surface::Access,
    ) -> Option<(u32, Ty<'tcx>, bool)> {
        let record = self.records.get(&record_def)?;
        let ty_params = record.ty_params.clone();
        let fields = record.fields.clone()?;
        let inst =
            Instantiation::from_pairs(ty_params.iter().map(|(_, g)| *g).zip(args.iter().copied()));
        let (idx, decl_ty, mutable) = match (&fields, acc) {
            (RecordFields::Named(fs), surface::Access::Named(name)) => {
                let i = fs.iter().position(|(n, _, _)| n == name)?;
                (i, fs[i].1, fs[i].2)
            }
            (RecordFields::Named(fs), surface::Access::Unnamed(n)) => {
                let i = *n as usize;
                let f = fs.get(i)?;
                (i, f.1, f.2)
            }
            (RecordFields::Unnamed(fs), surface::Access::Unnamed(n)) => {
                let i = *n as usize;
                let f = fs.get(i)?;
                (i, f.0, f.1)
            }
            _ => return None,
        };
        let field_ty = self.field_member_ty(decl_ty, mutable, base_flex, &inst);
        Some((idx as u32, field_ty, mutable))
    }

    // ----- constructors -----

    /// A constructor-call expression; see [`Self::infer_ctor`] for the dispatch rules.
    fn infer_ctor_call(&mut self, cc: &surface::CtorCall, span: Option<Span>) -> Expr<'tcx> {
        self.infer_ctor(&cc.name, &cc.ty_args, &cc.args, span)
    }

    /// Dispatch a constructor path: `Nullable::…` → [`Self::infer_nullable`]; a
    /// qualified path whose qualifier resolves to an enum (`Enum::Variant`,
    /// `m::Enum::Variant`) → [`Self::infer_variant`]; otherwise a struct —
    /// bare (`Foo`) or module-qualified (`m::Foo`) — → [`Self::infer_struct_def`].
    fn infer_ctor(
        &mut self,
        path: &surface::Path,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let qualifier = path.segments.last().copied();
        // The built-in nullable constructors.
        if qualifier.map(|k| self.sym(k)) == Some("Nullable") {
            return self.infer_nullable(self.sym(path.basename), args, span);
        }
        if let Some(enum_key) = qualifier {
            // The qualifier resolving to an enum wins (`Enum::Variant`); a
            // qualifier that is a module path instead falls through to a
            // module-qualified struct constructor (`m::Foo{…}`).
            if let Some(def) = self.resolve_ctor_qualifier(path)
                && matches!(self.records[&def].fields, Some(RecordFields::Variants(_)))
            {
                self.record_use(enum_key);
                return self.infer_variant(def, enum_key, path.basename, ty_args, args, span);
            }
            let Some(def) = self.resolve_record_ref(path) else {
                let hint = self.record_suggestion(enum_key);
                let shown = self.path_display(path);
                self.error(span, format!("unknown constructor `{shown}`{hint}"));
                return self.poison(span);
            };
            self.record_use(path.basename);
            return self.infer_struct_def(def, path.basename, ty_args, args, span);
        }
        // A bare struct constructor.
        self.infer_struct(path.basename, ty_args, args, span)
    }

    /// (NULL) the built-in nullable constructors: `Nullable::NonNull(e)` with `e ⇒ T`
    /// synthesizes `NullableCall(Some) : Nullable T`; `Nullable::Null` synthesizes
    /// `NullableCall(None) : Nullable α` for a fresh hole `α`.
    fn infer_nullable(
        &mut self,
        variant: &str,
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        match variant {
            "NonNull" => {
                let inner = match args {
                    [(_, e)] => self.infer_expr(e),
                    _ => {
                        self.error(span, "`Nullable::NonNull` takes one argument");
                        return self.poison(span);
                    }
                };
                let ty = self.tcx.mk_nullable(inner.ty);
                self.mk_expr(ExprKind::NullableCall(Some(Box::new(inner))), ty, span)
            }
            "Null" => {
                let hole = self.infer.new_hole_ty();
                let ty = self.tcx.mk_nullable(hole);
                self.mk_expr(ExprKind::NullableCall(None), ty, span)
            }
            other => {
                self.error(span, format!("unknown nullable constructor `{other}`"));
                self.poison(span)
            }
        }
    }

    /// (STRUCT) `Γ ⊢ name⟨ty_args⟩{args} ⇒ (CompoundCall{…} : R)`: resolve the struct,
    /// instantiate its generics to θ, check each field argument against its instantiated
    /// field type, and return the record type `R`. Constructing a regional struct
    /// outside a region is a diagnostic.
    fn infer_struct(
        &mut self,
        name: TokenKey,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let Some(def) = self.defs.resolve_record(name) else {
            let hint = self.record_suggestion(name);
            self.error(span, format!("unknown type `{}`{hint}", self.sym(name)));
            return self.poison(span);
        };
        self.record_use(name);
        self.infer_struct_def(def, name, ty_args, args, span)
    }

    /// The resolved-def core of [`Self::infer_struct`] (shared with the
    /// module-qualified constructor path).
    fn infer_struct_def(
        &mut self,
        def: DefId,
        name: TokenKey,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let record = self.records[&def].clone();
        if matches!(record.fields, Some(RecordFields::Variants(_))) {
            self.error(
                span,
                format!("`{}` is an enum, not a struct", self.sym(name)),
            );
            return self.poison(span);
        }
        // A regional record holds region-owned (`[field]`) links, so it can only
        // be created where a region owns it. Constructing one outside a region
        // would force lowering to emit a region-less flex rc (rejected by the
        // backend) or let the value escape with no owner.
        if matches!(record.default_cap, super::ctxt::DefaultCap::Regional) && !self.inside_region {
            self.error(
                span,
                "cannot construct a regional record outside of a region",
            );
        }
        let inst = self.instantiate(&record.generics_as_decls(), ty_args, span);
        let field_tys = self.field_types(&record.fields, &inst);

        let checked = self.check_ctor_args(name, args, &field_tys, span);
        let ty_args = self.inst_args(&record.generics_as_decls(), &inst);
        let result = self.record_ty(&record, &inst);
        self.mk_expr(
            ExprKind::CompoundCall {
                target: def,
                ty_args,
                args: checked,
            },
            result,
            span,
        )
    }

    /// (VARIANT) `Γ ⊢ Enum::Variant⟨ty_args⟩(args) ⇒ (VariantCall{…} : R)`: resolve the
    /// enum and variant index, instantiate generics to θ, check each payload argument
    /// against its instantiated type, and return the enum type `R`. Constructing a
    /// regional enum outside a region is a diagnostic.
    fn infer_variant(
        &mut self,
        def: DefId,
        enum_name: TokenKey,
        variant: TokenKey,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let record = self.records[&def].clone();
        let Some(RecordFields::Variants(variants)) = &record.fields else {
            self.error(span, format!("`{}` is not an enum", self.sym(enum_name)));
            return self.poison(span);
        };
        // As for structs: a regional enum can only be constructed inside a
        // region that owns it.
        if matches!(record.default_cap, super::ctxt::DefaultCap::Regional) && !self.inside_region {
            self.error(span, "cannot construct a regional enum outside of a region");
        }
        let Some(vidx) = variants.iter().position(|v| v.name == variant) else {
            self.error(
                span,
                format!(
                    "`{}` has no variant `{}`",
                    self.sym(enum_name),
                    self.sym(variant)
                ),
            );
            return self.poison(span);
        };
        let payload = variants[vidx].fields.clone();
        let inst = self.instantiate(&record.generics_as_decls(), ty_args, span);
        let payload_tys: Vec<Ty<'tcx>> = payload
            .iter()
            .map(|t| self.infer.instantiate_ty(*t, &inst))
            .collect();

        let mut checked = Vec::new();
        for ((_, arg), pty) in args.iter().zip(&payload_tys) {
            checked.push(self.check_expr(arg, *pty));
        }
        let ty_args = self.inst_args(&record.generics_as_decls(), &inst);
        let result = self.record_ty(&record, &inst);
        self.mk_expr(
            ExprKind::VariantCall {
                target: def,
                ty_args,
                variant: vidx,
                args: checked,
            },
            result,
            span,
        )
    }

    fn check_ctor_args(
        &mut self,
        name: TokenKey,
        args: &[(Option<TokenKey>, surface::Expr)],
        field_tys: &[(Option<TokenKey>, Ty<'tcx>)],
        span: Option<Span>,
    ) -> Vec<Expr<'tcx>> {
        if args.len() != field_tys.len() {
            self.error(
                span,
                format!(
                    "`{}` expects {} field(s), got {}",
                    self.sym(name),
                    field_tys.len(),
                    args.len()
                ),
            );
        }
        // Named arguments are matched by field name (an interned key compare);
        // positional by order. Field shorthand auto-forwards: a bare in-scope
        // variable whose name matches a field acts as that named argument, so
        // `Point { y, x }` is order-independent and composes with explicit
        // named arguments (`Point { x, y: 10 }`).
        let arg_fields: Vec<Option<TokenKey>> = args
            .iter()
            .map(|(f, e)| {
                f.or_else(|| match e.kind() {
                    surface::ExprKind::Var(path)
                        if path.segments.is_empty()
                            && field_tys
                                .iter()
                                .any(|(fname, _)| *fname == Some(path.basename))
                            && self.vars.lookup(path.basename).is_some() =>
                    {
                        Some(path.basename)
                    }
                    _ => None,
                })
            })
            .collect();
        let named = arg_fields.iter().any(|f| f.is_some());
        let mut out = Vec::new();
        if named {
            for (fname, fty) in field_tys {
                let Some(fname) = fname else { continue };
                match arg_fields.iter().position(|af| *af == Some(*fname)) {
                    Some(i) => out.push(self.check_expr(&args[i].1, *fty)),
                    None => {
                        self.error(span, format!("missing field `{}`", self.sym(*fname)));
                        out.push(self.poison(span));
                    }
                }
            }
        } else {
            for ((_, e), (_, fty)) in args.iter().zip(field_tys) {
                out.push(self.check_expr(e, *fty));
            }
        }
        out
    }

    fn field_types(
        &mut self,
        fields: &Option<RecordFields<'tcx>>,
        inst: &Instantiation<'tcx>,
    ) -> Vec<(Option<TokenKey>, Ty<'tcx>)> {
        // Construction site: a freshly-built record is born flex, so its
        // `[field]` link slots are seen at their writable (flex) coloring.
        let base_flex = Flexivity::Flex;
        match fields {
            Some(RecordFields::Named(fs)) => fs
                .iter()
                .map(|(n, t, is_field)| {
                    (
                        Some(*n),
                        self.field_member_ty(*t, *is_field, base_flex, inst),
                    )
                })
                .collect(),
            Some(RecordFields::Unnamed(fs)) => fs
                .iter()
                .map(|(t, is_field)| (None, self.field_member_ty(*t, *is_field, base_flex, inst)))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The type of a struct member as seen through a base of flexivity
    /// `base_flex` — the target a field argument or assignment source is
    /// checked against, and the type a projection reads out. A plain member
    /// is its (instantiated) declared type, except a plain *regional-record*
    /// member, whose coloring refines to `Rigid` ([`Self::rigid_member_ty`]:
    /// the slot holds a frozen value). A mutable `[field]` link — always a
    /// `Nullable<Record>` — takes the base's view ([`Self::field_link_ty`]):
    /// `Flex` (writable) through a flex base, `Rigid` (a frozen view)
    /// through anything else. Construction and assignment sites pass `Flex`
    /// (a fresh record is born flex; assignment requires a flex target), so
    /// the flex expectation there is what colors an otherwise-unknown value
    /// stored through the link — e.g. a standalone `Nullable::Null` has an
    /// unconstrained element, and checking it against the flex slot solves
    /// that hole to `flex` rather than the record's default `Regional`. (A
    /// concrete `rigid` value meeting the flex slot is a flexivity mismatch,
    /// caught by [`expect`].)
    fn field_member_ty(
        &mut self,
        decl_ty: Ty<'tcx>,
        is_field: bool,
        base_flex: Flexivity,
        inst: &Instantiation<'tcx>,
    ) -> Ty<'tcx> {
        let ty = self.infer.instantiate_ty(decl_ty, inst);
        if is_field {
            self.field_link_ty(ty, base_flex)
        } else {
            self.rigid_member_ty(ty)
        }
    }

    /// Color a non-`[field]` regional-record member `Rigid` — the counterpart
    /// of [`Self::flex_link_ty`]. Such a member holds an already-*frozen*
    /// value: it can only be written at construction (there is no assignment
    /// path to it), and the backend types its slot `rc<_, rigid>` on both
    /// sides (`getProjectedType`). Refining here makes both uses line up:
    /// a constructor argument is checked against the `Rigid` expectation
    /// (storing a still-live `Flex` value is a flexivity mismatch, caught by
    /// [`Self::expect`] — freeze it in its own `regional { }` first), and a
    /// projection reads a `Rigid` value out. Members with a concrete coloring
    /// and non-regional types pass through untouched.
    fn rigid_member_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        use crate::semi::ty::Flexivity;
        if let TyKind::Record {
            def,
            args,
            flex: Flexivity::Regional,
        } = ty.kind()
        {
            return self.tcx.mk_record(*def, args, Flexivity::Rigid);
        }
        ty
    }

    /// Color a `[field]` link's slot element with the base's view: `Flex`
    /// (writable) through a flex base, `Rigid` through anything else — after
    /// the region freezes, `x.f` on the rigid `x` reads a frozen
    /// `Nullable<rigid>` view (mirroring the dialect's `getProjectedType`,
    /// where only a flex reference projects a flex link). The link is a
    /// `Nullable<Record>`; this refines a concrete pointee record's coloring
    /// and leaves a generic (or any other) element untouched (a `[flex] T`
    /// requirement is tracked through the `regional_generics` channel
    /// instead).
    fn field_link_ty(&mut self, ty: Ty<'tcx>, base_flex: Flexivity) -> Ty<'tcx> {
        use crate::semi::ty::Flexivity;
        if let TyKind::Nullable(inner) = ty.kind()
            && let TyKind::Record { def, args, .. } = self.infer.shallow_resolve(*inner).kind()
        {
            let colored = if base_flex == Flexivity::Flex {
                Flexivity::Flex
            } else {
                Flexivity::Rigid
            };
            let element = self.tcx.mk_record(*def, args, colored);
            return self.tcx.mk_nullable(element);
        }
        ty
    }

    fn record_ty(
        &mut self,
        record: &super::ctxt::Record<'tcx>,
        inst: &Instantiation<'tcx>,
    ) -> Ty<'tcx> {
        let args: Vec<Ty<'tcx>> = record
            .ty_params
            .iter()
            .map(|(_, g)| inst.get(*g).unwrap_or_else(|| self.tcx.mk_generic(*g)))
            .collect();
        // A freshly-constructed regional record is a live, region-local value, so
        // it is born `Flex` (mutable, not yet frozen) — matching the reference
        // (`Tyck.hs`). This is what lets the result be assigned into or frozen on
        // region exit; binding it under a `[rigid]`/`[flex]` annotation just
        // re-reads the head (`unify` ignores flexivity). Construction outside a
        // region is reported separately by the caller.
        let flex = match record.default_cap {
            super::ctxt::DefaultCap::Regional => crate::semi::ty::Flexivity::Flex,
            _ => crate::semi::ty::Flexivity::Irrelevant,
        };
        self.tcx.mk_record(record.def, &args, flex)
    }

    // ----- instantiation helpers -----

    /// Build an instantiation for a list of generics, using explicit type
    /// arguments where given and fresh holes otherwise, registering each
    /// generic's bounds as obligations on the chosen type.
    pub(super) fn instantiate(
        &mut self,
        generics: &[(TokenKey, GenericId)],
        ty_args: &[Option<surface::Type>],
        span: Option<Span>,
    ) -> Instantiation<'tcx> {
        let mut pairs = Vec::new();
        let mut obligations: Vec<(TraitId, Ty<'tcx>)> = Vec::new();
        for (i, (_, gid)) in generics.iter().enumerate() {
            let bounds: Vec<TraitId> = self.generic_bounds(*gid).to_vec();
            let ty = match ty_args.get(i).and_then(Option::as_ref) {
                Some(t) => self.eval_type(t),
                None => self.infer.new_hole_ty(),
            };
            for b in bounds {
                obligations.push((b, ty));
            }
            pairs.push((*gid, ty));
        }
        for (b, ty) in obligations {
            self.register_bound(b, ty, span);
        }
        Instantiation::from_pairs(pairs)
    }

    fn inst_args(
        &mut self,
        generics: &[(TokenKey, GenericId)],
        inst: &Instantiation<'tcx>,
    ) -> Vec<Ty<'tcx>> {
        generics
            .iter()
            .map(|(_, g)| inst.get(*g).unwrap_or_else(|| self.tcx.mk_generic(*g)))
            .collect()
    }

    // ----- zonking -----

    /// Resolve every type in an expression tree against the solved holes,
    /// reporting any residual hole as an ambiguity error.
    ///
    /// Zonking is the inference boundary: bidirectional checking legitimately
    /// leaves a hole unsolved when nothing constrains it (e.g. the element
    /// type of a bare `Nullable::Null`), so a survivor here is a *user* error
    /// asking for an annotation — never something later passes should see.
    /// Downstream of zonking, HIR types are hole-free or errors were
    /// reported; monomorphization's `subst_ty` panic on a hole is a genuine
    /// ICE, not a reachable diagnostic.
    pub(super) fn zonk_expr(&mut self, mut e: Expr<'tcx>) -> Expr<'tcx> {
        // Children first: a hole shared along a spine (`if c { Nullable::Null }
        // else { x }`) is reported at the innermost expression that exhibits
        // it — where the annotation belongs — and suppressed on the ancestors.
        e.kind = self.zonk_kind(e.kind, e.span);
        e.ty = self.zonk_ty(e.ty, e.span);
        e
    }

    /// Resolve one type; report its not-yet-reported holes against `span`
    /// (the expression the type belongs to). See [`Elaborator::zonk_expr`].
    fn zonk_ty(&mut self, ty: Ty<'tcx>, span: Option<Span>) -> Ty<'tcx> {
        let ty = self.infer.resolve(ty);
        if ty_has_hole(ty) {
            let mut holes = Vec::new();
            collect_holes(ty, &mut holes);
            let mut fresh = false;
            for hole in holes {
                fresh |= self.reported_holes.insert(hole);
            }
            if fresh {
                self.error(
                    span,
                    "cannot infer the type of this expression; add a type annotation",
                );
            }
        }
        ty
    }

    fn zonk_kind(&mut self, kind: ExprKind<'tcx>, span: Option<Span>) -> ExprKind<'tcx> {
        use ExprKind::*;
        let zb = |s: &mut Self, e: Box<Expr<'tcx>>| Box::new(s.zonk_expr(*e));
        match kind {
            Negate(e) => Negate(zb(self, e)),
            Not(e) => Not(zb(self, e)),
            Arith(l, op, r) => Arith(zb(self, l), op, zb(self, r)),
            Cmp(l, op, r) => Cmp(zb(self, l), op, zb(self, r)),
            Cast(e, t) => {
                let t = self.zonk_ty(t, span);
                Cast(zb(self, e), t)
            }
            If(c, t, f) => If(zb(self, c), zb(self, t), zb(self, f)),
            RegionRun(e) => RegionRun(zb(self, e)),
            Proj(e, idx) => Proj(zb(self, e), idx),
            Assign(d, i, s) => Assign(zb(self, d), i, zb(self, s)),
            Let {
                var,
                name,
                span,
                value,
            } => Let {
                var,
                name,
                span,
                value: zb(self, value),
            },
            Seq(es) => Seq(es.into_iter().map(|e| self.zonk_expr(e)).collect()),
            FuncCall {
                target,
                ty_args,
                args,
                regional,
            } => FuncCall {
                target,
                ty_args: ty_args.into_iter().map(|t| self.zonk_ty(t, span)).collect(),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
                regional,
            },
            CompoundCall {
                target,
                ty_args,
                args,
            } => CompoundCall {
                target,
                ty_args: ty_args.into_iter().map(|t| self.zonk_ty(t, span)).collect(),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => VariantCall {
                target,
                ty_args: ty_args.into_iter().map(|t| self.zonk_ty(t, span)).collect(),
                variant,
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            NullableCall(e) => NullableCall(e.map(|e| zb(self, e))),
            ClosureCall { target, args } => ClosureCall {
                target: zb(self, target),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            Closure(c) => Closure(ClosureExpr {
                captures: c
                    .captures
                    .into_iter()
                    .map(|(v, t)| (v, self.zonk_ty(t, span)))
                    .collect(),
                params: c
                    .params
                    .into_iter()
                    .map(|(v, t)| (v, self.zonk_ty(t, span)))
                    .collect(),
                body: zb(self, c.body),
            }),
            Match(scrut, tree) => Match(zb(self, scrut), self.zonk_tree(tree)),
            other => other,
        }
    }
}

/// Map a surface arithmetic/boolean operator to the HIR form.
fn arith_op(op: BinOp) -> ArithOp {
    match op {
        BinOp::Add => ArithOp::Add,
        BinOp::Sub => ArithOp::Sub,
        BinOp::Mul => ArithOp::Mul,
        BinOp::Div => ArithOp::Div,
        BinOp::Mod => ArithOp::Mod,
        BinOp::And => ArithOp::And,
        BinOp::Or => ArithOp::Or,
        _ => unreachable!("not an arithmetic operator"),
    }
}

fn cmp_op(op: BinOp) -> CmpOp {
    match op {
        BinOp::Lt => CmpOp::Lt,
        BinOp::Gt => CmpOp::Gt,
        BinOp::Lte => CmpOp::Le,
        BinOp::Gte => CmpOp::Ge,
        BinOp::Equ => CmpOp::Eq,
        BinOp::Neq => CmpOp::Ne,
        _ => unreachable!("not a comparison operator"),
    }
}

/// Collect the free variables referenced in an expression tree.
fn free_vars<'tcx>(e: &Expr<'tcx>, out: &mut Vec<VarId>) {
    use ExprKind::*;
    let push = |out: &mut Vec<VarId>, v: VarId| {
        if !out.contains(&v) {
            out.push(v);
        }
    };
    match &e.kind {
        Var(v) => push(out, *v),
        Negate(e) | Not(e) | Cast(e, _) | RegionRun(e) | Proj(e, _) => free_vars(e, out),
        Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => {
            free_vars(l, out);
            free_vars(r, out);
        }
        If(c, t, f) => {
            free_vars(c, out);
            free_vars(t, out);
            free_vars(f, out);
        }
        Let { var, value, .. } => {
            free_vars(value, out);
            push(out, *var);
        }
        Seq(es) => es.iter().for_each(|e| free_vars(e, out)),
        FuncCall { args, .. } | CompoundCall { args, .. } | VariantCall { args, .. } => {
            args.iter().for_each(|e| free_vars(e, out))
        }
        NullableCall(e) => {
            if let Some(e) = e {
                free_vars(e, out)
            }
        }
        ClosureCall { target, args } => {
            free_vars(target, out);
            args.iter().for_each(|e| free_vars(e, out));
        }
        Closure(c) => free_vars(&c.body, out),
        Match(scrut, _) => free_vars(scrut, out),
        GlobalStr(_) | ConstChar(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Poison => {}
    }
}

impl<'tcx> super::ctxt::Record<'tcx> {
    /// The record's generics as `(name, id)` declaration pairs.
    fn generics_as_decls(&self) -> Vec<(TokenKey, GenericId)> {
        self.ty_params.clone()
    }
}
