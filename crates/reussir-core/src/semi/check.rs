//! The bidirectional type checker: surface expressions → typed Semi HIR.

use reussir_syntax::kind::TokenKey;

use crate::semi::infer::Instantiation;
use crate::semi::resolve::DefKind;
use crate::semi::traits::{Evidence, Obligation, SelectCtxt, SelectError, TraitId, TraitRef};
use crate::semi::ty::CellKind;
use crate::semi::ty::{DefId, Flexivity, GenericId, Ty, TyKind};
use crate::surface::{self, BinOp, Const, Span, UnaryOp};

use super::ctxt::{Elaborator, FuncProto, RecordFields};
use super::fulfill::{collect_holes, ty_has_hole};
use super::hir::{ArithOp, ClosureExpr, CmpOp, Expr, ExprKind, Function, VarId};

/// Why a field access failed to resolve: the field does not exist (or the
/// base is not a suitable record), or it exists but is private outside its
/// defining module. Callers keep the two diagnostics distinct, mirroring the
/// item-level "is private" vs "not found" split of extern resolution.
enum FieldErr {
    Missing,
    Private { field: String, record: String },
}

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
        // from many files/modules in one pass. An impl member's scope is the
        // module of the block that declared it (its path-derived parent is
        // the *type*, not a module), so it sees exactly what that module
        // sees — Rust's rule for package-level impls.
        self.enter_item_scope(def, proto.file);
        if let Some(module) = &proto.impl_module {
            self.defs.set_module(module.clone());
        }
        self.enter_function(&proto.generics);
        // Body type annotations in an impl member resolve `Self` to the
        // target; `enter_function` cleared the previous item's alias.
        self.self_alias = proto.self_ty;
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
            if let Some(message) = self.arc_reconciliation_error(e, f) {
                self.error(span, message);
                return;
            }
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

    /// The rich diagnostic for a mismatch that is *only* about the arc
    /// coloring: `Arc<T>` against `T` (possibly under matching `Nullable`
    /// wrappers). Plain "type mismatch" reads as noise here — the user either
    /// fed a bare value where the atomic world demands an arc'd one (the
    /// creation side) or is surprised that a field read through an `Arc` came
    /// out arc'd (the projection side) — so explain the §3.3 rule the way
    /// rustc explains lifetimes: what clashed, why the type is what it is,
    /// and what to do about it. Returns `None` for any other mismatch.
    fn arc_reconciliation_error(&mut self, expected: Ty<'tcx>, found: Ty<'tcx>) -> Option<String> {
        // Peel matching `Nullable` wrappers: a promoted `Nullable<Arc<T>>`
        // member mismatches a bare `Nullable<T>` the same way.
        let (mut e, mut f) = (expected, found);
        while let (TyKind::Nullable(ei), TyKind::Nullable(fi)) = (e.kind(), f.kind()) {
            (e, f) = (*ei, *fi);
        }
        // Exactly one side is arc'd, and underneath they are the same type.
        // Probe with a real unification so a still-unsolved hole on the bare
        // side (`List<_>` against `Arc<List<i32>>`) is recognized — and
        // solved, which also silences the "cannot infer" cascade the failed
        // outer unification would otherwise leave behind.
        let (inner, arc_is_expected) = match (e.kind(), f.kind()) {
            (TyKind::Arc(inner), _) if self.infer.unify(*inner, f).is_ok() => (*inner, true),
            (_, TyKind::Arc(inner)) if self.infer.unify(e, *inner).is_ok() => (*inner, false),
            _ => return None,
        };
        let inner = self.infer.resolve(inner);
        let shown = self.ty_display(inner);
        // Re-resolve both sides: the probe above may have just solved holes
        // the failed outer unification left behind (`List<_>` → `List<i32>`).
        let (expected, found) = (self.infer.resolve(expected), self.infer.resolve(found));
        let mut message = format!(
            "cannot reconcile thread-safety: expected `{}`, found `{}`",
            self.ty_display(expected),
            self.ty_display(found),
        );
        // When the type is genuinely recursive, the arc'd side usually exists
        // because of the SCC promotion — name the group and the rule.
        if let TyKind::Record { def, .. } = *inner.kind()
            && self.def_is_recursive(def)
        {
            let group = self.arc_recursive_group(def);
            let mut names: Vec<&str> = group
                .iter()
                .filter_map(|d| self.records.get(d).map(|r| self.sym(r.name)))
                .collect();
            names.sort_unstable();
            message.push_str(&format!(
                "\nnote: fields of the recursive group `{}` are promoted to \
                 `Arc` inside an arc'd box",
                names.join("`, `"),
            ));
        }
        if arc_is_expected {
            message.push_str(&format!(
                "\nnote: a plain value is never re-colored; construct it as \
                 `Arc<{shown}>::…` from the start",
            ));
        } else {
            message.push_str("\nnote: a value read out of an arc'd box keeps its `Arc` coloring");
        }
        Some(message)
    }

    /// Whether `def` sits on a genuine recursion: some chain of record member
    /// references starting from its members returns to it. (Unlike
    /// [`Elaborator::arc_recursive_group`], which always contains its root,
    /// this distinguishes a recursive record from a plain one.)
    fn def_is_recursive(&self, def: DefId) -> bool {
        use crate::semi::traits::sync::SyncEnv;
        let env = crate::semi::ty_eval::ElabSyncEnv { el: self };
        let mut stack = env.member_record_defs(def).unwrap_or_default();
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(next) = stack.pop() {
            if next == def {
                return true;
            }
            if seen.insert(next) {
                stack.extend(env.member_record_defs(next).unwrap_or_default());
            }
        }
        false
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
        // (LAM⇐) a literal lambda checks against an expected closure type:
        // unannotated parameters take the expected parameter types and the
        // body checks against the expected result, so lambdas in argument
        // position need no annotations.
        if let surface::ExprKind::Lambda(lam) = e.kind()
            && let TyKind::Closure { params, ret } = *self.infer.shallow_resolve(expected).kind()
            && params.len() == lam.args.len()
        {
            let h = self.lambda(&lam, Some((params, ret)), Some(e.span()));
            self.expect(h.ty, expected, h.span);
            return h;
        }
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
            surface::ExprKind::If(c, t, f) => self.infer_if(c, t, f.as_ref(), span),
            surface::ExprKind::Let(name, ty, value) => {
                self.infer_let(name, ty.as_ref(), value, span)
            }
            surface::ExprKind::BinOpExpr(op, l, r) => self.infer_binop(*op, l, r, span),
            surface::ExprKind::UnaryOpExpr(op, e) => self.infer_unop(*op, e, span),
            surface::ExprKind::Cast(ty, e) => self.infer_cast(ty, e, span),
            surface::ExprKind::FuncCallExpr(fc) => self.infer_func_call(fc, span),
            surface::ExprKind::CtorCallExpr(cc) => self.infer_ctor_call(cc, span),
            surface::ExprKind::CallExpr(callee, args) => {
                // `e.f(x)` is a postfix call: the method wins when one
                // exists, the closure-valued field otherwise. A
                // parenthesized callee — `(e.f)(x)` — opts out of method
                // dispatch and always applies the field.
                match callee.kind() {
                    surface::ExprKind::AccessChain(base, accs) if !callee.parenthesized() => {
                        self.infer_postfix_call(&base, &accs, args, span)
                    }
                    _ => self.infer_closure_call(callee, args, span),
                }
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
                self.register_bound(self.lang.integral, hole, span);
                self.mk_expr(ExprKind::ConstInt(self.tcx.alloc(i.clone())), hole, span)
            }
            Const::ConstFloat(f) => {
                let hole = self.infer.new_hole_ty();
                self.register_bound(self.lang.floating_point, hole, span);
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
            // A bare name that is not a local may be a `fn` used as a value:
            // lift it to a closure. A local binding shadows a same-named
            // function (checked first above), matching `infer_func_call`.
            if let Some(def) = self.resolve_function_ref(path) {
                self.record_use(path.basename);
                let proto = self.functions[&def].clone();
                let inst = self.instantiate(&proto.generics, &[], span);
                if self.reject_intrinsic_value(def, span) {
                    return self.poison(span);
                }
                return self.lift_function(def, &proto, &inst, span);
            }
            // A bare name bound by an import to a *qualified* path (e.g.
            // `import List::Nil;`) is a nullary-constructor reference.
            if let Some(expanded) = self.expand_path(path)
                && !expanded.segments.is_empty()
            {
                return self.infer_ctor(&expanded, &[], &[], span);
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
    ///
    /// An omitted else branch is sugar for an empty `else {}`: the elaborated
    /// HIR carries a synthesized empty unit block, so the whole expression is
    /// unit and the then-branch is checked against unit (`t ⇐ Unit`) — that
    /// way a non-unit then-branch is reported at the branch itself rather
    /// than at an else block the programmer never wrote.
    fn infer_if(
        &mut self,
        c: &surface::Expr,
        t: &surface::Expr,
        f: Option<&surface::Expr>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let bool_ty = self.tcx.mk_bool();
        let c = self.check_expr(c, bool_ty);
        let (t, f) = match f {
            Some(f) => {
                let t = self.infer_expr(t);
                let f = self.check_expr(f, t.ty);
                (t, f)
            }
            None => {
                let unit = self.tcx.mk_unit();
                let t = self.check_expr(t, unit);
                let f = self.mk_expr(ExprKind::Seq(Vec::new()), unit, span);
                (t, f)
            }
        };
        let result = t.ty;
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
    /// (BIT) `op∈{&,|,^,<<,>>}`: `l ⇒ T`, `Integral ⊳ T`, `r ⇐ T`, result `T`.
    /// (LOGIC) `op∈{&&,||}`: `l ⇐ Bool`, `r ⇐ Bool`, result `Bool`, and the
    /// pair desugars to (IF) so it short-circuits — `l && r` elaborates to
    /// `if l { r } else { false }` and `l || r` to `if l { true } else { r }`,
    /// putting the right operand in a branch that only runs when the left one
    /// leaves the answer open. The strict `ArithOp::And`/`Or` remain reachable
    /// from hand-written HIR/MIR text, where they mean the eager bitwise form.
    /// (CMP-SCALAR) comparisons with `l ⇒ T` for scalar-or-hole `T`: `r ⇐ T`, result
    /// `Bool`, lowered intrinsically; equality registers `PartialEq ⊳ T`, orderings
    /// `PartialOrd ⊳ T`.
    /// (CMP-DISPATCH) any other `T` desugars to the operator method of the active
    /// comparison trait: `l == r` ⇒ `PartialEq::eq(l, r)`, `l != r` ⇒ its negation,
    /// each ordering its own `PartialOrd` method — four methods, never an operand
    /// swap, so evaluation order is always source order.
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
                self.register_bound(self.lang.num, ty, span);
                let r = self.check_expr(r, ty);
                let aop = arith_op(op);
                self.mk_expr(ExprKind::Arith(Box::new(l), aop, Box::new(r)), ty, span)
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                // (BIT) like (ARITH) but integers only, `Integral ⊳ T`.
                // Both operands share `T` — a shift amount is the operand
                // type, not a separate width.
                let l = self.infer_expr(l);
                let ty = l.ty;
                self.register_bound(self.lang.integral, ty, span);
                let r = self.check_expr(r, ty);
                let aop = arith_op(op);
                self.mk_expr(ExprKind::Arith(Box::new(l), aop, Box::new(r)), ty, span)
            }
            BinOp::And | BinOp::Or => {
                // (LOGIC) is sugar for (IF), which is what makes it
                // short-circuiting: the right operand becomes a branch body,
                // so it is evaluated only when the left operand does not
                // already decide the result. Emitting `Arith` here instead —
                // `arith.andi`/`arith.ori` — would evaluate both sides, which
                // is a different language.
                let bool_ty = self.tcx.mk_bool();
                let l = self.check_expr(l, bool_ty);
                let r = self.check_expr(r, bool_ty);
                let shortcut =
                    self.mk_expr(ExprKind::ConstBool(matches!(op, BinOp::Or)), bool_ty, span);
                let (t, f) = match op {
                    // `l && r` ⇒ `if l { r } else { false }`
                    BinOp::And => (r, shortcut),
                    // `l || r` ⇒ `if l { true } else { r }`
                    _ => (shortcut, r),
                };
                self.mk_expr(
                    ExprKind::If(Box::new(l), Box::new(t), Box::new(f)),
                    bool_ty,
                    span,
                )
            }
            _ => {
                let l = self.infer_expr(l);
                let ty = self.infer.shallow_resolve(l.ty);
                if matches!(ty.kind(), TyKind::Hole(_)) {
                    // Inference: the operand is still an open hole — commit
                    // to the intrinsic form under a comparison obligation
                    // (defaulting or context solves the hole; a non-scalar
                    // solution is caught at monomorphization). Everything
                    // resolved dispatches through the trait: whether the
                    // comparison is intrinsic is the *impl*'s property, not
                    // a type test here.
                    let r = self.check_expr(r, l.ty);
                    self.register_cmp_bound(op, ty, span);
                    let bool_ty = self.tcx.mk_bool();
                    let cop = cmp_op(op);
                    self.mk_expr(ExprKind::Cmp(Box::new(l), cop, Box::new(r)), bool_ty, span)
                } else {
                    self.dispatch_cmp_operator(op, l, r, span)
                }
            }
        }
    }

    /// The active comparison-tower trait for `item`, through the lang
    /// registry (fallback or `core`-declared alike — both register).
    fn cmp_lang_trait(&self, item: crate::semi::lang::LangItem) -> Option<TraitId> {
        self.lang
            .get(item)
            .and_then(|def| self.traits.trait_by_def(def))
    }

    /// (CMP-SCALAR)'s obligation: `PartialEq ⊳ T` for equality,
    /// `PartialOrd ⊳ T` for orderings. Without a comparison tower (its
    /// root names were squatted before it could register), orderings
    /// degrade to the numeric constraint.
    fn register_cmp_bound(&mut self, op: BinOp, ty: Ty<'tcx>, span: Option<Span>) {
        use crate::semi::lang::LangItem;
        let ordering = matches!(op, BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte);
        let item = if ordering {
            LangItem::PartialOrd
        } else {
            LangItem::PartialEq
        };
        match self.cmp_lang_trait(item) {
            Some(tid) => self.register_bound(tid, ty, span),
            None if ordering => self.register_bound(self.lang.num, ty, span),
            None => {}
        }
    }

    /// (CMP-DISPATCH): desugar a comparison on a non-scalar operand into
    /// the operator method of the active comparison trait, resolved by
    /// name. The fallback traits are method-less, so operator overloading
    /// needs `core`'s declarations — dispatch on them reports as much.
    fn dispatch_cmp_operator(
        &mut self,
        op: BinOp,
        l: Expr<'tcx>,
        r: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        use crate::semi::lang::LangItem;
        let (item, name_idx, negate) = match op {
            BinOp::Equ => (LangItem::PartialEq, 0, false),
            BinOp::Neq => (LangItem::PartialEq, 0, true),
            BinOp::Lt => (LangItem::PartialOrd, 1, false),
            BinOp::Lte => (LangItem::PartialOrd, 2, false),
            BinOp::Gt => (LangItem::PartialOrd, 3, false),
            BinOp::Gte => (LangItem::PartialOrd, 4, false),
            _ => unreachable!("arith and logic ops are handled above"),
        };
        let sym = cmp_symbol(op);
        let shown_ty = self.ty_display(l.ty);
        let Some(tid) = self.cmp_lang_trait(item) else {
            self.error(
                span,
                format!(
                    "`{sym}` is not defined for `{shown_ty}`: the `{}` lang trait                      is not available in this session",
                    item.name()
                ),
            );
            return self.poison(span);
        };
        let name = self.cmp_method_names[name_idx];
        let methods = &self.traits.trait_def(tid).methods;
        let Some(midx) = methods.iter().position(|m| m.name == name) else {
            // The method-less fallback tower: selection decides. A
            // compiler impl (the scalars) grounds the intrinsic here; a
            // bound on a generic grounds it at monomorphization; no
            // evidence is the ordinary missing-impl diagnostic.
            let ty = self.infer.resolve(l.ty);
            let goal = Obligation::Trait(TraitRef {
                trait_id: tid,
                args: vec![ty],
            });
            let outcome = SelectCtxt::new(&self.traits, self.tcx, &self.generics).select(&goal);
            return match outcome {
                Ok(_) => {
                    let r = self.check_expr(r, l.ty);
                    let bool_ty = self.tcx.mk_bool();
                    let cop = cmp_op(op);
                    self.mk_expr(ExprKind::Cmp(Box::new(l), cop, Box::new(r)), bool_ty, span)
                }
                Err(SelectError::NoImpl(inner)) => {
                    let msg = self.no_impl_message(ty, tid, &inner);
                    self.error(span, msg);
                    self.poison(span)
                }
                Err(SelectError::DepthLimit(goal)) => {
                    let msg = self.depth_limit_message(&goal);
                    self.error(span, msg);
                    self.poison(span)
                }
            };
        };
        let call = self.dispatch_trait_method(tid, midx, l, std::slice::from_ref(r), span);
        if !negate {
            return call;
        }
        // `!=` is `!(l == r)`; `eq`'s conformance-checked signature returns
        // `Bool`, and an errored dispatch is already poison.
        let bool_ty = self.tcx.mk_bool();
        self.mk_expr(ExprKind::Not(Box::new(call)), bool_ty, span)
    }

    /// Unary operators synthesize: (NEG) `-e ⇒ (Negate he : T)` with `e ⇒ T` and
    /// `Num ⊳ T`; (NOT) `!e ⇒ (Not he : Bool)` with `e ⇐ Bool`.
    fn infer_unop(&mut self, op: UnaryOp, e: &surface::Expr, span: Option<Span>) -> Expr<'tcx> {
        match op {
            UnaryOp::Negate => {
                let e = self.infer_expr(e);
                let ty = e.ty;
                self.register_bound(self.lang.num, ty, span);
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
        //
        // `char` is its own rule outside `Num`: the value is a code point, so
        // a cast to any integer type is allowed (`as u8` truncates to the low
        // byte, wider targets zero-extend), but nothing casts *to* `char` —
        // an arbitrary integer is not necessarily a scalar value, and the FFI
        // boundary relies on every `char` being one.
        let src_ty = self.infer.shallow_resolve(e.ty);
        let target_head = self.infer.shallow_resolve(target);
        if matches!(target_head.kind(), TyKind::Char) {
            self.error(
                span,
                "cannot cast to `char`: not every integer is a valid code point",
            );
            return self.poison(span);
        }
        if matches!(src_ty.kind(), TyKind::Char) {
            if !matches!(target_head.kind(), TyKind::Int(_)) {
                self.error(
                    span,
                    format!(
                        "`char` casts only to an integer type, not `{}`",
                        self.ty_display(target_head)
                    ),
                );
                return self.poison(span);
            }
            return self.mk_expr(ExprKind::Cast(Box::new(e), target), target, span);
        }
        self.register_bound(self.lang.num, target, span);
        self.register_bound(self.lang.num, src_ty, span);

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
        // Expand the callee path through the file's `import` bindings
        // *here*, before the textual intrinsic check below: `arr::get(…)` with
        // `import core::intrinsic::array as arr;` must dispatch as the
        // intrinsic. Locals were checked first — a binding never shadows one.
        let expanded_fc;
        let fc = match self.expand_path(&fc.name) {
            Some(name) => {
                expanded_fc = surface::FuncCall {
                    name,
                    ty_args: fc.ty_args.clone(),
                    args: fc.args.clone(),
                };
                &expanded_fc
            }
            None => fc,
        };
        // The built-in `core` package: `core::intrinsic::<family>::<fn>` calls
        // are intrinsics, and nothing else lives under `core` (the package name
        // is reserved, so this can never shadow user code).
        if let Some(done) = self.infer_intrinsic(fc, span) {
            return done;
        }
        let Some(def) = self.resolve_function_ref(&fc.name) else {
            // A hit on a *private* item of a loaded extern package reports
            // access, not absence.
            if let Some(msg) = self.extern_private_msg(&fc.name, DefKind::Function) {
                self.error(span, msg);
                return self.poison(span);
            }
            // `Trait::method(recv, …)` — the path prefix names a trait.
            if !fc.name.segments.is_empty()
                && let Some(done) = self.try_trait_path_call(fc, span)
            {
                return done;
            }
            // `Type::f(…)` / `G::f(…)` — the prefix names a type whose
            // traits (impls or bounds) declare `f`.
            if !fc.name.segments.is_empty()
                && let Some(done) = self.try_type_assoc_call(fc, span)
            {
                return done;
            }
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

        // Partial application: fewer arguments than parameters lifts the
        // function to a closure and applies the supplied arguments, yielding a
        // residual closure over the rest (`add(5)` : `i32 -> i32`).
        if fc.args.len() < proto.params.len() {
            if self.reject_intrinsic_value(def, span) {
                return self.poison(span);
            }
            let lifted = self.lift_function(def, &proto, &inst, span);
            return self.closure_apply(lifted, &fc.args, span);
        }
        if fc.args.len() > proto.params.len() {
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

    /// (POSTFIX-APP) `Γ ⊢ base.acc₁.….accₙ(args)`: the chain prefix projects
    /// as in (PROJ); the last access resolves **method-first**, Rust-style —
    /// a dot-callable method (a function under the record's qualified path
    /// whose first parameter is named `self`) wins whenever one exists.
    /// Dispatch is decided by existence and shape only, never by visibility,
    /// so a private field cannot obstruct a `pub` method. Without a method
    /// the call applies a closure-valued field, and a parenthesized callee —
    /// `(e.f)(x)` — always forces the field, since it routes through
    /// (PROJ)+(APP) instead of this rule. Method-first order is
    /// deterministic, so no speculative inference is needed.
    fn infer_postfix_call(
        &mut self,
        base: &surface::Expr,
        accs: &[surface::Access],
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let base = self.infer_expr(base);
        let mut raw = self.infer.shallow_resolve(base.ty);
        let mut indices = Vec::new();
        let (last, prefix_accs) = accs.split_last().expect("a postfix call has an access");
        for acc in prefix_accs {
            let Some((idx, ty)) = self.project_step(raw, acc, span) else {
                return self.poison(span);
            };
            indices.push(idx);
            raw = ty;
        }

        let arced = matches!(raw.kind(), TyKind::Arc(_));
        let cur = self.peel_arc(raw);
        // A bare-generic receiver has no fields; its dot calls dispatch
        // through the traits its bounds provide.
        if let TyKind::Generic(g) = *cur.kind()
            && let surface::Access::Named(name) = last
        {
            let receiver = if indices.is_empty() {
                base
            } else {
                self.mk_expr(ExprKind::Proj(Box::new(base), indices), raw, span)
            };
            return self.dispatch_generic_dot(g, cur, receiver, *name, args, span);
        }
        // A scalar (or `str`) receiver has no fields either; its dot calls
        // dispatch through the traits the builtin impls provide (`a.eq(b)`,
        // `a.cmp(b)` — lowered intrinsically by the selected impl, or a
        // method-carrying builtin impl like `Hash for str`).
        if (cur.is_scalar() || matches!(cur.kind(), TyKind::Str))
            && let surface::Access::Named(name) = last
        {
            let receiver = if indices.is_empty() {
                base
            } else {
                self.mk_expr(ExprKind::Proj(Box::new(base), indices), raw, span)
            };
            return self.dispatch_scalar_dot(cur, receiver, *name, args, span);
        }
        let TyKind::Record {
            def,
            args: rargs,
            flex,
        } = cur.kind()
        else {
            let shown = self.infer.resolve(cur);
            self.error(
                span,
                format!("cannot access a field of `{}`", self.ty_display(shown)),
            );
            return self.poison(span);
        };
        let (def, flex) = (*def, *flex);
        if let surface::Access::Named(name) = last {
            if let Some(method) = self.dot_method_candidate(def, *name) {
                let receiver = if indices.is_empty() {
                    base
                } else {
                    self.mk_expr(ExprKind::Proj(Box::new(base), indices), raw, span)
                };
                return self.infer_method_call(receiver, Some(def), method, *name, args, span);
            }
            // No inherent method: trait methods are next in line, still
            // ahead of fields. An unresolved receiver cannot probe impls —
            // ask for annotations when a matching trait method could exist.
            let rcur = self.infer.resolve(cur);
            if ty_has_hole(rcur) {
                if !self.traits_declaring(*name).is_empty() {
                    self.error(
                        span,
                        format!(
                            "type annotations needed: cannot resolve `{}` on `{}`",
                            self.sym(*name),
                            self.ty_display(rcur)
                        ),
                    );
                    return self.poison(span);
                }
            } else {
                let cands = self.ground_trait_candidates(rcur, *name);
                match cands[..] {
                    [] => {}
                    [(tid, midx)] => {
                        let receiver = if indices.is_empty() {
                            base
                        } else {
                            self.mk_expr(ExprKind::Proj(Box::new(base), indices), raw, span)
                        };
                        return self.dispatch_trait_method(tid, midx, receiver, args, span);
                    }
                    [(a, _), (b, _), ..] => {
                        self.ambiguous_method_error(*name, rcur, a, b, span);
                        return self.poison(span);
                    }
                }
            }
        }
        match self.resolve_field(def, rargs, flex, last) {
            Ok((idx, field_ty, _)) => {
                // One flat projection over the whole chain, then closure
                // application — the same shape a parenthesized callee takes.
                indices.push(idx);
                let field_ty = self.infer.shallow_resolve(field_ty);
                let field_ty = if arced {
                    let group = self.arc_recursive_group(def);
                    self.arc_promote_member_ty(&group, field_ty)
                } else {
                    field_ty
                };
                let callee = self.mk_expr(ExprKind::Proj(Box::new(base), indices), field_ty, span);
                self.closure_apply(callee, args, span)
            }
            Err(FieldErr::Private { field, record }) => {
                self.error(
                    span,
                    format!("field `{field}` of record `{record}` is private"),
                );
                self.poison(span)
            }
            Err(FieldErr::Missing) => {
                let surface::Access::Named(name) = last else {
                    self.error(span, "no such field");
                    return self.poison(span);
                };
                // A receiver-less associated function is not a method
                // candidate; when one exists, point at the path spelling
                // instead of reporting absence.
                let type_display = self.defs.path(def).display(self.resolver);
                let mut mpath = self.defs.path(def).0.clone();
                mpath.push(*name);
                if self.defs.lookup_function(&mpath).is_some() {
                    self.error(
                        span,
                        format!(
                            "`{type_display}::{0}` takes no receiver and is not callable \
                             with `.`; call it as a path",
                            self.sym(*name)
                        ),
                    );
                } else {
                    // A trait declares it but the receiver holds no impl:
                    // steer toward the missing impl instead of a bare miss.
                    let hint = self
                        .traits_declaring(*name)
                        .first()
                        .map(|&(tid, _)| {
                            format!(
                                "; trait `{}` declares it, but `{type_display}` does not \
                                 implement `{0}`",
                                self.trait_display(tid)
                            )
                        })
                        .unwrap_or_default();
                    self.error(
                        span,
                        format!(
                            "no field or method `{}` on `{type_display}`{hint}",
                            self.sym(*name)
                        ),
                    );
                }
                self.poison(span)
            }
        }
    }

    /// A dot-callable method on `record_def` named `name`: a function
    /// declared under the record's qualified path whose first parameter is
    /// named `self`. Shape only — visibility gates *after* dispatch, in
    /// [`Self::infer_method_call`], so privacy can never redirect a call.
    fn dot_method_candidate(&self, record_def: DefId, name: TokenKey) -> Option<DefId> {
        let mut mpath = self.defs.path(record_def).0.clone();
        mpath.push(name);
        let def = self.defs.lookup_function(&mpath)?;
        let takes_self = self
            .functions
            .get(&def)?
            .params
            .first()
            .is_some_and(|(n, _)| self.sym(*n) == "self");
        takes_self.then_some(def)
    }

    /// (METHOD) `Γ ⊢ recv.m(args)`: type the candidate resolved by
    /// [`Self::dot_method_candidate`] as an ordinary `FuncCall` with the
    /// receiver prepended. Type arguments are always inferred (dot calls
    /// carry no turbofish); the receiver unifies against the instantiated
    /// `self` parameter, which yields the arc-reconciliation and flexivity
    /// diagnostics for free.
    fn infer_method_call(
        &mut self,
        receiver: Expr<'tcx>,
        record_def: Option<DefId>,
        def: DefId,
        name: TokenKey,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        // A builtin receiver has no def; its type spelling names it.
        let type_display = match record_def {
            Some(record_def) => self.defs.path(record_def).display(self.resolver),
            None => {
                let recv = self.infer.resolve(receiver.ty);
                self.ty_display(recv)
            }
        };
        // The absolute candidate lookup bypassed `resolve_extern`'s pub
        // gate, so an extern-owned method must re-check visibility here —
        // with the same is-private (not not-found) wording as path
        // resolution. After dispatch, deliberately: privacy reports, it
        // never redirects the call to a field.
        if let Some(&head) = self.extern_defs.get(&def)
            && self.functions[&def].visibility != surface::Visibility::Public
        {
            self.error(
                span,
                format!(
                    "function `{}` in package `{}` is private",
                    self.sym(name),
                    self.sym(head)
                ),
            );
            return self.poison(span);
        }
        let proto = self.functions[&def].clone();
        self.record_use(name);
        if proto.is_regional && !self.inside_region {
            self.error(span, "cannot call a regional function outside of a region");
        }
        let inst = self.instantiate(&proto.generics, &[], span);
        let expected_recv = self.infer.instantiate_ty(proto.params[0].1, &inst);
        self.expect(receiver.ty, expected_recv, span);
        let rest = &proto.params[1..];
        if args.len() < rest.len() {
            self.error(
                span,
                format!(
                    "partial application of a method call is not supported; \
                     call `{type_display}::{}` instead",
                    self.sym(name)
                ),
            );
        } else if args.len() > rest.len() {
            self.error(
                span,
                format!(
                    "method `{}` expects {} argument(s) besides the receiver, got {}",
                    self.sym(name),
                    rest.len(),
                    args.len()
                ),
            );
        }
        let mut out = vec![receiver];
        for (arg, (_, pty)) in args.iter().zip(rest) {
            let expected = self.infer.instantiate_ty(*pty, &inst);
            out.push(self.check_expr(arg, expected));
        }
        let ty_args = self.inst_args(&proto.generics, &inst);
        let result = self.infer.instantiate_ty(proto.return_ty, &inst);
        self.mk_expr(
            ExprKind::FuncCall {
                target: def,
                ty_args,
                args: out,
                regional: proto.is_regional,
            },
            result,
            span,
        )
    }

    // ----- trait-method dispatch -----

    /// Every trait whose method table declares `name`, with the method's
    /// index. Deterministic: ascending trait id. Multi-parameter traits
    /// never qualify — a dot call (or `Trait::m` path call) supplies only
    /// `Self`, so their non-`Self` arguments have no instantiation source
    /// until turbofish lands on trait-method calls.
    fn traits_declaring(&self, name: TokenKey) -> Vec<(TraitId, usize)> {
        (0..self.traits.traits_len() as u32)
            .map(TraitId)
            .filter_map(|tid| {
                let def = self.traits.trait_def(tid);
                if !def.params.is_empty() {
                    return None;
                }
                def.methods
                    .iter()
                    .position(|m| m.name == name)
                    .map(|idx| (tid, idx))
            })
            .collect()
    }

    /// Dot candidates for a *ground-headed* receiver: the declaring traits
    /// whose goal `recv : Trait` the solver proves. Probing runs real
    /// selection, so a candidate impl's where-clauses count — `Box<bool>`
    /// is not a `Show` candidate under `impl<T: Num> Show for Box<T>`.
    fn ground_trait_candidates(&self, recv: Ty<'tcx>, name: TokenKey) -> Vec<(TraitId, usize)> {
        self.traits_declaring(name)
            .into_iter()
            .filter(|&(tid, _)| {
                let goal = Obligation::Trait(TraitRef {
                    trait_id: tid,
                    args: vec![recv],
                });
                SelectCtxt::new(&self.traits, self.tcx, &self.generics)
                    .select(&goal)
                    .is_ok()
            })
            .collect()
    }

    /// Dot candidates for a bare-generic receiver: declaring traits
    /// reachable from the generic's bounds — a bound's super-traits
    /// dispatch too (`T: Integral` reaches a method `Num` declares).
    fn generic_trait_candidates(&self, g: GenericId, name: TokenKey) -> Vec<(TraitId, usize)> {
        let mut reach: Vec<TraitId> = Vec::new();
        let mut work: Vec<TraitId> = self.generic_bounds(g).to_vec();
        while let Some(t) = work.pop() {
            if reach.contains(&t) {
                continue;
            }
            reach.push(t);
            work.extend(
                self.traits
                    .trait_def(t)
                    .supertraits
                    .iter()
                    .map(|s| s.trait_id),
            );
        }
        reach.sort_by_key(|t| t.0);
        reach
            .into_iter()
            .filter_map(|tid| {
                let def = self.traits.trait_def(tid);
                if !def.params.is_empty() {
                    return None;
                }
                def.methods
                    .iter()
                    .position(|m| m.name == name)
                    .map(|idx| (tid, idx))
            })
            .collect()
    }

    /// (T-METHOD) Dispatch a resolved trait method — `recv.m(args)` or the
    /// `Trait::m(recv, args…)` spelling — per the receiver's shape:
    ///
    /// * a **ground-headed** receiver commits at check time: coherence makes
    ///   the selected impl unique, and the call becomes an ordinary
    ///   [`ExprKind::FuncCall`] to that impl's method;
    /// * a **bare-generic** receiver has no impl until monomorphization
    ///   grounds `Self`, so the call is a serializable
    ///   [`ExprKind::TraitCall`] resolved per instance.
    fn dispatch_trait_method(
        &mut self,
        tid: TraitId,
        midx: usize,
        receiver: Expr<'tcx>,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        // An associated function has no receiver slot: dot dispatch (and the
        // UFCS spelling that feeds a receiver here) cannot reach it.
        if self.traits.trait_def(tid).methods[midx].receiver.is_none() {
            let mname = self.traits.trait_def(tid).methods[midx].name;
            let shown = self.trait_display(tid);
            let m = self.sym(mname);
            self.error(
                span,
                format!(
                    "`{m}` is an associated function of `{shown}` (no receiver); \
                     call it as `{shown}::{m}(…)` or `Type::{m}(…)`"
                ),
            );
            return self.poison(span);
        }
        let raw = self.infer.resolve(receiver.ty);
        let peeled = self.peel_arc(raw);
        if let TyKind::Generic(_) = peeled.kind() {
            return self.trait_call_generic(tid, midx, receiver, peeled, args, span);
        }
        let mname = self.traits.trait_def(tid).methods[midx].name;
        if ty_has_hole(peeled) {
            self.error(
                span,
                format!(
                    "type annotations needed: cannot resolve `{}` on `{}`",
                    self.sym(mname),
                    self.ty_display(peeled)
                ),
            );
            return self.poison(span);
        }
        let goal = Obligation::Trait(TraitRef {
            trait_id: tid,
            args: vec![peeled],
        });
        let outcome = SelectCtxt::new(&self.traits, self.tcx, &self.generics).select(&goal);
        match outcome {
            Ok(Evidence::Impl { impl_id, .. }) => {
                let Some(&def) = self.traits.impl_def(impl_id).methods.get(midx) else {
                    // A method-less impl: the builtin special form (or its
                    // no-`core` fallback twin). For the comparison tower
                    // the call *is* the intrinsic — this is also what makes
                    // method syntax on scalars work, `(5).eq(3)`. Any other
                    // method-less selection means the impl elaborated with
                    // errors, reported at its site.
                    return self.builtin_cmp_call(tid, mname, receiver, args, span);
                };
                // A method-carrying impl's self type is a record or a
                // builtin scalar; only a record has a def to display.
                let head = match *peeled.kind() {
                    TyKind::Record { def: head, .. } => Some(head),
                    _ => None,
                };
                self.infer_method_call(receiver, head, def, mname, args, span)
            }
            Ok(_) => {
                // Super-projected evidence proves the bound, but only a
                // direct impl carries the method body.
                let shown = self.trait_display(tid);
                self.error(
                    span,
                    format!(
                        "`{ty}` reaches `{shown}` only through a sub-trait; calling `{m}` \
                         needs a direct `impl {shown} for {ty}`",
                        ty = self.ty_display(peeled),
                        m = self.sym(mname),
                    ),
                );
                self.poison(span)
            }
            Err(SelectError::NoImpl(inner)) => {
                let msg = self.no_impl_message(peeled, tid, &inner);
                self.error(span, msg);
                self.poison(span)
            }
            Err(SelectError::DepthLimit(g)) => {
                let msg = self.depth_limit_message(&g);
                self.error(span, msg);
                self.poison(span)
            }
        }
    }

    /// Dot dispatch on a scalar receiver: the traits its impls provide
    /// are the only method sources (there are no fields and no inherent
    /// methods). The unique named candidate wins.
    fn dispatch_scalar_dot(
        &mut self,
        recv_ty: Ty<'tcx>,
        receiver: Expr<'tcx>,
        name: TokenKey,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let mut candidates: Vec<(TraitId, usize)> = self
            .traits
            .impls()
            .filter(|imp| imp.self_ty == recv_ty)
            .filter_map(|imp| {
                let tid = imp.trait_ref.trait_id;
                self.traits
                    .trait_def(tid)
                    .methods
                    .iter()
                    .position(|m| m.name == name)
                    .map(|midx| (tid, midx))
            })
            .collect();
        candidates.sort_unstable_by_key(|(tid, _)| tid.0);
        candidates.dedup();
        match candidates.as_slice() {
            [] => {
                self.error(
                    span,
                    format!(
                        "no method `{}` on `{}`",
                        self.sym(name),
                        self.ty_display(recv_ty)
                    ),
                );
                self.poison(span)
            }
            [(tid, midx)] => self.dispatch_trait_method(*tid, *midx, receiver, args, span),
            more => {
                let shown: Vec<String> = more
                    .iter()
                    .map(|(tid, _)| format!("`{}`", self.trait_display(*tid)))
                    .collect();
                self.error(
                    span,
                    format!(
                        "ambiguous method `{}` on `{}`: provided by {}",
                        self.sym(name),
                        self.ty_display(recv_ty),
                        shown.join(", ")
                    ),
                );
                self.poison(span)
            }
        }
    }

    /// A selected method-less impl of a comparison-tower trait: the call
    /// lowers intrinsically at check time. `eq`/`lt`/`le`/`gt`/`ge` become
    /// the `Cmp` node; `cmp` becomes a let-bound three-way comparison
    /// constructing `Ordering`. Selection already proved the impl, so no
    /// obligation is registered.
    fn builtin_cmp_call(
        &mut self,
        tid: TraitId,
        mname: TokenKey,
        receiver: Expr<'tcx>,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let lang_cmp = self
            .lang
            .item_of(self.traits.trait_def(tid).def)
            .is_some_and(|item| item.is_cmp_trait());
        let name_idx = self.cmp_method_names.iter().position(|&k| k == mname);
        let (Some(idx), true, [other]) = (name_idx, lang_cmp, args) else {
            // Not the builtin form: the impl elaborated with errors,
            // already reported at its declaration site.
            return self.poison(span);
        };
        let recv_ty = self.infer.resolve(receiver.ty);
        let other = self.check_expr(other, recv_ty);
        let bool_ty = self.tcx.mk_bool();
        let op = match idx {
            0 => CmpOp::Eq,
            1 => CmpOp::Lt,
            2 => CmpOp::Le,
            3 => CmpOp::Gt,
            4 => CmpOp::Ge,
            _ => {
                // `cmp`: bind both operands once, then the three-way
                // comparison over them.
                return self.three_way_cmp(tid, mname, receiver, other, recv_ty, span);
            }
        };
        self.mk_expr(
            ExprKind::Cmp(Box::new(receiver), op, Box::new(other)),
            bool_ty,
            span,
        )
    }

    /// `Ord::cmp` on a builtin impl: `{ let a = recv; let b = other;
    /// if a < b { Less } else { if a == b { Equal } else { Greater } } }`.
    fn three_way_cmp(
        &mut self,
        tid: TraitId,
        mname: TokenKey,
        receiver: Expr<'tcx>,
        other: Expr<'tcx>,
        recv_ty: Ty<'tcx>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        // Ground the declared return type — `Ordering` — for this receiver.
        let tdef = self.traits.trait_def(tid);
        let sig = tdef
            .methods
            .iter()
            .find(|m| m.name == mname)
            .expect("dispatched by name");
        let mut subst = crate::full::subst::Subst::default();
        subst.insert(tdef.self_param, recv_ty);
        let ret = crate::full::subst::subst_ty(self.tcx, sig.ret, &subst);
        let TyKind::Record { def, .. } = *ret.kind() else {
            self.error(span, "`cmp` must return `core`'s `Ordering`");
            return self.poison(span);
        };
        let variant_of = |el: &Self, name: &str| -> Option<usize> {
            let record = el.records.get(&def)?;
            let Some(RecordFields::Variants(variants)) = &record.fields else {
                return None;
            };
            variants.iter().position(|v| el.sym(v.name) == name)
        };
        let (Some(less), Some(equal), Some(greater)) = (
            variant_of(self, "Less"),
            variant_of(self, "Equal"),
            variant_of(self, "Greater"),
        ) else {
            self.error(
                span,
                "`cmp` must return `core`'s `Ordering` (with variants `Less`, `Equal`, `Greater`)",
            );
            return self.poison(span);
        };
        let bool_ty = self.tcx.mk_bool();
        let unit = self.tcx.mk_unit();
        let a = self.vars.fresh(mname, recv_ty, span);
        let b = self.vars.fresh(mname, recv_ty, span);
        let bind = |el: &mut Self, var, value: Expr<'tcx>| {
            el.mk_expr(
                ExprKind::Let {
                    var,
                    name: mname,
                    span,
                    value: Box::new(value),
                },
                unit,
                span,
            )
        };
        let var = |el: &mut Self, v| el.mk_expr(ExprKind::Var(v), recv_ty, span);
        let cmp = |el: &mut Self, op| {
            let l = var(el, a);
            let r = var(el, b);
            el.mk_expr(ExprKind::Cmp(Box::new(l), op, Box::new(r)), bool_ty, span)
        };
        let ctor = |el: &mut Self, variant| {
            el.mk_expr(
                ExprKind::VariantCall {
                    target: def,
                    ty_args: Vec::new(),
                    variant,
                    args: Vec::new(),
                },
                ret,
                span,
            )
        };
        let bind_a = bind(self, a, receiver);
        let bind_b = bind(self, b, other);
        let lt = cmp(self, CmpOp::Lt);
        let eq = cmp(self, CmpOp::Eq);
        let less = ctor(self, less);
        let equal = ctor(self, equal);
        let greater = ctor(self, greater);
        let inner = self.mk_expr(
            ExprKind::If(Box::new(eq), Box::new(equal), Box::new(greater)),
            ret,
            span,
        );
        let outer = self.mk_expr(
            ExprKind::If(Box::new(lt), Box::new(less), Box::new(inner)),
            ret,
            span,
        );
        self.mk_expr(ExprKind::Seq(vec![bind_a, bind_b, outer]), ret, span)
    }

    /// The generic-receiver half of (T-METHOD): type the call against the
    /// trait's [`MethodSig`] with `Self ↦` the receiver and the method's own
    /// generics fresh, and emit the [`ExprKind::TraitCall`].
    ///
    /// [`MethodSig`]: crate::semi::traits::MethodSig
    fn trait_call_generic(
        &mut self,
        tid: TraitId,
        midx: usize,
        receiver: Expr<'tcx>,
        self_ty: Ty<'tcx>,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let tdef = self.traits.trait_def(tid);
        if !tdef.params.is_empty() {
            let shown = self.trait_display(tid);
            self.error(
                span,
                format!(
                    "cannot dispatch a method of multi-parameter trait `{shown}` \
                     on a generic receiver"
                ),
            );
            return self.poison(span);
        }
        let trait_def_id = tdef.def;
        let self_param = tdef.self_param;
        let sig = tdef.methods[midx].clone();
        self.record_use(sig.name);
        if sig.is_regional && !self.inside_region {
            self.error(span, "cannot call a regional function outside of a region");
        }
        let mut inst = self.instantiate(&sig.generics, &[], span);
        inst.insert(self_param, self_ty);
        let expected_recv = self.infer.instantiate_ty(sig.params[0], &inst);
        self.expect(receiver.ty, expected_recv, span);
        let rest = &sig.params[1..];
        if args.len() != rest.len() {
            self.error(
                span,
                format!(
                    "method `{}` expects {} argument(s) besides the receiver, got {}",
                    self.sym(sig.name),
                    rest.len(),
                    args.len()
                ),
            );
        }
        let out: Vec<Expr<'tcx>> = std::iter::once(receiver)
            .chain(args.iter().zip(rest).map(|(arg, &pty)| {
                let expected = self.infer.instantiate_ty(pty, &inst);
                self.check_expr(arg, expected)
            }))
            .collect();
        let ty_args: Vec<Ty<'tcx>> = std::iter::once(self_ty)
            .chain(self.inst_args(&sig.generics, &inst))
            .collect();
        let result = self.infer.instantiate_ty(sig.ret, &inst);
        self.mk_expr(
            ExprKind::TraitCall {
                trait_def: trait_def_id,
                method: midx as u32,
                ty_args,
                args: out,
            },
            result,
            span,
        )
    }

    /// The dot form on a bare-generic receiver: search the generic's bounds
    /// for the (unique) declaring trait.
    fn dispatch_generic_dot(
        &mut self,
        g: GenericId,
        recv_ty: Ty<'tcx>,
        receiver: Expr<'tcx>,
        name: TokenKey,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let cands = self.generic_trait_candidates(g, name);
        match cands[..] {
            [] => {
                self.error(
                    span,
                    format!(
                        "no method `{}` on `{}`; its bounds do not declare one",
                        self.sym(name),
                        self.ty_display(recv_ty)
                    ),
                );
                self.poison(span)
            }
            [(tid, midx)] => self.dispatch_trait_method(tid, midx, receiver, args, span),
            [(a, _), (b, _), ..] => {
                self.ambiguous_method_error(name, recv_ty, a, b, span);
                self.poison(span)
            }
        }
    }

    fn ambiguous_method_error(
        &mut self,
        name: TokenKey,
        recv_ty: Ty<'tcx>,
        a: TraitId,
        b: TraitId,
        span: Option<Span>,
    ) {
        let (da, db) = (self.trait_display(a), self.trait_display(b));
        self.error(
            span,
            format!(
                "multiple traits provide method `{m}` for `{ty}`: `{da}` and `{db}`; \
                 disambiguate with a trait path call like `{da}::{m}(…)`",
                m = self.sym(name),
                ty = self.ty_display(recv_ty),
            ),
        );
    }

    /// The `Trait::method(recv, args…)` path spelling: the path prefix names
    /// the trait, the basename its method, and the first argument is the
    /// receiver. `None` when the prefix is not a trait — the caller falls
    /// back to its unknown-function diagnostic.
    fn try_trait_path_call(
        &mut self,
        fc: &surface::FuncCall,
        span: Option<Span>,
    ) -> Option<Expr<'tcx>> {
        let (tname, prefix) = fc.name.segments.split_last()?;
        let tpath = surface::Path {
            basename: *tname,
            segments: prefix.iter().copied().collect(),
        };
        let def = self.resolve_trait_ref_quiet(&tpath)?;
        let tid = self
            .traits
            .trait_by_def(def)
            .expect("every trait-namespace def has a TraitDb entry");
        let mname = self.sym(fc.name.basename).to_string();
        let Some(midx) = self
            .traits
            .trait_def(tid)
            .methods
            .iter()
            .position(|m| m.name == fc.name.basename)
        else {
            let shown = self.trait_display(tid);
            self.error(span, format!("trait `{shown}` has no method `{mname}`"));
            return Some(self.poison(span));
        };
        if !fc.ty_args.is_empty() {
            self.error(
                span,
                "type arguments on a trait-method call are not supported yet",
            );
            return Some(self.poison(span));
        }
        // An associated function: no receiver argument, and `Self` comes
        // from context — a fresh hole that unification solves (the result
        // type mentions `Self` in every interesting case, e.g. `default()
        // -> Self`), left generic in a bounded function, ground otherwise.
        if self.traits.trait_def(tid).methods[midx].receiver.is_none() {
            self.record_use(fc.name.basename);
            let self_ty = self.infer.new_hole_ty();
            return Some(self.assoc_trait_call(tid, midx, self_ty, &fc.args, span));
        }
        let Some(recv_src) = fc.args.first() else {
            let shown = self.trait_display(tid);
            self.error(
                span,
                format!("`{shown}::{mname}` takes its receiver as the first argument"),
            );
            return Some(self.poison(span));
        };
        self.record_use(fc.name.basename);
        let receiver = self.infer_expr(recv_src);
        Some(self.dispatch_trait_method(tid, midx, receiver, &fc.args[1..], span))
    }

    /// An associated-function call with `Self = self_ty`: every argument is
    /// positional, and the call is always the serializable deferred
    /// [`ExprKind::TraitCall`] — a generic `Self` resolves per instance at
    /// monomorphization, and a ground (or ground-by-unification) `Self`
    /// resolves there identically, so one form covers both.
    fn assoc_trait_call(
        &mut self,
        tid: TraitId,
        midx: usize,
        self_ty: Ty<'tcx>,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let tdef = self.traits.trait_def(tid);
        if !tdef.params.is_empty() {
            let shown = self.trait_display(tid);
            self.error(
                span,
                format!(
                    "cannot call an associated function of multi-parameter \
                     trait `{shown}` without a concrete trait reference"
                ),
            );
            return self.poison(span);
        }
        let trait_def_id = tdef.def;
        let self_param = tdef.self_param;
        let sig = tdef.methods[midx].clone();
        debug_assert!(sig.receiver.is_none(), "assoc_trait_call needs no receiver");
        if sig.is_regional && !self.inside_region {
            self.error(span, "cannot call a regional function outside of a region");
        }
        let mut inst = self.instantiate(&sig.generics, &[], span);
        inst.insert(self_param, self_ty);
        if args.len() != sig.params.len() {
            self.error(
                span,
                format!(
                    "associated function `{}` expects {} argument(s), got {}",
                    self.sym(sig.name),
                    sig.params.len(),
                    args.len()
                ),
            );
        }
        let out: Vec<Expr<'tcx>> = args
            .iter()
            .zip(&sig.params)
            .map(|(arg, &pty)| {
                let expected = self.infer.instantiate_ty(pty, &inst);
                self.check_expr(arg, expected)
            })
            .collect();
        let ty_args: Vec<Ty<'tcx>> = std::iter::once(self_ty)
            .chain(self.inst_args(&sig.generics, &inst))
            .collect();
        let result = self.infer.instantiate_ty(sig.ret, &inst);
        // The bound obligation is what turns an unimplemented `Self` into a
        // real diagnostic (and constrains a hole `Self` like any bound).
        self.register_bound(tid, self_ty, span);
        self.mk_expr(
            ExprKind::TraitCall {
                trait_def: trait_def_id,
                method: midx as u32,
                ty_args,
                args: out,
            },
            result,
            span,
        )
    }

    /// The `Type::f(…)` / `G::f(…)` spellings: the path prefix names a type
    /// — a generic parameter in scope or a record — and `f` resolves among
    /// the traits that type implements (or that bound it). An associated
    /// function calls with `Self` fixed to the named type; a method with a
    /// receiver dispatches UFCS style, its first argument the receiver.
    /// `None` when the prefix names no type — the caller keeps its
    /// unknown-function diagnostic.
    fn try_type_assoc_call(
        &mut self,
        fc: &surface::FuncCall,
        span: Option<Span>,
    ) -> Option<Expr<'tcx>> {
        let (tname, prefix) = fc.name.segments.split_last()?;
        let self_ty = if prefix.is_empty()
            && let Some(&g) = self.generic_names.get(tname)
        {
            self.tcx.mk_generic(g)
        } else {
            let tpath = surface::Path {
                basename: *tname,
                segments: prefix.iter().copied().collect(),
            };
            let def = self.resolve_record_ref(&tpath)?;
            // A generic record's arguments are holes for unification to
            // solve; the common associated-function shapes mention them in
            // the parameters or the result.
            let rec = &self.records[&def];
            let default_cap = rec.default_cap;
            let n = rec.ty_params.len();
            let args: Vec<Ty<'tcx>> = (0..n).map(|_| self.infer.new_hole_ty()).collect();
            let flex = match default_cap {
                crate::semi::ctxt::DefaultCap::Value | crate::semi::ctxt::DefaultCap::Shared => {
                    Flexivity::Irrelevant
                }
                crate::semi::ctxt::DefaultCap::Regional => Flexivity::Rigid,
            };
            self.tcx.mk_record(def, &args, flex)
        };
        let name = fc.name.basename;
        let cands = match self_ty.kind() {
            TyKind::Generic(g) => self.generic_trait_candidates(*g, name),
            _ => self.ground_trait_candidates(self_ty, name),
        };
        let (tid, midx) = match cands[..] {
            [] => return None,
            [one] => one,
            [(a, _), (b, _), ..] => {
                self.ambiguous_method_error(name, self_ty, a, b, span);
                return Some(self.poison(span));
            }
        };
        if !fc.ty_args.is_empty() {
            self.error(
                span,
                "type arguments on a trait-method call are not supported yet",
            );
            return Some(self.poison(span));
        }
        self.record_use(name);
        if self.traits.trait_def(tid).methods[midx].receiver.is_some() {
            // UFCS: `Type::method(recv, args…)`.
            let Some(recv_src) = fc.args.first() else {
                let m = self.sym(name);
                self.error(
                    span,
                    format!("`{m}` takes its receiver as the first argument"),
                );
                return Some(self.poison(span));
            };
            // The prefix constrains the receiver: unify them — up to the arc
            // peel, and carrying the receiver's own flexivity refinement —
            // so `Box::get(p)` on a `Pair` errors rather than silently
            // dispatching on `Pair`'s impl after selecting by `Box`.
            let receiver = self.infer_expr(recv_src);
            let resolved = self.infer.shallow_resolve(receiver.ty);
            let peeled = self.peel_arc(resolved);
            let expected = match (self_ty.kind(), peeled.kind()) {
                (TyKind::Record { def, args, .. }, TyKind::Record { flex, .. }) => {
                    self.tcx.mk_record(*def, args, *flex)
                }
                _ => self_ty,
            };
            self.expect(peeled, expected, span);
            return Some(self.dispatch_trait_method(tid, midx, receiver, &fc.args[1..], span));
        }
        Some(self.assoc_trait_call(tid, midx, self_ty, &fc.args, span))
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

    /// (LIFT) Reify a named function `def` as a first-class closure value —
    /// `|p₀,…,pₙ| def(p₀,…,pₙ)` — so a bare `fn` name can appear in value
    /// position and a `fn` can be partially applied. A top-level function
    /// closes over nothing, so the closure has no captures; its parameter and
    /// return types come from the prototype instantiated at `inst` (holes for
    /// an unapplied generic, which the surrounding context then solves). This
    /// reuses the ordinary closure path, so ownership and codegen need no
    /// special case.
    /// Reject lifting a bound intrinsic prototype: the declaration is a
    /// location anchor; the operation itself has no function value (calls
    /// through its path are intercepted as the intrinsic).
    fn reject_intrinsic_value(&mut self, def: DefId, span: Option<Span>) -> bool {
        if self.lang.is_intrinsic_def(def) {
            let shown = self.defs.path(def).display(self.resolver);
            self.error(
                span,
                format!("`{shown}` is an intrinsic and cannot be used as a value"),
            );
            return true;
        }
        false
    }

    fn lift_function(
        &mut self,
        def: DefId,
        proto: &FuncProto<'tcx>,
        inst: &Instantiation<'tcx>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        // A regional function takes an implicit region handle, so it has no
        // plain closure type to lift to.
        if proto.is_regional {
            self.error(span, "cannot use a regional function as a closure value");
            return self.poison(span);
        }
        // One fresh parameter per function parameter, at the instantiated type;
        // the body forwards them to a direct call. The params are scoped to the
        // synthesized body alone (mark/restore), never the enclosing block.
        let mark = self.vars.mark();
        let mut params = Vec::with_capacity(proto.params.len());
        let mut args = Vec::with_capacity(proto.params.len());
        for (name, pty) in &proto.params {
            let ty = self.infer.instantiate_ty(*pty, inst);
            let var = self.vars.fresh(*name, ty, span);
            params.push((var, ty));
            args.push(self.mk_expr(ExprKind::Var(var), ty, span));
        }
        self.vars.restore(mark);
        let ty_args = self.inst_args(&proto.generics, inst);
        let ret = self.infer.instantiate_ty(proto.return_ty, inst);
        let body = self.mk_expr(
            ExprKind::FuncCall {
                target: def,
                ty_args,
                args,
                regional: false,
            },
            ret,
            span,
        );
        let param_tys: Vec<Ty<'tcx>> = params.iter().map(|(_, t)| *t).collect();
        let ty = self.tcx.mk_closure(&param_tys, ret);
        self.mk_expr(
            ExprKind::Closure(ClosureExpr {
                captures: Vec::new(),
                params,
                body: Box::new(body),
            }),
            ty,
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
        self.lambda(lam, None, span)
    }

    /// The shared lambda rule: synthesis when `expected` is `None`, checking
    /// against an expected `(params) -> ret` otherwise (unannotated
    /// parameters take the expected types; annotations must agree with them).
    fn lambda(
        &mut self,
        lam: &surface::Lambda,
        expected: Option<(&'tcx [Ty<'tcx>], Ty<'tcx>)>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let mark = self.vars.mark();
        let mut params = Vec::new();
        for (i, (name, ty)) in lam.args.iter().enumerate() {
            let exp_p = expected.map(|(ps, _)| ps[i]);
            let pty = match ty {
                Some(t) => {
                    let t = self.eval_type(t);
                    if let Some(exp_p) = exp_p {
                        self.expect(t, exp_p, span);
                    }
                    t
                }
                None => exp_p.unwrap_or_else(|| self.infer.new_hole_ty()),
            };
            let var = self.vars.fresh(*name, pty, None);
            params.push((var, pty));
        }
        let ret = match &lam.ret_ty {
            Some(rt) => {
                let rt = self.eval_type(rt);
                if let Some((_, exp_r)) = expected {
                    self.expect(rt, exp_r, span);
                }
                Some(rt)
            }
            None => expected.map(|(_, r)| r),
        };
        let body = match ret {
            Some(expected) => self.check_expr(&lam.body, expected),
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
        // A tensor is confined the same way: the closure may outlive the
        // chain the view belongs to (flex : freeze :: tensor : materialize).
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
            if crate::semi::ctxt::contains_tensor(self.infer.resolve(ty)) {
                let name = self.sym(self.vars.def(v).name);
                self.error(
                    span,
                    format!(
                        "closure cannot capture `{name}`: a tensor is scope-confined \
                         (materialize to an array first)"
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
        if crate::semi::ctxt::contains_tensor(self.infer.resolve(body.ty)) {
            self.error(
                span,
                "closure cannot return a tensor: tensors are scope-confined \
                 (materialize to an array first)",
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
        let (idx, field_ty, mutable) = match self.resolve_field(*def, args, Flexivity::Flex, acc) {
            Ok(resolved) => resolved,
            Err(FieldErr::Missing) => {
                self.error(span, "no such field on assignment target");
                return self.poison(span);
            }
            Err(FieldErr::Private { field, record }) => {
                self.error(
                    span,
                    format!("field `{field}` of record `{record}` is private"),
                );
                return self.poison(span);
            }
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
    /// Reads see through the arc coloring: `Arc<X>` projects and matches as
    /// `X` — only the rc discipline differs. Returns the shallow-resolved
    /// type, unwrapped once if it is an arc.
    pub(super) fn peel_arc(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.infer.shallow_resolve(ty);
        match ty.kind() {
            TyKind::Arc(inner) => self.infer.shallow_resolve(*inner),
            _ => ty,
        }
    }

    fn infer_access(
        &mut self,
        base: &surface::Expr,
        accs: &[surface::Access],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let base = self.infer_expr(base);
        let mut raw = self.infer.shallow_resolve(base.ty);
        let mut indices = Vec::new();
        for acc in accs {
            let Some((idx, ty)) = self.project_step(raw, acc, span) else {
                return self.poison(span);
            };
            indices.push(idx);
            raw = ty;
        }
        self.mk_expr(ExprKind::Proj(Box::new(base), indices), raw, span)
    }

    /// One projection step of an access chain: see through the arc coloring,
    /// resolve the field, and produce `(index, member type as seen through
    /// the base)`. Reads see through the arc coloring, but the coloring is
    /// the atomic *context* of everything inside: a bare shared field read
    /// through an arc'd base comes out at its promoted (`Arc`) type — the
    /// §3.3 whole-subtree rule at the typing level. Diagnostics are emitted
    /// here; `None` means the caller poisons.
    fn project_step(
        &mut self,
        raw: Ty<'tcx>,
        acc: &surface::Access,
        span: Option<Span>,
    ) -> Option<(u32, Ty<'tcx>)> {
        let arced = matches!(raw.kind(), TyKind::Arc(_));
        let cur = self.peel_arc(raw);
        let TyKind::Record { def, args, flex } = cur.kind() else {
            let shown = self.infer.resolve(cur);
            self.error(
                span,
                format!("cannot access a field of `{}`", self.ty_display(shown)),
            );
            return None;
        };
        let (idx, field_ty, _) = match self.resolve_field(*def, args, *flex, acc) {
            Ok(resolved) => resolved,
            Err(FieldErr::Missing) => {
                self.error(span, "no such field");
                return None;
            }
            Err(FieldErr::Private { field, record }) => {
                self.error(
                    span,
                    format!("field `{field}` of record `{record}` is private"),
                );
                return None;
            }
        };
        let field_ty = self.infer.shallow_resolve(field_ty);
        let out = if arced {
            let group = self.arc_recursive_group(*def);
            self.arc_promote_member_ty(&group, field_ty)
        } else {
            field_ty
        };
        Some((idx, out))
    }

    /// May the code being checked read or write `record_def`'s private
    /// fields? Rust's rule: a private field is accessible from the record's
    /// defining module and its descendant modules. An extern record's path is
    /// headed by its package name, which is never a prefix of a consumer
    /// module, so cross-package privacy falls out of the same prefix test.
    fn may_access_private_fields(&self, record_def: DefId) -> bool {
        let path = &self.defs.path(record_def).0;
        let def_module = &path[..path.len() - 1];
        self.defs.module().starts_with(def_module)
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
    ) -> Result<(u32, Ty<'tcx>, bool), FieldErr> {
        let record = self.records.get(&record_def).ok_or(FieldErr::Missing)?;
        let ty_params = record.ty_params.clone();
        let fields = record.fields.clone().ok_or(FieldErr::Missing)?;
        let inst =
            Instantiation::from_pairs(ty_params.iter().map(|(_, g)| *g).zip(args.iter().copied()));
        let (idx, decl_ty, mutable, vis) = match (&fields, acc) {
            (RecordFields::Named(fs), surface::Access::Named(name)) => {
                let i = fs
                    .iter()
                    .position(|(n, _, _, _)| n == name)
                    .ok_or(FieldErr::Missing)?;
                (i, fs[i].1, fs[i].2, fs[i].3)
            }
            (RecordFields::Named(fs), surface::Access::Unnamed(n)) => {
                let i = *n as usize;
                let f = fs.get(i).ok_or(FieldErr::Missing)?;
                (i, f.1, f.2, f.3)
            }
            (RecordFields::Unnamed(fs), surface::Access::Unnamed(n)) => {
                let i = *n as usize;
                let f = fs.get(i).ok_or(FieldErr::Missing)?;
                (i, f.0, f.1, f.2)
            }
            _ => return Err(FieldErr::Missing),
        };
        if vis == surface::Visibility::Private && !self.may_access_private_fields(record_def) {
            let label = match (&fields, acc) {
                (RecordFields::Named(fs), _) => self.sym(fs[idx].0).to_owned(),
                _ => idx.to_string(),
            };
            return Err(FieldErr::Private {
                field: label,
                record: self.defs.path(record_def).display(self.resolver),
            });
        }
        let field_ty = self.field_member_ty(decl_ty, mutable, base_flex, &inst);
        Ok((idx as u32, field_ty, mutable))
    }

    // ----- constructors -----

    /// A constructor-call expression; see [`Self::infer_ctor`] for the dispatch rules.
    fn infer_ctor_call(&mut self, cc: &surface::CtorCall, span: Option<Span>) -> Expr<'tcx> {
        self.infer_ctor(&cc.name, &cc.ty_args, &cc.args, span)
    }

    /// The built-in `core` package's intrinsics, spelled
    /// `core::intrinsic::<family>::<fn>`. Routes to the family checker;
    /// `None` only when the call does not target `core` at all, so ordinary
    /// resolution proceeds. Adding a family adds an arm here plus its worker.
    fn infer_intrinsic(
        &mut self,
        fc: &surface::FuncCall,
        span: Option<Span>,
    ) -> Option<Expr<'tcx>> {
        if fc.name.segments.first().map(|&k| self.sym(k)) != Some("core") {
            return None;
        }
        let segs: Vec<&str> = fc.name.segments.iter().map(|&k| self.sym(k)).collect();
        // Everything under `core` is an intrinsic; nothing else lives there.
        let ["core", "intrinsic", family] = segs.as_slice() else {
            let shown = self.path_display(&fc.name);
            self.error(
                span,
                format!("the built-in `core` package has no function `{shown}`"),
            );
            return Some(self.poison(span));
        };
        match *family {
            "panic" => Some(self.infer_panic_intrinsic(fc, span)),
            "math" => Some(self.infer_math_intrinsic(fc, span)),
            "array" => Some(self.infer_array_intrinsic(fc, span)),
            "tensor" => Some(self.infer_tensor_intrinsic(fc, span)),
            "cell" => Some(self.infer_cell_intrinsic(fc, span)),
            "str" => Some(self.infer_str_intrinsic(fc, span)),
            other => {
                self.error(
                    span,
                    format!("unknown intrinsic family `core::intrinsic::{other}`"),
                );
                Some(self.poison(span))
            }
        }
    }

    /// (PANIC) `core::intrinsic::panic::panic::<T?>()`: emit an aborting
    /// operation and inhabit the explicitly supplied or contextually expected
    /// result type. A fresh inference variable is deliberate here: unlike the
    /// error-recovery-only bottom type, it is solved by the surrounding
    /// expression and therefore reaches code generation as a concrete type.
    fn infer_panic_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::IntrinsicOp;

        let name = self.sym(fc.name.basename);
        if name != "panic" {
            self.error(
                span,
                format!("unknown panic intrinsic `core::intrinsic::panic::{name}`"),
            );
            return self.poison(span);
        }
        if !fc.args.is_empty() {
            self.error(
                span,
                format!("`panic` expects 0 arguments, got {}", fc.args.len()),
            );
            return self.poison(span);
        }
        let result = match fc.ty_args.as_slice() {
            [] | [None] => self.infer.new_hole_ty(),
            [Some(t)] => self.eval_type(t),
            _ => {
                self.error(
                    span,
                    "`panic` takes at most one type argument (the result type)",
                );
                return self.poison(span);
            }
        };
        self.mk_expr(
            ExprKind::Intrinsic {
                op: IntrinsicOp::Panic,
                args: Vec::new(),
            },
            result,
            span,
        )
    }

    /// (MATH) `core::intrinsic::math::<fn>::<T?>(args…, flag)`: the value
    /// operands share one `FloatingPoint`-bounded element type `T` (explicit
    /// or inferred; `fpowi`'s exponent is `i32`), the `is*` checks return
    /// `bool` and everything else returns `T`, and the trailing argument is a
    /// **constant** `i32` fast-math flag (`0..=127`, `0` = none, `127` = fast).
    fn infer_math_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::{FastMath, IntrinsicOp, MathFn, MathKind};

        let name = self.sym(fc.name.basename);
        let Some(func) = MathFn::parse(name) else {
            self.error(
                span,
                format!("unknown math intrinsic `core::intrinsic::math::{name}`"),
            );
            return self.poison(span);
        };

        // The shared element type: an optional single explicit type argument,
        // else inferred — either way bounded by `FloatingPoint` (`Integral`
        // for the integer ops).
        let elem = match fc.ty_args.as_slice() {
            [] | [None] => self.infer.new_hole_ty(),
            [Some(t)] => self.eval_type(t),
            _ => {
                self.error(
                    span,
                    format!("`{name}` takes at most one type argument (the operand type)"),
                );
                return self.poison(span);
            }
        };
        let kind = func.kind();
        let bound = if kind == MathKind::IntUnary {
            self.lang.integral
        } else {
            self.lang.floating_point
        };
        self.register_bound(bound, elem, span);

        // The integer ops (`ctpop`, `ctlz`, `cttz`) take no fast-math flag:
        // their MLIR forms have no such attribute.
        if kind == MathKind::IntUnary {
            if fc.args.len() != 1 {
                self.error(
                    span,
                    format!("`{name}` expects exactly 1 argument, got {}", fc.args.len()),
                );
                return self.poison(span);
            }
            let arg = self.check_expr(&fc.args[0], elem);
            let op = IntrinsicOp::Math { func, flag: 0 };
            return self.mk_expr(
                ExprKind::Intrinsic {
                    op,
                    args: vec![arg],
                },
                elem,
                span,
            );
        }

        let want = kind.value_args() + 1;
        if fc.args.len() != want {
            self.error(
                span,
                format!(
                    "`{name}` expects {want} argument(s) (including the trailing                      fast-math flag), got {}",
                    fc.args.len()
                ),
            );
            return self.poison(span);
        }
        let i32_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Signed(32));
        let (value_args, flag_arg) = fc.args.split_at(want - 1);
        let mut args = Vec::with_capacity(value_args.len());
        for (i, a) in value_args.iter().enumerate() {
            // `fpowi` raises a float to an integer power; everything else is
            // uniform over the element type.
            let expected = if kind == MathKind::Fpowi && i == 1 {
                i32_ty
            } else {
                elem
            };
            args.push(self.check_expr(a, expected));
        }
        // The flag selects `arith.fastmath` bits at compile time, so it must
        // be a literal — not merely an `i32`-typed expression.
        let flag_expr = self.check_expr(&flag_arg[0], i32_ty);
        let flag = match &flag_expr.kind {
            ExprKind::ConstInt(n) => u32::try_from(*n).ok().filter(|f| *f <= FastMath::MAX),
            _ => None,
        };
        let Some(flag) = flag else {
            self.error(
                flag_expr.span,
                format!(
                    "the fast-math flag of `{name}` must be a constant integer \
                     in 0..=127 (0 = none, 127 = fast)"
                ),
            );
            return self.poison(span);
        };

        let result = if kind == MathKind::Check {
            self.tcx.mk_bool()
        } else {
            elem
        };
        let op = IntrinsicOp::Math { func, flag };
        self.mk_expr(ExprKind::Intrinsic { op, args }, result, span)
    }

    /// (CELL) The shared-cell intrinsics, spelled `core::intrinsic::cell::*`
    /// and dispatched on the operand's cell flavor. `alloc`/`get`/`set` work
    /// on both a plain `Cell<T>` and an exclusive `RefCell<T>`; `rmw` and
    /// `in_use` require a `RefCell` operand. A cell operand is borrowed;
    /// values transferred into `alloc`/`set`, and the update closure passed
    /// to `rmw`, are consumed according to the ordinary owned calling
    /// convention. `rmw` checks `update : T -> T`, moving the stored value
    /// through the callback without the clone performed by `get`; `in_use`
    /// reads the exclusive cell's guard flag as a `bool`.
    ///
    /// `alloc` has no cell operand to dispatch on: it defaults to a plain
    /// `Cell`, and an explicit type argument naming the cell type selects
    /// the flavor (`alloc<RefCell<i64>>(1)`).
    /// (STR) `core::intrinsic::str::<fn>(…)`: `len(s: str) -> u64`,
    /// `byte_at(s: str, index: u64) -> u8` (`0` out of bounds), and
    /// `slice(s: str, offset: u64) -> str` (clamped past the end; byte
    /// granularity, so an offset inside a multi-byte character splits it).
    /// No type arguments.
    fn infer_str_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::{IntrinsicOp, StrFn};

        let name = self.sym(fc.name.basename).to_string();
        let Some(func) = StrFn::parse(&name) else {
            self.error(
                span,
                format!("unknown str intrinsic `core::intrinsic::str::{name}`"),
            );
            return self.poison(span);
        };
        if !fc.ty_args.is_empty() {
            self.error(
                span,
                format!("`core::intrinsic::str::{name}` takes no type arguments"),
            );
            return self.poison(span);
        }
        let str_ty = self.tcx.mk_str();
        let u64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Unsigned(64));
        let (params, result) = match func {
            StrFn::Len => (vec![str_ty], u64_ty),
            StrFn::ByteAt => (
                vec![str_ty, u64_ty],
                self.tcx.mk_int(crate::semi::ty::IntTy::Unsigned(8)),
            ),
            StrFn::Slice => (vec![str_ty, u64_ty], str_ty),
        };
        if fc.args.len() != params.len() {
            self.error(
                span,
                format!(
                    "`core::intrinsic::str::{name}` expects {} argument(s), got {}",
                    params.len(),
                    fc.args.len()
                ),
            );
            return self.poison(span);
        }
        let args = fc
            .args
            .iter()
            .zip(&params)
            .map(|(a, &t)| self.check_expr(a, t))
            .collect();
        self.mk_expr(
            ExprKind::Intrinsic {
                op: IntrinsicOp::Str { func },
                args,
            },
            result,
            span,
        )
    }

    fn infer_cell_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::{CellFn, IntrinsicOp};

        let name = self.sym(fc.name.basename).to_string();
        let Some(func) = CellFn::parse(&name) else {
            self.error(
                span,
                format!("unknown cell intrinsic `core::intrinsic::cell::{name}`"),
            );
            return self.poison(span);
        };
        if func != CellFn::Alloc && !fc.ty_args.is_empty() {
            self.error(
                span,
                format!("`core::intrinsic::cell::{name}` takes no type arguments"),
            );
            return self.poison(span);
        }

        let op = IntrinsicOp::Cell { func };
        match func {
            CellFn::Alloc => {
                let [value] = fc.args.as_slice() else {
                    self.error(span, "`core::intrinsic::cell::alloc` expects one argument");
                    return self.poison(span);
                };
                let result = match fc.ty_args.as_slice() {
                    [] | [None] => None,
                    [Some(t)] => {
                        let ty = self.eval_type(t);
                        let resolved = self.infer.shallow_resolve(ty);
                        let TyKind::Cell { .. } = *resolved.kind() else {
                            let shown = self.infer.resolve(resolved);
                            self.error(
                                span,
                                format!(
                                    "the type argument of `core::intrinsic::cell::alloc` must \
                                     be `Cell<T>` or `RefCell<T>`, found `{}`",
                                    self.ty_display(shown)
                                ),
                            );
                            return self.poison(span);
                        };
                        Some(resolved)
                    }
                    _ => {
                        self.error(
                            span,
                            "`core::intrinsic::cell::alloc` takes at most one type argument \
                             (the cell type)",
                        );
                        return self.poison(span);
                    }
                };
                let value = match result {
                    Some(cell_ty) => {
                        let TyKind::Cell { elem, .. } = *cell_ty.kind() else {
                            unreachable!("checked above");
                        };
                        self.check_expr(value, elem)
                    }
                    None => self.infer_expr(value),
                };
                let result = result.unwrap_or_else(|| self.tcx.mk_cell(value.ty, CellKind::Plain));
                self.mk_expr(
                    ExprKind::Intrinsic {
                        op,
                        args: vec![value],
                    },
                    result,
                    span,
                )
            }
            CellFn::Get | CellFn::InUse => {
                let [cell] = fc.args.as_slice() else {
                    self.error(
                        span,
                        format!("`core::intrinsic::cell::{name}` expects one argument"),
                    );
                    return self.poison(span);
                };
                let cell = self.infer_expr(cell);
                let Some(elem) = self.cell_element(cell.ty, func, &name, span) else {
                    return self.poison(span);
                };
                let result = if func == CellFn::InUse {
                    self.tcx.mk_bool()
                } else {
                    elem
                };
                self.mk_expr(
                    ExprKind::Intrinsic {
                        op,
                        args: vec![cell],
                    },
                    result,
                    span,
                )
            }
            CellFn::Set | CellFn::Rmw => {
                let [cell, value] = fc.args.as_slice() else {
                    self.error(
                        span,
                        format!("`core::intrinsic::cell::{name}` expects two arguments"),
                    );
                    return self.poison(span);
                };
                let cell = self.infer_expr(cell);
                let Some(elem) = self.cell_element(cell.ty, func, &name, span) else {
                    return self.poison(span);
                };
                let expected = if func == CellFn::Rmw {
                    self.tcx.mk_closure(&[elem], elem)
                } else {
                    elem
                };
                let value = self.check_expr(value, expected);
                let result = self.tcx.mk_unit();
                self.mk_expr(
                    ExprKind::Intrinsic {
                        op,
                        args: vec![cell, value],
                    },
                    result,
                    span,
                )
            }
            CellFn::Rdlock => {
                // `rdlock(cell, reader)`: the reader runs over a *clone* of
                // the element inside the shared read lock — `reader : T -> ρ`
                // for a fresh `ρ`, which is the operation's result. The clone
                // is consumed by the reader like any owned argument.
                let [cell, reader] = fc.args.as_slice() else {
                    self.error(
                        span,
                        format!("`core::intrinsic::cell::{name}` expects two arguments"),
                    );
                    return self.poison(span);
                };
                let cell = self.infer_expr(cell);
                let Some(elem) = self.cell_element(cell.ty, func, &name, span) else {
                    return self.poison(span);
                };
                let result = self.infer.new_hole_ty();
                let expected = self.tcx.mk_closure(&[elem], result);
                let reader = self.check_expr(reader, expected);
                self.mk_expr(
                    ExprKind::Intrinsic {
                        op,
                        args: vec![cell, reader],
                    },
                    result,
                    span,
                )
            }
        }
    }

    /// Return the element of a cell operand, diagnosing a non-cell value and
    /// an exclusive-only operation applied to a plain `Cell`.
    fn cell_element(
        &mut self,
        ty: Ty<'tcx>,
        func: crate::intrinsic::CellFn,
        method: &str,
        span: Option<Span>,
    ) -> Option<Ty<'tcx>> {
        let resolved = self.infer.shallow_resolve(ty);
        if let TyKind::Cell { elem, kind } = *resolved.kind() {
            if !func.supported_on(kind) {
                let shown = self.infer.resolve(resolved);
                self.error(
                    span,
                    format!(
                        "`core::intrinsic::cell::{method}` is not supported on `{}`; \
                         it requires {}",
                        self.ty_display(shown),
                        func.requirement()
                    ),
                );
                return None;
            }
            Some(elem)
        } else {
            let shown = self.infer.resolve(resolved);
            self.error(
                span,
                format!(
                    "`core::intrinsic::cell::{method}` expects a cell as its first argument, \
                     found `{}`",
                    self.ty_display(shown)
                ),
            );
            None
        }
    }

    /// (ARR) The `core::intrinsic::array` family (issue #344) — special
    /// *vararg* functions over statically shaped arrays:
    ///
    /// * `splat<[T; e…]>(v)` / `tabulate<[T; e…]>(|i…| e)` construct an
    ///   array; the explicit array type argument carries the shape (there is
    ///   nothing to infer it from);
    /// * `get(a, i…)` / `set(a, i…, v)` / `fold(a, init, |acc, x| e)` take the
    ///   array as their first argument and one `i64` index per dimension.
    fn infer_array_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::ArrayFn;
        let name = self.sym(fc.name.basename).to_string();
        match ArrayFn::parse(&name) {
            Some(ArrayFn::Splat | ArrayFn::Tabulate) => self.infer_array_ctor(fc, span),
            Some(_) => {
                if !fc.ty_args.is_empty() {
                    self.error(
                        span,
                        format!(
                            "`core::intrinsic::array::{name}` takes no type arguments \
                             (the shape comes from its array argument)"
                        ),
                    );
                    return self.poison(span);
                }
                if fc.args.is_empty() {
                    self.error(
                        span,
                        format!(
                            "`core::intrinsic::array::{name}` expects an array as its \
                             first argument"
                        ),
                    );
                    return self.poison(span);
                }
                let base = self.infer_expr(&fc.args[0]);
                if !matches!(
                    self.infer.shallow_resolve(base.ty).kind(),
                    TyKind::Array { .. }
                ) {
                    let shown = self.infer.resolve(base.ty);
                    self.error(
                        span,
                        format!(
                            "`core::intrinsic::array::{name}` expects an array as its \
                             first argument, found `{}`",
                            self.ty_display(shown)
                        ),
                    );
                    return self.poison(span);
                }
                self.infer_array_op(base, &name, &fc.args[1..], span)
            }
            None => {
                self.error(
                    span,
                    format!("unknown array intrinsic `core::intrinsic::array::{name}`"),
                );
                self.poison(span)
            }
        }
    }

    /// The constructing array intrinsics: `splat` checks its one argument
    /// against the element type, `tabulate` checks its kernel — an ordinary
    /// closure taking one `i64` index per dimension — against
    /// `(i64, …) -> elem`. Any closure-typed expression works (a literal
    /// lambda fuses into the loop nest under optimization; anything else is
    /// called per element).
    fn infer_array_ctor(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::ArrayFn;
        let method = self.sym(fc.name.basename).to_string();
        let op = ArrayFn::parse(&method).expect("routed here for splat/tabulate only");
        let [Some(targ)] = fc.ty_args.as_slice() else {
            self.error(
                span,
                format!(
                    "`core::intrinsic::array::{method}` requires one explicit array type \
                     argument, e.g. `core::intrinsic::array::{method}<[f64; 512]>(…)`"
                ),
            );
            return self.poison(span);
        };
        let arr = self.eval_type(targ);
        let TyKind::Array { elem, dims } = *arr.kind() else {
            if !matches!(arr.kind(), TyKind::Bottom) {
                self.error(
                    Some(targ.span()),
                    format!(
                        "`core::intrinsic::array::{method}` expects an array type argument \
                         (`[T; extents…]`), found `{}`",
                        self.ty_display(arr)
                    ),
                );
            }
            return self.poison(span);
        };
        // A dynamic dimension (`[T; _, 4]`) has no extent in the type; its
        // runtime value is a leading `u64` operand of the constructor, in
        // dimension order.
        let n_dynamic = dims
            .iter()
            .filter(|&&d| d == crate::semi::ty::DYNAMIC_EXTENT)
            .count();
        if fc.args.len() != n_dynamic + 1 {
            let extents = if n_dynamic == 0 {
                String::new()
            } else {
                format!(" and {n_dynamic} leading dynamic extent(s)")
            };
            self.error(
                span,
                format!("`core::intrinsic::array::{method}` expects exactly one argument{extents}"),
            );
            return self.poison(span);
        }
        let u64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Unsigned(64));
        let extent_args: Vec<Expr<'tcx>> = fc.args[..n_dynamic]
            .iter()
            .map(|a| self.check_expr(a, u64_ty))
            .collect();
        let payload_arg = &fc.args[n_dynamic];
        // `tabulate` fills the payload through the raw view with no
        // per-element ownership hooks, so rc elements stay out of it; a
        // deferred (generic) element re-checks at monomorphization. `splat`
        // supports them (its lowering retains per stored slot).
        if op == ArrayFn::Tabulate
            && !crate::semi::ty_eval::is_plain_scalar_or_deferred(self.infer.shallow_resolve(elem))
        {
            self.error(
                span,
                format!(
                    "`tabulate` does not support rc element types yet; found `{}` \
                     (build with `splat` and `set` instead)",
                    self.ty_display(elem)
                ),
            );
            return self.poison(span);
        }
        match op {
            ArrayFn::Splat => {
                let v = self.check_expr(payload_arg, elem);
                let mut args = extent_args;
                args.push(v);
                self.mk_expr(ExprKind::ArrayOp { op, args }, arr, span)
            }
            ArrayFn::Tabulate => {
                let i64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Signed(64));
                let params = vec![i64_ty; dims.len()];
                let kernel_ty = self.tcx.mk_closure(&params, elem);
                let kernel = self.check_expr(payload_arg, kernel_ty);
                let mut args = extent_args;
                args.push(kernel);
                self.mk_expr(ExprKind::ArrayOp { op, args }, arr, span)
            }
            _ => unreachable!("filtered above"),
        }
    }

    /// The accessing array intrinsics over an already-inferred array `base`:
    /// `get(a, i…)`, `set(a, i…, v)`, `fold(a, init, |acc, x| e)`. Indices are
    /// `i64` (checked against the extents at runtime); `get` borrows and
    /// yields the element, `set` consumes the array and yields the updated
    /// one, `fold` borrows and reduces row-major.
    fn infer_array_op(
        &mut self,
        base: Expr<'tcx>,
        method: &str,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        use crate::intrinsic::ArrayFn;
        let bty = self.infer.shallow_resolve(base.ty);
        let TyKind::Array { elem, dims } = *bty.kind() else {
            unreachable!("routed here only for array bases");
        };
        // `fold` runs the payload through a borrowed view with no
        // per-element ownership hooks, so rc elements stay out of it (the
        // supported traffic is `splat`/`get`/`set`). A deferred (generic)
        // element re-checks at monomorphization.
        if method == "fold"
            && !crate::semi::ty_eval::is_plain_scalar_or_deferred(self.infer.shallow_resolve(elem))
        {
            self.error(
                span,
                format!(
                    "`fold` does not support rc element types yet; found `{}` \
                     (use `get` in a recursive loop instead)",
                    self.ty_display(elem)
                ),
            );
            return self.poison(span);
        }
        let rank = dims.len();
        let i64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Signed(64));
        match method {
            "get" => {
                if args.len() != rank {
                    self.error(
                        span,
                        format!(
                            "`get` on `{}` expects {rank} index argument(s), got {}",
                            self.ty_display(bty),
                            args.len()
                        ),
                    );
                    return self.poison(span);
                }
                let mut hargs = vec![base];
                for a in args {
                    hargs.push(self.check_expr(a, i64_ty));
                }
                self.mk_expr(
                    ExprKind::ArrayOp {
                        op: ArrayFn::Get,
                        args: hargs,
                    },
                    elem,
                    span,
                )
            }
            "set" => {
                if args.len() != rank + 1 {
                    self.error(
                        span,
                        format!(
                            "`set` on `{}` expects {rank} index argument(s) and a value, got {} argument(s)",
                            self.ty_display(bty),
                            args.len()
                        ),
                    );
                    return self.poison(span);
                }
                let mut hargs = vec![base];
                for a in &args[..rank] {
                    hargs.push(self.check_expr(a, i64_ty));
                }
                hargs.push(self.check_expr(&args[rank], elem));
                self.mk_expr(
                    ExprKind::ArrayOp {
                        op: ArrayFn::Set,
                        args: hargs,
                    },
                    bty,
                    span,
                )
            }
            "fold" => {
                if args.len() != 2 {
                    self.error(
                        span,
                        "`fold` expects an initial accumulator and a kernel closure",
                    );
                    return self.poison(span);
                }
                let init = self.infer_expr(&args[0]);
                let acc_ty = init.ty;
                let kernel_ty = self.tcx.mk_closure(&[acc_ty, elem], acc_ty);
                let kernel = self.check_expr(&args[1], kernel_ty);
                self.mk_expr(
                    ExprKind::ArrayOp {
                        op: ArrayFn::Fold,
                        args: vec![base, init, kernel],
                    },
                    acc_ty,
                    span,
                )
            }
            "dim" => {
                if args.len() != 1 {
                    self.error(span, "`dim` expects one dimension-index argument");
                    return self.poison(span);
                }
                let u64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Unsigned(64));
                let k = self.check_expr(&args[0], u64_ty);
                self.mk_expr(
                    ExprKind::ArrayOp {
                        op: ArrayFn::Dim,
                        args: vec![base, k],
                    },
                    u64_ty,
                    span,
                )
            }
            _ => unreachable!("routed here only for get/set/fold/dim"),
        }
    }

    /// The tensor boundary intrinsics (docs/design/tensor-kernels.md):
    /// `of(a)` views an array as a tensor of the same shape (consuming the
    /// reference — the tensor owns the rooted dup, which is what keeps a
    /// later in-place update of the array from mutating the viewed buffer);
    /// `materialize(t)` builds a fresh array from the tensor's value;
    /// `dim(t, k)` reads the `k`-th extent as `u64`.
    fn infer_tensor_intrinsic(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        use crate::intrinsic::TensorFn;
        let name = self.sym(fc.name.basename).to_string();
        let Some(op) = TensorFn::parse(&name) else {
            self.error(
                span,
                format!("unknown tensor intrinsic `core::intrinsic::tensor::{name}`"),
            );
            return self.poison(span);
        };
        if !fc.ty_args.is_empty() {
            self.error(
                span,
                format!(
                    "`core::intrinsic::tensor::{name}` takes no type arguments \
                     (the shape comes from its argument)"
                ),
            );
            return self.poison(span);
        }
        let expected_args = match op {
            TensorFn::Of | TensorFn::Materialize => 1,
            TensorFn::Dim => 2,
        };
        if fc.args.len() != expected_args {
            self.error(
                span,
                format!(
                    "`core::intrinsic::tensor::{name}` expects exactly \
                     {expected_args} argument(s), got {}",
                    fc.args.len()
                ),
            );
            return self.poison(span);
        }
        let base = self.infer_expr(&fc.args[0]);
        let bty = self.infer.shallow_resolve(base.ty);
        match op {
            TensorFn::Of => {
                let TyKind::Array { elem, dims } = *bty.kind() else {
                    let shown = self.infer.resolve(base.ty);
                    self.error(
                        span,
                        format!(
                            "`of` expects an array argument, found `{}`",
                            self.ty_display(shown)
                        ),
                    );
                    return self.poison(span);
                };
                let tensor = self.tcx.mk_tensor(elem, dims);
                self.mk_expr(
                    ExprKind::TensorOp {
                        op,
                        args: vec![base],
                    },
                    tensor,
                    span,
                )
            }
            TensorFn::Materialize => {
                let TyKind::Tensor { elem, dims } = *bty.kind() else {
                    let shown = self.infer.resolve(base.ty);
                    self.error(
                        span,
                        format!(
                            "`materialize` expects a tensor argument, found `{}`",
                            self.ty_display(shown)
                        ),
                    );
                    return self.poison(span);
                };
                let arr = self.tcx.mk_array(elem, dims);
                self.mk_expr(
                    ExprKind::TensorOp {
                        op,
                        args: vec![base],
                    },
                    arr,
                    span,
                )
            }
            TensorFn::Dim => {
                if !matches!(bty.kind(), TyKind::Tensor { .. }) {
                    let shown = self.infer.resolve(base.ty);
                    self.error(
                        span,
                        format!(
                            "`dim` expects a tensor argument, found `{}` \
                             (use `core::intrinsic::array::dim` for arrays)",
                            self.ty_display(shown)
                        ),
                    );
                    return self.poison(span);
                }
                let u64_ty = self.tcx.mk_int(crate::semi::ty::IntTy::Unsigned(64));
                let k = self.check_expr(&fc.args[1], u64_ty);
                self.mk_expr(
                    ExprKind::TensorOp {
                        op,
                        args: vec![base, k],
                    },
                    u64_ty,
                    span,
                )
            }
        }
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
        // Expand `import` bindings before the builtin-spelling checks so an
        // imported `Nullable`/`Arc`/enum qualifier dispatches the same as
        // its full spelling.
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        let qualifier = path.segments.last().copied();
        // The built-in nullable constructors, canonically
        // `core::intrinsic::nullable::Nullable::…` and prelude-imported as bare
        // `Nullable::…`; a user record named `Nullable` shadows the bare form.
        if qualifier.map(|k| self.sym(k)) == Some("Nullable")
            && self.ctor_qualifier_prelude_applies(path, "nullable")
        {
            return self.infer_nullable(self.sym(path.basename), args, span);
        }
        // The built-in arc-coloring constructors, canonically
        // `core::intrinsic::arc::Arc` and prelude-imported as bare `Arc`:
        // `Arc<Inner>{…}` builds the `[shared]` struct `Inner` behind an
        // atomically counted box, `Arc<Enum<…>>::Variant{…}` the enum flavor.
        // `Arc<X>` is a specially colored version of `X`, so the constructor
        // is the inner record's own, at the arc'd type. A user record named
        // `Arc` shadows the bare form like any other prelude name.
        if self.sym(path.basename) == "Arc"
            && (path.segments.is_empty() && self.defs.resolve_record(path.basename).is_none()
                || self.is_core_intrinsic_prefix(&path.segments, "arc"))
        {
            return self.infer_arc_ctor(ty_args, None, args, span);
        }
        if qualifier.map(|k| self.sym(k)) == Some("Arc")
            && self.ctor_qualifier_prelude_applies(path, "arc")
        {
            return self.infer_arc_ctor(ty_args, Some(path.basename), args, span);
        }
        if let Some(enum_key) = qualifier {
            // The qualifier resolving to an enum wins (`Enum::Variant`); a
            // qualifier that is a module path instead falls through to a
            // module-qualified struct constructor (`m::Foo{…}`).
            if let Some(def) = self.resolve_ctor_qualifier(path)
                && matches!(self.records[&def].fields, Some(RecordFields::Variants(_)))
            {
                self.record_use(enum_key);
                return self.infer_variant(
                    def,
                    enum_key,
                    path.basename,
                    ty_args,
                    args,
                    false,
                    span,
                );
            }
            let Some(def) = self.resolve_record_ref(path) else {
                // Either the qualifier (`dep::Enum::Variant`) or the record
                // itself (`dep::Struct{…}`) may be the private extern hit.
                if let Some(msg) = self
                    .extern_private_ctor_msg(path)
                    .or_else(|| self.extern_private_msg(path, DefKind::Record))
                {
                    self.error(span, msg);
                    return self.poison(span);
                }
                let hint = self.record_suggestion(enum_key);
                let shown = self.path_display(path);
                self.error(span, format!("unknown constructor `{shown}`{hint}"));
                return self.poison(span);
            };
            self.record_use(path.basename);
            return self.infer_struct_def(def, path.basename, ty_args, args, false, span);
        }
        // A bare struct constructor.
        self.infer_struct(path.basename, ty_args, args, span)
    }

    /// (ARC) `Γ ⊢ Arc<Inner>{args} ⇒ (CompoundCall{…} : Arc<R>)` and
    /// `Γ ⊢ Arc<Enum<…>>::Variant{args} ⇒ (VariantCall{…} : Arc<R>)`: the inner
    /// type names the `[shared]` record being built; the payload is checked
    /// exactly like the plain constructor and only the node's type is wrapped.
    /// The box is atomically counted from birth — an arc is never re-colored
    /// from an existing plain shared value.
    fn infer_arc_ctor(
        &mut self,
        ty_args: &[Option<surface::Type>],
        variant: Option<TokenKey>,
        args: &[(Option<TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let [Some(inner)] = ty_args else {
            self.error(
                span,
                "`Arc` construction takes exactly one explicit type argument \
                 (the `[shared]` record being built)",
            );
            return self.poison(span);
        };
        let surface::TypeKind::TypeExpr(ipath, iargs) = inner.kind() else {
            self.error(
                span,
                "`Arc` constructs a `[shared]` record; this inner type is not one",
            );
            return self.poison(span);
        };
        let Some(def) = self.resolve_record_ref(&ipath) else {
            if let Some(msg) = self.extern_private_msg(&ipath, DefKind::Record) {
                self.error(span, msg);
                return self.poison(span);
            }
            let shown = self.path_display(&ipath);
            self.error(span, format!("unknown record `{shown}`"));
            return self.poison(span);
        };
        if !matches!(
            self.records[&def].default_cap,
            super::ctxt::DefaultCap::Shared
        ) {
            let shown = self.path_display(&ipath);
            self.error(
                span,
                format!("`Arc` constructs a `[shared]` record; `{shown}` is not one"),
            );
            return self.poison(span);
        }
        self.record_use(ipath.basename);
        let iargs: Vec<Option<surface::Type>> = iargs.iter().cloned().map(Some).collect();
        let mut e = match variant {
            None => self.infer_struct_def(def, ipath.basename, &iargs, args, true, span),
            Some(v) => self.infer_variant(def, ipath.basename, v, &iargs, args, true, span),
        };
        // Wrap only a successfully built constructor; an error path stays the
        // poison it already is.
        if matches!(e.ty.kind(), TyKind::Record { .. }) {
            // The coloring site: the arc's contents must be `Sync`. (Ctors
            // run during body checking, after record fields populate; a
            // generic-blocked verdict defers to the mono backstop.)
            self.report_arc_wf(e.ty, span);
            e.ty = self.tcx.mk_arc(e.ty);
        }
        e
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
        let resolved = self
            .defs
            .resolve_record(name)
            .or_else(|| self.prelude_lookup(name, crate::semi::lang::LangItemKind::Record));
        let Some(def) = resolved else {
            let hint = self.record_suggestion(name);
            self.error(span, format!("unknown type `{}`{hint}", self.sym(name)));
            return self.poison(span);
        };
        self.record_use(name);
        self.infer_struct_def(def, name, ty_args, args, false, span)
    }

    /// The resolved-def core of [`Self::infer_struct`] (shared with the
    /// module-qualified constructor path).
    fn infer_struct_def(
        &mut self,
        def: DefId,
        name: TokenKey,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        arc: bool,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        if self.lang.is_former_anchor(def) {
            self.error(
                span,
                format!(
                    "`{}` is a built-in type; construct it with its own syntax, not as a struct",
                    self.sym(name)
                ),
            );
            return self.poison(span);
        }
        let record = self.records[&def].clone();
        if matches!(record.fields, Some(RecordFields::Variants(_))) {
            self.error(
                span,
                format!("`{}` is an enum, not a struct", self.sym(name)),
            );
            return self.poison(span);
        }
        if matches!(record.fields, Some(RecordFields::Opaque)) {
            self.error(
                span,
                format!(
                    "`{}` is an opaque `#[ffi]` record and has no constructor; \
                     obtain values through its `#[ffi(import)]` functions",
                    self.sym(name)
                ),
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
        // Constructing a record supplies every field, so a single private
        // field makes the constructor itself module-private. One error listing
        // the private fields, then checking continues (no poison — matches the
        // regional check above). Variant construction is exempt by design:
        // variant payloads carry no per-field visibility.
        if !self.may_access_private_fields(def) {
            let private: Vec<String> = match &record.fields {
                Some(RecordFields::Named(fs)) => fs
                    .iter()
                    .filter(|(_, _, _, v)| *v == surface::Visibility::Private)
                    .map(|(n, _, _, _)| format!("`{}`", self.sym(*n)))
                    .collect(),
                Some(RecordFields::Unnamed(fs)) => fs
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, _, v))| *v == surface::Visibility::Private)
                    .map(|(i, _)| format!("`{i}`"))
                    .collect(),
                _ => Vec::new(),
            };
            if !private.is_empty() {
                self.error(
                    span,
                    format!(
                        "cannot construct `{}` outside its module: field(s) {} are private",
                        self.defs.path(def).display(self.resolver),
                        private.join(", ")
                    ),
                );
            }
        }
        let inst = self.instantiate(&record.generics_as_decls(), ty_args, span);
        let mut field_tys = self.field_types(&record.fields, &inst);
        if arc {
            // The arc'd constructor is the coloring site: its atomic context
            // flows into the fields (§3.3) — a bare same-group shared field
            // is expected at its promoted `Arc` type, so the whole spine is
            // built atomic from birth (the closed colored world).
            let group = self.arc_recursive_group(def);
            for (_, ty) in field_tys.iter_mut() {
                *ty = self.arc_promote_member_ty(&group, *ty);
            }
        }

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
    #[allow(clippy::too_many_arguments)]
    fn infer_variant(
        &mut self,
        def: DefId,
        enum_name: TokenKey,
        variant: TokenKey,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<TokenKey>, surface::Expr)],
        arc: bool,
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
        // Under the arc'd constructor, the atomic context flows into the
        // payload (§3.3): bare same-group shared fields are expected at their
        // promoted `Arc` types — the spine is built atomic from birth.
        let group = if arc {
            self.arc_recursive_group(def)
        } else {
            rustc_hash::FxHashSet::default()
        };
        let payload_tys: Vec<Ty<'tcx>> = payload
            .iter()
            .map(|t| {
                let t = self.infer.instantiate_ty(*t, &inst);
                if arc {
                    self.arc_promote_member_ty(&group, t)
                } else {
                    t
                }
            })
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
                .map(|(n, t, is_field, _)| {
                    (
                        Some(*n),
                        self.field_member_ty(*t, *is_field, base_flex, inst),
                    )
                })
                .collect(),
            Some(RecordFields::Unnamed(fs)) => fs
                .iter()
                .map(|(t, is_field, _)| {
                    (None, self.field_member_ty(*t, *is_field, base_flex, inst))
                })
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
            Intrinsic { op, args } => Intrinsic {
                op,
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
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
            ArrayOp { op, args } => ArrayOp {
                op,
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            TensorOp { op, args } => TensorOp {
                op,
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            TraitCall {
                trait_def,
                method,
                ty_args,
                args,
            } => TraitCall {
                trait_def,
                method,
                ty_args: ty_args.into_iter().map(|t| self.zonk_ty(t, span)).collect(),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            other => other,
        }
    }
}

/// Map a surface arithmetic/boolean/bitwise operator to the HIR form.
fn arith_op(op: BinOp) -> ArithOp {
    match op {
        BinOp::Add => ArithOp::Add,
        BinOp::Sub => ArithOp::Sub,
        BinOp::Mul => ArithOp::Mul,
        BinOp::Div => ArithOp::Div,
        BinOp::Mod => ArithOp::Mod,
        // `&&`/`||` deliberately do not route here: (LOGIC) desugars them to
        // (IF) so they short-circuit, and mapping them onto the eager
        // `ArithOp::And`/`Or` is exactly the bug that desugaring fixed.
        BinOp::And | BinOp::Or => {
            unreachable!("`&&`/`||` elaborate to `If`, never to `Arith`")
        }
        BinOp::BitAnd => ArithOp::BitAnd,
        BinOp::BitOr => ArithOp::BitOr,
        BinOp::BitXor => ArithOp::BitXor,
        BinOp::Shl => ArithOp::Shl,
        BinOp::Shr => ArithOp::Shr,
        _ => unreachable!("not an arithmetic operator"),
    }
}

fn cmp_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Equ => "==",
        BinOp::Neq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Lte => "<=",
        BinOp::Gte => ">=",
        _ => unreachable!("not a comparison"),
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

/// Collect the free variables of an expression tree: every `Var` occurrence
/// minus every variable the tree itself binds (`let`s, pattern bindings,
/// nested closure parameters). Variable ids are fresh per binder within a
/// function body, so a single flat bound set is enough — no scoping needed.
fn free_vars<'tcx>(e: &Expr<'tcx>, out: &mut Vec<VarId>) {
    let mut used = Vec::new();
    let mut bound = Vec::new();
    var_occurrences(e, &mut used, &mut bound);
    for v in used {
        if !bound.contains(&v) && !out.contains(&v) {
            out.push(v);
        }
    }
}

fn var_occurrences<'tcx>(e: &Expr<'tcx>, used: &mut Vec<VarId>, bound: &mut Vec<VarId>) {
    use ExprKind::*;
    match &e.kind {
        Var(v) => used.push(*v),
        Intrinsic { args, .. } => {
            for a in args {
                var_occurrences(a, used, bound);
            }
        }
        Negate(e) | Not(e) | Cast(e, _) | RegionRun(e) | Proj(e, _) => {
            var_occurrences(e, used, bound)
        }
        Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => {
            var_occurrences(l, used, bound);
            var_occurrences(r, used, bound);
        }
        If(c, t, f) => {
            var_occurrences(c, used, bound);
            var_occurrences(t, used, bound);
            var_occurrences(f, used, bound);
        }
        Let { var, value, .. } => {
            var_occurrences(value, used, bound);
            bound.push(*var);
        }
        Seq(es) => es.iter().for_each(|e| var_occurrences(e, used, bound)),
        FuncCall { args, .. }
        | TraitCall { args, .. }
        | CompoundCall { args, .. }
        | VariantCall { args, .. } => args.iter().for_each(|e| var_occurrences(e, used, bound)),
        NullableCall(e) => {
            if let Some(e) = e {
                var_occurrences(e, used, bound)
            }
        }
        ClosureCall { target, args } => {
            var_occurrences(target, used, bound);
            args.iter().for_each(|e| var_occurrences(e, used, bound));
        }
        Closure(c) => {
            bound.extend(c.params.iter().map(|&(v, _)| v));
            var_occurrences(&c.body, used, bound);
        }
        ArrayOp { args, .. } | TensorOp { args, .. } => {
            args.iter().for_each(|e| var_occurrences(e, used, bound))
        }
        Match(scrut, tree) => {
            var_occurrences(scrut, used, bound);
            tree_occurrences(tree, used, bound);
        }
        GlobalStr(_) | ConstChar(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Poison => {}
    }
}

fn tree_occurrences<'tcx>(
    t: &crate::semi::hir::DecisionTree<'tcx>,
    used: &mut Vec<VarId>,
    bound: &mut Vec<VarId>,
) {
    use crate::semi::hir::{DecisionTree, SwitchCases};
    match t {
        DecisionTree::Uncovered | DecisionTree::Unreachable => {}
        DecisionTree::Leaf { body, bindings } => {
            bound.extend(bindings.iter().map(|&(v, _)| v));
            var_occurrences(body, used, bound);
        }
        DecisionTree::Guard {
            bindings,
            guard,
            success,
            failure,
        } => {
            bound.extend(bindings.iter().map(|&(v, _)| v));
            var_occurrences(guard, used, bound);
            tree_occurrences(success, used, bound);
            tree_occurrences(failure, used, bound);
        }
        DecisionTree::Switch { cases, .. } => match cases {
            SwitchCases::Int { cases, default } => {
                cases
                    .iter()
                    .for_each(|(_, t)| tree_occurrences(t, used, bound));
                tree_occurrences(default, used, bound);
            }
            SwitchCases::Bool { if_true, if_false } => {
                tree_occurrences(if_true, used, bound);
                tree_occurrences(if_false, used, bound);
            }
            SwitchCases::Char { cases, default } => {
                cases
                    .iter()
                    .for_each(|(_, t)| tree_occurrences(t, used, bound));
                tree_occurrences(default, used, bound);
            }
            SwitchCases::Ctor(subtrees) => subtrees
                .iter()
                .for_each(|t| tree_occurrences(t, used, bound)),
            SwitchCases::String { cases, default } => {
                cases
                    .iter()
                    .for_each(|(_, t)| tree_occurrences(t, used, bound));
                tree_occurrences(default, used, bound);
            }
            SwitchCases::Nullable { non_null, null } => {
                tree_occurrences(non_null, used, bound);
                tree_occurrences(null, used, bound);
            }
        },
    }
}

impl<'tcx> super::ctxt::Record<'tcx> {
    /// The record's generics as `(name, id)` declaration pairs.
    fn generics_as_decls(&self) -> Vec<(TokenKey, GenericId)> {
        self.ty_params.clone()
    }
}
