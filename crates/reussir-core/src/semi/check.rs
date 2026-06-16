//! The bidirectional type checker: surface expressions → typed Semi HIR.

use crate::semi::infer::Instantiation;
use crate::semi::traits::{Obligation, TraitId, TraitRef};
use crate::semi::ty::{GenericId, Ty, TyKind};
use crate::surface::{self, BinOp, Const, Span, UnaryOp};

use super::ctxt::{Elaborator, RecordFields};
use super::hir::{ArithOp, ClosureExpr, CmpOp, Expr, ExprKind, Function, VarId};

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    pub(super) fn check_function(&mut self, func: &surface::Function, span: Option<Span>) {
        let Some(proto) = self.functions.get(self.sym(func.name)).cloned() else {
            return;
        };
        self.enter_function(&proto.generics);

        let mut params = Vec::new();
        for (name, ty) in &proto.params {
            let var = self.vars.fresh(name, *ty, None);
            params.push((name.clone(), var, *ty));
        }

        let body = func
            .body
            .as_ref()
            .map(|b| self.check_expr(b, proto.return_ty));
        self.resolve_obligations();
        let body = body.map(|b| self.zonk_expr(b));
        let return_ty = self.infer.resolve(proto.return_ty);

        self.elaborated.push(Function {
            name: proto.name,
            generics: proto.generics,
            params,
            return_ty,
            is_regional: proto.is_regional,
            body,
            span,
        });
    }

    fn mk_expr(&mut self, kind: ExprKind<'tcx>, ty: Ty<'tcx>, span: Option<Span>) -> Expr<'tcx> {
        let id = self.fresh_expr_id();
        Expr { kind, ty, span, id }
    }

    fn poison(&mut self, span: Option<Span>) -> Expr<'tcx> {
        let ty = self.tcx.mk(TyKind::Bottom);
        self.mk_expr(ExprKind::Poison, ty, span)
    }

    /// Unify `found` with `expected`, reporting a mismatch.
    fn expect(&mut self, found: Ty<'tcx>, expected: Ty<'tcx>, span: Option<Span>) {
        if self.infer.unify(expected, found).is_err() {
            let e = self.infer.resolve(expected);
            let f = self.infer.resolve(found);
            self.error(
                span,
                format!("type mismatch: expected `{e:?}`, found `{f:?}`"),
            );
        }
    }

    /// Check an expression against an expected type.
    pub(super) fn check_expr(&mut self, e: &surface::Expr, expected: Ty<'tcx>) -> Expr<'tcx> {
        let h = self.infer_expr(e);
        self.expect(h.ty, expected, h.span);
        h
    }

    /// Infer the type of an expression, producing typed HIR.
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

    fn infer_const(&mut self, c: &Const, span: Option<Span>) -> Expr<'tcx> {
        match c {
            Const::ConstInt(i) => {
                let hole = self.infer.new_hole_ty();
                self.register_bound(self.builtins.integral, hole, span);
                self.mk_expr(ExprKind::ConstInt(*i as i128), hole, span)
            }
            Const::ConstDouble(f) => {
                let hole = self.infer.new_hole_ty();
                self.register_bound(self.builtins.floating_point, hole, span);
                self.mk_expr(ExprKind::ConstFloat(*f), hole, span)
            }
            Const::ConstString(s) => {
                let token = self.strings.allocate(s);
                let ty = self.tcx.mk_str();
                self.mk_expr(ExprKind::GlobalStr(token), ty, span)
            }
            Const::ConstBool(b) => {
                let ty = self.tcx.mk_bool();
                self.mk_expr(ExprKind::ConstBool(*b), ty, span)
            }
        }
    }

    fn register_bound(&mut self, trait_id: TraitId, ty: Ty<'tcx>, span: Option<Span>) {
        self.fulfill.register(
            Obligation::Trait(TraitRef {
                trait_id,
                args: vec![ty],
            }),
            span,
        );
    }

    fn infer_var(&mut self, path: &surface::Path, span: Option<Span>) -> Expr<'tcx> {
        if path.segments.is_empty() {
            let name = self.sym(path.basename);
            if let Some((id, ty)) = self.vars.lookup(name) {
                return self.mk_expr(ExprKind::Var(id), ty, span);
            }
            self.error(span, format!("unknown variable `{name}`"));
            return self.poison(span);
        }
        // A qualified path with no arguments: a nullary constructor.
        self.infer_ctor(path, &[], &[], span)
    }

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
        self.mk_expr(ExprKind::Seq(out), ty, span)
    }

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
        let bound = self.sym(name.value).to_owned();
        let var = self.vars.fresh(&bound, var_ty, name_span);
        let unit = self.tcx.mk_unit();
        self.mk_expr(
            ExprKind::Let {
                var,
                name: bound,
                span: name_span,
                value: Box::new(value),
            },
            unit,
            span,
        )
    }

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

    fn infer_cast(
        &mut self,
        ty: &surface::Type,
        e: &surface::Expr,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let e = self.infer_expr(e);
        let target = self.eval_type(ty);
        self.mk_expr(ExprKind::Cast(Box::new(e), target), target, span)
    }

    fn infer_func_call(&mut self, fc: &surface::FuncCall, span: Option<Span>) -> Expr<'tcx> {
        let fname = self.sym(fc.name.basename);
        let Some(proto) = self.functions.get(fname).cloned() else {
            self.error(span, format!("unknown function `{fname}`"));
            return self.poison(span);
        };
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
                target: proto.name,
                ty_args,
                args,
                regional: proto.is_regional,
            },
            result,
            span,
        )
    }

    fn infer_closure_call(
        &mut self,
        callee: &surface::Expr,
        args: &[surface::Expr],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let callee = self.infer_expr(callee);
        let (params, ret) = match callee.ty.kind() {
            TyKind::Closure { params, ret } => (params.to_vec(), *ret),
            _ => {
                self.error(span, "called value is not a closure");
                return self.poison(span);
            }
        };
        let mut out = Vec::new();
        for (arg, pty) in args.iter().zip(&params) {
            out.push(self.check_expr(arg, *pty));
        }
        self.mk_expr(
            ExprKind::ClosureCall {
                target: Box::new(callee),
                args: out,
            },
            ret,
            span,
        )
    }

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

    fn infer_lambda(&mut self, lam: &surface::Lambda, span: Option<Span>) -> Expr<'tcx> {
        let mark = self.vars.mark();
        let mut params = Vec::new();
        for (name, ty) in &lam.args {
            let pty = match ty {
                Some(t) => self.eval_type(t),
                None => self.infer.new_hole_ty(),
            };
            let name = self.sym(*name);
            let var = self.vars.fresh(name, pty, None);
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
            path,
            args,
            flex: crate::semi::ty::Capability::Flex,
        } = dty.kind()
        else {
            self.error(span, "assignment target must be a flex record");
            return self.poison(span);
        };
        let Some((idx, field_ty, mutable)) = self.resolve_field(path, args, acc) else {
            self.error(span, "no such field on assignment target");
            return self.poison(span);
        };
        if !mutable {
            self.error(span, "cannot assign to an immutable field");
        }
        // A mutable field's type is already the nullable link type.
        let src = self.check_expr(src, field_ty);
        let unit = self.tcx.mk_unit();
        self.mk_expr(
            ExprKind::Assign(Box::new(dst), idx, Box::new(src)),
            unit,
            span,
        )
    }

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
            let TyKind::Record { path, args, .. } = cur.kind() else {
                self.error(span, format!("cannot access a field of `{cur:?}`"));
                return self.poison(span);
            };
            let Some((idx, field_ty, _)) = self.resolve_field(path, args, acc) else {
                self.error(span, "no such field");
                return self.poison(span);
            };
            indices.push(idx);
            cur = self.infer.shallow_resolve(field_ty);
        }
        self.mk_expr(ExprKind::Proj(Box::new(base), indices), cur, span)
    }

    /// Resolve an access on a record type to `(field index, field type, mutable)`.
    /// The field type is instantiated with the record's type arguments.
    fn resolve_field(
        &mut self,
        record_path: &str,
        args: &[Ty<'tcx>],
        acc: &surface::Access,
    ) -> Option<(u32, Ty<'tcx>, bool)> {
        let record = self.records.get(record_path)?;
        let ty_params = record.ty_params.clone();
        let fields = record.fields.clone()?;
        let inst =
            Instantiation::from_pairs(ty_params.iter().map(|(_, g)| *g).zip(args.iter().copied()));
        let (idx, decl_ty, mutable) = match (&fields, acc) {
            (RecordFields::Named(fs), surface::Access::Named(name)) => {
                let name = self.sym(*name);
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
        let field_ty = self.infer.instantiate_ty(decl_ty, &inst);
        Some((idx as u32, field_ty, mutable))
    }

    // ----- constructors -----

    fn infer_ctor_call(&mut self, cc: &surface::CtorCall, span: Option<Span>) -> Expr<'tcx> {
        self.infer_ctor(&cc.name, &cc.ty_args, &cc.args, span)
    }

    fn infer_ctor(
        &mut self,
        path: &surface::Path,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<reussir_syntax::kind::TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let qualifier = path.segments.last().map(|s| self.sym(*s));
        let basename = self.sym(path.basename);
        // The built-in nullable constructors.
        if qualifier == Some("Nullable") {
            return self.infer_nullable(basename, args, span);
        }
        // An enum variant: `Enum::Variant`.
        if let Some(enum_name) = qualifier {
            return self.infer_variant(enum_name, basename, ty_args, args, span);
        }
        // A struct constructor.
        self.infer_struct(basename, ty_args, args, span)
    }

    fn infer_nullable(
        &mut self,
        variant: &str,
        args: &[(Option<reussir_syntax::kind::TokenKey>, surface::Expr)],
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

    fn infer_struct(
        &mut self,
        name: &str,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<reussir_syntax::kind::TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let Some(record) = self.records.get(name).cloned() else {
            self.error(span, format!("unknown type `{name}`"));
            return self.poison(span);
        };
        if matches!(record.fields, Some(RecordFields::Variants(_))) {
            self.error(span, format!("`{name}` is an enum, not a struct"));
            return self.poison(span);
        }
        let inst = self.instantiate(&record.generics_as_decls(), ty_args, span);
        let field_tys = self.field_types(&record.fields, &inst);

        let checked = self.check_ctor_args(name, args, &field_tys, span);
        let ty_args = self.inst_args(&record.generics_as_decls(), &inst);
        let result = self.record_ty(name, &record, &inst);
        self.mk_expr(
            ExprKind::CompoundCall {
                target: name.to_owned(),
                ty_args,
                args: checked,
            },
            result,
            span,
        )
    }

    fn infer_variant(
        &mut self,
        enum_name: &str,
        variant: &str,
        ty_args: &[Option<surface::Type>],
        args: &[(Option<reussir_syntax::kind::TokenKey>, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let Some(record) = self.records.get(enum_name).cloned() else {
            self.error(span, format!("unknown enum `{enum_name}`"));
            return self.poison(span);
        };
        let Some(RecordFields::Variants(variants)) = &record.fields else {
            self.error(span, format!("`{enum_name}` is not an enum"));
            return self.poison(span);
        };
        let Some(vidx) = variants.iter().position(|v| v.name == variant) else {
            self.error(span, format!("`{enum_name}` has no variant `{variant}`"));
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
        let result = self.record_ty(enum_name, &record, &inst);
        self.mk_expr(
            ExprKind::VariantCall {
                target: enum_name.to_owned(),
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
        name: &str,
        args: &[(Option<reussir_syntax::kind::TokenKey>, surface::Expr)],
        field_tys: &[(Option<String>, Ty<'tcx>)],
        span: Option<Span>,
    ) -> Vec<Expr<'tcx>> {
        if args.len() != field_tys.len() {
            self.error(
                span,
                format!(
                    "`{name}` expects {} field(s), got {}",
                    field_tys.len(),
                    args.len()
                ),
            );
        }
        // Named arguments are matched by field name; positional by order.
        let named = args.iter().any(|(f, _)| f.is_some());
        let arg_fields: Vec<Option<&str>> =
            args.iter().map(|(f, _)| f.map(|k| self.sym(k))).collect();
        let mut out = Vec::new();
        if named {
            for (fname, fty) in field_tys {
                let Some(fname) = fname else { continue };
                match arg_fields.iter().position(|af| *af == Some(fname.as_str())) {
                    Some(i) => out.push(self.check_expr(&args[i].1, *fty)),
                    None => {
                        self.error(span, format!("missing field `{fname}`"));
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
    ) -> Vec<(Option<String>, Ty<'tcx>)> {
        match fields {
            Some(RecordFields::Named(fs)) => fs
                .iter()
                .map(|(n, t, _)| (Some(n.clone()), self.infer.instantiate_ty(*t, inst)))
                .collect(),
            Some(RecordFields::Unnamed(fs)) => fs
                .iter()
                .map(|(t, _)| (None, self.infer.instantiate_ty(*t, inst)))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn record_ty(
        &mut self,
        name: &str,
        record: &super::ctxt::Record<'tcx>,
        inst: &Instantiation<'tcx>,
    ) -> Ty<'tcx> {
        let args: Vec<Ty<'tcx>> = record
            .ty_params
            .iter()
            .map(|(_, g)| inst.get(*g).unwrap_or_else(|| self.tcx.mk_generic(*g)))
            .collect();
        let flex = match record.default_cap {
            super::ctxt::DefaultCap::Regional => crate::semi::ty::Capability::Regional,
            _ => crate::semi::ty::Capability::Irrelevant,
        };
        self.tcx.mk_record(name, &args, flex)
    }

    // ----- instantiation helpers -----

    /// Build an instantiation for a list of generics, using explicit type
    /// arguments where given and fresh holes otherwise, registering each
    /// generic's bounds as obligations on the chosen type.
    pub(super) fn instantiate(
        &mut self,
        generics: &[(String, GenericId)],
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
        generics: &[(String, GenericId)],
        inst: &Instantiation<'tcx>,
    ) -> Vec<Ty<'tcx>> {
        generics
            .iter()
            .map(|(_, g)| inst.get(*g).unwrap_or_else(|| self.tcx.mk_generic(*g)))
            .collect()
    }

    // ----- zonking -----

    /// Resolve every type in an expression tree against the solved holes.
    pub(super) fn zonk_expr(&mut self, mut e: Expr<'tcx>) -> Expr<'tcx> {
        e.ty = self.infer.resolve(e.ty);
        e.kind = self.zonk_kind(e.kind);
        e
    }

    fn zonk_kind(&mut self, kind: ExprKind<'tcx>) -> ExprKind<'tcx> {
        use ExprKind::*;
        let zb = |s: &mut Self, e: Box<Expr<'tcx>>| Box::new(s.zonk_expr(*e));
        match kind {
            Negate(e) => Negate(zb(self, e)),
            Not(e) => Not(zb(self, e)),
            Arith(l, op, r) => Arith(zb(self, l), op, zb(self, r)),
            Cmp(l, op, r) => Cmp(zb(self, l), op, zb(self, r)),
            Cast(e, t) => {
                let t = self.infer.resolve(t);
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
                ty_args: ty_args.into_iter().map(|t| self.infer.resolve(t)).collect(),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
                regional,
            },
            CompoundCall {
                target,
                ty_args,
                args,
            } => CompoundCall {
                target,
                ty_args: ty_args.into_iter().map(|t| self.infer.resolve(t)).collect(),
                args: args.into_iter().map(|e| self.zonk_expr(e)).collect(),
            },
            VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => VariantCall {
                target,
                ty_args: ty_args.into_iter().map(|t| self.infer.resolve(t)).collect(),
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
                    .map(|(v, t)| (v, self.infer.resolve(t)))
                    .collect(),
                params: c
                    .params
                    .into_iter()
                    .map(|(v, t)| (v, self.infer.resolve(t)))
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
        GlobalStr(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Poison => {}
    }
}

impl<'tcx> super::ctxt::Record<'tcx> {
    /// The record's generics as `(name, id)` declaration pairs.
    fn generics_as_decls(&self) -> Vec<(String, GenericId)> {
        self.ty_params.clone()
    }
}
