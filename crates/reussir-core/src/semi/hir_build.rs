//! Re-intern pass: rebuild the (owned) HIR from the `hir_raw` AST the grammar
//! produces. The MIR twin is [`crate::full::ir_build`]; the HIR differs in that
//! expressions are owned (`Box`/`Vec`, no arena), calls resolve a `#path` to a
//! fresh [`DefId`], and functions/types carry generics. Text-faithful, as on the
//! MIR side: `print(parse(t)) == t`.

use reussir_syntax::kind::{InternKey, Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::full::ir_lex::lex;
use crate::semi::hir::{
    ArithOp, ClosureExpr, CmpOp, DecisionTree, Expr, ExprId, ExprKind, Function, PatVarRef,
    SwitchCases, VarId,
};
use crate::semi::hir_ir;
use crate::semi::hir_raw as raw;
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, DefId, FpTy, GenericId, HoleId, IntTy, Ty, TyCtxt, TyKind};

/// A parsed HIR program plus the fresh tables needed to re-print it.
pub struct Parsed<'tcx> {
    pub funcs: Vec<Function<'tcx>>,
    pub defs: DefTable,
    pub names: Names,
}

/// A fresh `TokenKey` interner for source names and `#path` segments.
#[derive(Default)]
pub struct Names {
    strings: Vec<String>,
    map: FxHashMap<String, TokenKey>,
}

impl Names {
    fn intern(&mut self, s: &str) -> TokenKey {
        if let Some(&k) = self.map.get(s) {
            return k;
        }
        let key = TokenKey::try_from_u32(self.strings.len() as u32).expect("token-key space");
        self.strings.push(s.to_owned());
        self.map.insert(s.to_owned(), key);
        key
    }
}

impl Resolver<TokenKey> for Names {
    fn try_resolve(&self, key: TokenKey) -> Option<&str> {
        self.strings
            .get(key.into_u32() as usize)
            .map(String::as_str)
    }
}

/// Parse `text` into a fresh HIR program (interning types into `tcx`).
pub fn parse_program<'tcx>(tcx: &TyCtxt<'tcx>, text: &str) -> Result<Parsed<'tcx>, String> {
    let raw = hir_ir::ProgramParser::new()
        .parse(lex(text))
        .map_err(|e| format!("{e:?}"))?;
    let mut b = Builder {
        tcx,
        names: Names::default(),
        defs: DefTable::new(),
    };
    let funcs = raw.funcs.iter().map(|f| b.func(f)).collect();
    Ok(Parsed {
        funcs,
        defs: b.defs,
        names: b.names,
    })
}

struct Builder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    names: Names,
    defs: DefTable,
}

impl<'tcx> Builder<'_, 'tcx> {
    /// Resolve-or-declare a function path to a stable `DefId` (shared by the
    /// definition and its call sites, in any order).
    fn function_def(&mut self, path: &str) -> DefId {
        let key = self.names.intern(path);
        self.defs
            .resolve_function(key)
            .unwrap_or_else(|| self.defs.declare_function(key).expect("fresh fn decl"))
    }

    fn record_def(&mut self, path: &str) -> DefId {
        let key = self.names.intern(path);
        self.defs
            .resolve_record(key)
            .unwrap_or_else(|| self.defs.declare_record(key).expect("fresh record decl"))
    }

    fn placeholder(&self) -> Ty<'tcx> {
        self.tcx.mk_unit()
    }

    fn func(&mut self, f: &raw::Func) -> Function<'tcx> {
        let def = self.function_def(&f.path);
        let name = self.names.intern(&f.path);
        let generics: Vec<(TokenKey, GenericId)> = f
            .generics
            .iter()
            .map(|g| {
                let gname = self.names.intern(&format!("${}", g.id));
                (gname, GenericId(g.id))
            })
            .collect();
        let regional_generics: Vec<GenericId> = f
            .generics
            .iter()
            .filter(|g| g.regional)
            .map(|g| GenericId(g.id))
            .collect();
        let params: Vec<(TokenKey, VarId, Ty<'tcx>)> = f
            .params
            .iter()
            .map(|p| (self.names.intern(&p.name), VarId(p.var), self.ty(&p.ty)))
            .collect();
        let return_ty = self.ty(&f.ret);
        let body = f.body.as_ref().map(|b| self.expr(b));
        Function {
            def,
            name,
            visibility: if f.is_pub {
                crate::surface::Visibility::Public
            } else {
                crate::surface::Visibility::Private
            },
            generics,
            regional_generics,
            params,
            return_ty,
            is_regional: f.regional,
            body,
            span: None,
        }
    }

    fn ty(&mut self, t: &raw::Ty) -> Ty<'tcx> {
        match t {
            raw::Ty::Signed(w) => self.tcx.mk_int(IntTy::Signed(*w)),
            raw::Ty::Unsigned(w) => self.tcx.mk_int(IntTy::Unsigned(*w)),
            raw::Ty::Ieee(w) => self.tcx.mk_fp(FpTy::Ieee(*w)),
            raw::Ty::BFloat16 => self.tcx.mk_fp(FpTy::BFloat16),
            raw::Ty::Float8 => self.tcx.mk_fp(FpTy::Float8),
            raw::Ty::Bool => self.tcx.mk_bool(),
            raw::Ty::Str => self.tcx.mk_str(),
            raw::Ty::Unit => self.tcx.mk_unit(),
            raw::Ty::Bottom => self.tcx.mk(TyKind::Bottom),
            raw::Ty::Generic(g) => self.tcx.mk_generic(GenericId(*g)),
            raw::Ty::Hole(h) => self.tcx.mk_hole(HoleId(*h)),
            raw::Ty::Nullable(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_nullable(inner)
            }
            raw::Ty::Record { cap, path, args } => {
                let def = self.record_def(path);
                let args: Vec<Ty<'tcx>> = args.iter().map(|a| self.ty(a)).collect();
                self.tcx.mk_record(def, &args, capability(*cap))
            }
            raw::Ty::Closure { params, ret } => {
                let params: Vec<Ty<'tcx>> = params.iter().map(|p| self.ty(p)).collect();
                let ret = self.ty(ret);
                self.tcx.mk_closure(&params, ret)
            }
        }
    }

    fn tys(&mut self, ts: &[raw::Ty]) -> Vec<Ty<'tcx>> {
        ts.iter().map(|t| self.ty(t)).collect()
    }

    fn boxed(&mut self, e: &raw::Expr) -> Box<Expr<'tcx>> {
        Box::new(self.expr(e))
    }

    fn exprs(&mut self, es: &[raw::Expr]) -> Vec<Expr<'tcx>> {
        es.iter().map(|e| self.expr(e)).collect()
    }

    fn expr(&mut self, e: &raw::Expr) -> Expr<'tcx> {
        let (kind, ty): (ExprKind<'tcx>, Ty<'tcx>) = match e {
            raw::Expr::ConstInt(n, t) => (ExprKind::ConstInt(*n), self.ty(t)),
            raw::Expr::ConstFloat(f, t) => (ExprKind::ConstFloat(*f), self.ty(t)),
            raw::Expr::ConstBool(b) => (ExprKind::ConstBool(*b), self.tcx.mk_bool()),
            raw::Expr::GlobalStr(_) => {
                unimplemented!("string literals are not yet round-tripped")
            }
            raw::Expr::Var(v) => (ExprKind::Var(VarId(*v)), self.placeholder()),
            raw::Expr::Poison => (ExprKind::Poison, self.placeholder()),
            raw::Expr::Negate(x) => (ExprKind::Negate(self.boxed(x)), self.placeholder()),
            raw::Expr::Not(x) => (ExprKind::Not(self.boxed(x)), self.placeholder()),
            raw::Expr::Arith(l, op, r, t) => {
                let kind = ExprKind::Arith(self.boxed(l), arith(*op), self.boxed(r));
                (kind, self.ty(t))
            }
            raw::Expr::Cmp(l, op, r, t) => {
                let kind = ExprKind::Cmp(self.boxed(l), cmp(*op), self.boxed(r));
                (kind, self.ty(t))
            }
            raw::Expr::Cast(x, t) => {
                let ty = self.ty(t);
                (ExprKind::Cast(self.boxed(x), ty), ty)
            }
            raw::Expr::If(c, t, f) => (
                ExprKind::If(self.boxed(c), self.boxed(t), self.boxed(f)),
                self.placeholder(),
            ),
            raw::Expr::RegionRun(x) => (ExprKind::RegionRun(self.boxed(x)), self.placeholder()),
            raw::Expr::Proj(base, path) => (
                ExprKind::Proj(self.boxed(base), path.clone()),
                self.placeholder(),
            ),
            raw::Expr::Assign(dst, field, src) => (
                ExprKind::Assign(self.boxed(dst), *field, self.boxed(src)),
                self.tcx.mk_unit(),
            ),
            raw::Expr::Let {
                var, name, value, ..
            } => {
                let name = self.names.intern(name);
                (
                    ExprKind::Let {
                        var: VarId(*var),
                        name,
                        span: None,
                        value: self.boxed(value),
                    },
                    self.tcx.mk_unit(),
                )
            }
            raw::Expr::Seq(items) => (ExprKind::Seq(self.exprs(items)), self.placeholder()),
            raw::Expr::FuncCall {
                regional,
                path,
                ty_args,
                args,
                ty,
            } => {
                let target = self.function_def(path);
                let ty_args = self.tys(ty_args);
                let args = self.exprs(args);
                (
                    ExprKind::FuncCall {
                        target,
                        ty_args,
                        args,
                        regional: *regional,
                    },
                    self.ty(ty),
                )
            }
            raw::Expr::CompoundCall {
                path,
                ty_args,
                args,
                ty,
            } => {
                let target = self.record_def(path);
                let ty_args = self.tys(ty_args);
                let args = self.exprs(args);
                (
                    ExprKind::CompoundCall {
                        target,
                        ty_args,
                        args,
                    },
                    self.ty(ty),
                )
            }
            raw::Expr::VariantCall {
                path,
                ty_args,
                variant,
                args,
                ty,
            } => {
                let target = self.record_def(path);
                let ty_args = self.tys(ty_args);
                let args = self.exprs(args);
                (
                    ExprKind::VariantCall {
                        target,
                        ty_args,
                        variant: *variant,
                        args,
                    },
                    self.ty(ty),
                )
            }
            raw::Expr::NullableCall(inner) => (
                ExprKind::NullableCall(inner.as_ref().map(|x| self.boxed(x))),
                self.placeholder(),
            ),
            raw::Expr::ClosureCall { target, args } => {
                let target = self.boxed(target);
                let args = self.exprs(args);
                (ExprKind::ClosureCall { target, args }, self.placeholder())
            }
            raw::Expr::Closure {
                captures,
                params,
                body,
            } => {
                let captures = captures
                    .iter()
                    .map(|v| (VarId(*v), self.placeholder()))
                    .collect();
                let params = params
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let body = self.boxed(body);
                (
                    ExprKind::Closure(ClosureExpr {
                        captures,
                        params,
                        body,
                    }),
                    self.placeholder(),
                )
            }
            raw::Expr::Match(scrut, tree) => {
                let scrut = self.boxed(scrut);
                (ExprKind::Match(scrut, self.tree(tree)), self.placeholder())
            }
        };
        Expr {
            kind,
            ty,
            span: None,
            id: ExprId(0),
        }
    }

    // ----- decision trees -----

    fn tree(&mut self, t: &raw::Tree) -> DecisionTree<'tcx> {
        match t {
            raw::Tree::Uncovered => DecisionTree::Uncovered,
            raw::Tree::Unreachable => DecisionTree::Unreachable,
            raw::Tree::Leaf { bindings, body } => DecisionTree::Leaf {
                body: self.boxed(body),
                bindings: self.bindings(bindings),
            },
            raw::Tree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => DecisionTree::Guard {
                bindings: self.bindings(bindings),
                guard: self.boxed(guard),
                success: Box::new(self.tree(success)),
                failure: Box::new(self.tree(failure)),
            },
            raw::Tree::Switch { scrutinee, cases } => DecisionTree::Switch {
                scrutinee: PatVarRef(scrutinee.clone()),
                cases: self.cases(cases),
            },
        }
    }

    fn bindings(&mut self, bs: &[raw::Binding]) -> Vec<(VarId, PatVarRef)> {
        bs.iter()
            .map(|(v, path)| (VarId(*v), PatVarRef(path.clone())))
            .collect()
    }

    fn cases(&mut self, c: &raw::Cases) -> SwitchCases<'tcx> {
        match c {
            raw::Cases::Int { cases, default } => {
                let cases = cases.iter().map(|(n, t)| (*n, self.tree(t))).collect();
                SwitchCases::Int {
                    cases,
                    default: Box::new(self.tree(default)),
                }
            }
            raw::Cases::Bool { if_true, if_false } => SwitchCases::Bool {
                if_true: Box::new(self.tree(if_true)),
                if_false: Box::new(self.tree(if_false)),
            },
            raw::Cases::Ctor(arms) => {
                SwitchCases::Ctor(arms.iter().map(|t| self.tree(t)).collect())
            }
            raw::Cases::Str { .. } => {
                unimplemented!("string switches are not yet round-tripped")
            }
            raw::Cases::Nullable { non_null, null } => SwitchCases::Nullable {
                non_null: Box::new(self.tree(non_null)),
                null: Box::new(self.tree(null)),
            },
        }
    }
}

fn capability(c: raw::Cap) -> Capability {
    match c {
        raw::Cap::None => Capability::Irrelevant,
        raw::Cap::Flex => Capability::Flex,
        raw::Cap::Rigid => Capability::Rigid,
        raw::Cap::Regional => Capability::Regional,
    }
}

fn arith(op: raw::ArithOp) -> ArithOp {
    match op {
        raw::ArithOp::Add => ArithOp::Add,
        raw::ArithOp::Sub => ArithOp::Sub,
        raw::ArithOp::Mul => ArithOp::Mul,
        raw::ArithOp::Div => ArithOp::Div,
        raw::ArithOp::Mod => ArithOp::Mod,
        raw::ArithOp::And => ArithOp::And,
        raw::ArithOp::Or => ArithOp::Or,
    }
}

fn cmp(op: raw::CmpOp) -> CmpOp {
    match op {
        raw::CmpOp::Lt => CmpOp::Lt,
        raw::CmpOp::Gt => CmpOp::Gt,
        raw::CmpOp::Le => CmpOp::Le,
        raw::CmpOp::Ge => CmpOp::Ge,
        raw::CmpOp::Eq => CmpOp::Eq,
        raw::CmpOp::Ne => CmpOp::Ne,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::semi::elaborate;
    use crate::semi::hir_print::Printer;
    use crate::{surface, with_tcx};

    /// Print the elaborated HIR, parse it back, re-print, and assert text
    /// equality.
    fn roundtrip(source: &str) {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let text = Printer::new(&elab.defs, elab.resolver).program(&elab.elaborated);
            let parsed = parse_program(tcx, &text).expect("re-parse");
            let text2 = Printer::new(&parsed.defs, &parsed.names).program(&parsed.funcs);
            assert_eq!(
                text, text2,
                "round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    #[test]
    fn roundtrips_a_polymorphic_function() {
        roundtrip("fn id<T>(x: T) -> T { x } pub fn use_it(n: i32) -> i32 { id(n) }");
    }

    #[test]
    fn roundtrips_scalar_control_flow() {
        roundtrip(
            "pub fn fib(n: u64) -> u64 { \
             let m = n + 1; \
             if n <= 1 { m } else { fib(n - 1) + fib(n - 2) } }",
        );
    }

    #[test]
    fn roundtrips_a_match() {
        roundtrip(
            "enum Opt { None, Some(i32) } \
             pub fn unwrap(o: Opt) -> i32 { match o { Opt::None => 0, Opt::Some(x) => x } }",
        );
    }
}
