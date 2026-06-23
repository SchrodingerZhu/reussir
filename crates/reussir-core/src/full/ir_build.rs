//! Re-intern pass: rebuild the arena-allocated MIR from the owned [`ir_raw`] AST
//! the grammar produces. Identifiers/symbols are interned into *fresh* tables
//! (the original session ids are not recoverable from text), so the round trip
//! is text-faithful (`print(parse(t)) == t`) rather than value-identical, and
//! the result re-prints through the [`crate::full::print`] serializer.

use lasso::Rodeo;
use reussir_syntax::kind::{InternKey, Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::full::ir_lex::lex;
use crate::full::{ir, ir_raw as raw, mir};
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyCtxt, TyKind};

/// A parsed program plus the fresh tables needed to re-print it.
pub struct Parsed<'tcx> {
    pub program: mir::Program<'tcx>,
    pub defs: DefTable,
    pub names: Names,
}

/// A minimal [`Resolver`]: a fresh `TokenKey` interner for source names and
/// record-path segments encountered while parsing.
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

/// Parse `text` into a fresh MIR program (interning into `tcx`).
pub fn parse_program<'tcx>(tcx: &TyCtxt<'tcx>, text: &str) -> Result<Parsed<'tcx>, String> {
    let raw = ir::ProgramParser::new()
        .parse(lex(text))
        .map_err(|e| format!("{e:?}"))?;
    let mut b = Builder {
        tcx,
        symbols: Rodeo::default(),
        names: Names::default(),
        defs: DefTable::new(),
    };
    let program = b.program(raw);
    Ok(Parsed {
        program,
        defs: b.defs,
        names: b.names,
    })
}

struct Builder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    symbols: Rodeo,
    names: Names,
    defs: DefTable,
}

impl<'tcx> Builder<'_, 'tcx> {
    fn sym(&mut self, s: &str) -> mir::Symbol {
        mir::Symbol(self.symbols.get_or_intern(s))
    }

    /// The `DefId` for a record path, declaring it on first sight.
    fn record_def(&mut self, path: &str) -> crate::semi::ty::DefId {
        let key = self.names.intern(path);
        match self.defs.resolve_record(key) {
            Some(d) => d,
            None => self.defs.declare_record(key).expect("fresh record decl"),
        }
    }

    /// A placeholder for a node whose type the printer never emits (vars,
    /// blocks); sound for the text-faithful round trip.
    fn placeholder(&self) -> Ty<'tcx> {
        self.tcx.mk_unit()
    }

    fn program(&mut self, raw: raw::Program) -> mir::Program<'tcx> {
        let records: Vec<mir::RecordInstance<'tcx>> = raw
            .records
            .iter()
            .map(|s| mir::RecordInstance {
                symbol: self.sym(s),
                ty: self.placeholder(),
            })
            .collect();
        let trampolines: Vec<mir::Trampoline> = raw
            .trampolines
            .iter()
            .map(|t| mir::Trampoline {
                export: self.sym(&t.export),
                abi: t.abi.clone(),
                target: self.sym(&t.target),
            })
            .collect();
        let functions: Vec<mir::Function<'tcx>> = raw.funcs.iter().map(|f| self.func(f)).collect();

        // Move the freshly-built symbol interner into the program.
        let symbols = std::mem::take(&mut self.symbols);
        mir::Program {
            functions,
            records,
            trampolines,
            symbols,
        }
    }

    fn func(&mut self, f: &raw::Func) -> mir::Function<'tcx> {
        let symbol = self.sym(&f.symbol);
        let params = f
            .params
            .iter()
            .map(|p| {
                let name = self.names.intern(&p.name);
                mir::Param {
                    name,
                    var: VarId(p.var),
                    ty: self.ty(&p.ty),
                }
            })
            .collect();
        let return_ty = self.ty(&f.ret);
        let body = f.body.as_ref().map(|b| self.expr_ref(b));
        mir::Function {
            symbol,
            visibility: if f.is_pub {
                crate::surface::Visibility::Public
            } else {
                crate::surface::Visibility::Private
            },
            is_regional: f.regional,
            params,
            return_ty,
            body,
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
            raw::Ty::Nullable(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_nullable(inner)
            }
            raw::Ty::Record { cap: c, path, args } => {
                let def = self.record_def(path);
                let args: Vec<Ty<'tcx>> = args.iter().map(|a| self.ty(a)).collect();
                self.tcx.mk_record(def, &args, cap(*c))
            }
            raw::Ty::Closure { params, ret } => {
                let params: Vec<Ty<'tcx>> = params.iter().map(|p| self.ty(p)).collect();
                let ret = self.ty(ret);
                self.tcx.mk_closure(&params, ret)
            }
        }
    }

    fn expr_ref(&mut self, e: &raw::Expr) -> &'tcx mir::Expr<'tcx> {
        let lowered = self.expr(e);
        self.tcx.alloc(lowered)
    }

    fn expr_slice(&mut self, es: &[raw::Expr]) -> &'tcx [mir::Expr<'tcx>] {
        let v: Vec<mir::Expr<'tcx>> = es.iter().map(|e| self.expr(e)).collect();
        self.tcx.alloc_slice(&v)
    }

    fn expr(&mut self, e: &raw::Expr) -> mir::Expr<'tcx> {
        use mir::ExprKind as M;
        let (kind, ty): (M<'tcx>, Ty<'tcx>) = match e {
            raw::Expr::ConstInt(n, t) => (M::ConstInt(*n), self.ty(t)),
            raw::Expr::ConstFloat(f, t) => (M::ConstFloat(*f), self.ty(t)),
            raw::Expr::ConstBool(b) => (M::ConstBool(*b), self.tcx.mk_bool()),
            raw::Expr::GlobalStr(_) => {
                // String-table refs print only their tag; reconstructing the
                // `StringToken` needs the table, so defer until that lands.
                unimplemented!("string literals are not yet round-tripped")
            }
            raw::Expr::Var(v) => (M::Var(VarId(*v)), self.placeholder()),
            raw::Expr::Poison => (M::Poison, self.placeholder()),
            raw::Expr::Negate(x) => (M::Negate(self.expr_ref(x)), self.placeholder()),
            raw::Expr::Not(x) => (M::Not(self.expr_ref(x)), self.placeholder()),
            raw::Expr::Arith(l, op, r, t) => {
                let kind = M::Arith(self.expr_ref(l), arith(*op), self.expr_ref(r));
                (kind, self.ty(t))
            }
            raw::Expr::Cmp(l, op, r, t) => {
                let kind = M::Cmp(self.expr_ref(l), cmp(*op), self.expr_ref(r));
                (kind, self.ty(t))
            }
            raw::Expr::Cast(x, t) => {
                let ty = self.ty(t);
                (M::Cast(self.expr_ref(x), ty), ty)
            }
            raw::Expr::If(c, t, f) => {
                let kind = M::If(self.expr_ref(c), self.expr_ref(t), self.expr_ref(f));
                (kind, self.placeholder())
            }
            raw::Expr::RegionRun(x) => (M::RegionRun(self.expr_ref(x)), self.placeholder()),
            raw::Expr::Proj(base, path) => {
                let base = self.expr_ref(base);
                let path = self.tcx.alloc_slice(path);
                (M::Proj(base, path), self.placeholder())
            }
            raw::Expr::Assign(dst, field, src) => {
                let kind = M::Assign(self.expr_ref(dst), *field, self.expr_ref(src));
                (kind, self.tcx.mk_unit())
            }
            raw::Expr::Let {
                var,
                name,
                ty,
                value,
            } => {
                let name = self.names.intern(name);
                let _ = self.ty(ty); // the printer reads the value's type, set below
                let value = self.expr_ref(value);
                (
                    M::Let {
                        var: VarId(*var),
                        name,
                        value,
                    },
                    self.tcx.mk_unit(),
                )
            }
            raw::Expr::Seq(items) => (M::Seq(self.expr_slice(items)), self.placeholder()),
            raw::Expr::Call {
                regional,
                symbol,
                args,
                ty,
            } => {
                let callee = self.sym(symbol);
                let args = self.expr_slice(args);
                (
                    M::Call {
                        callee,
                        args,
                        regional: *regional,
                    },
                    self.ty(ty),
                )
            }
            raw::Expr::Ctor { symbol, args } => {
                let record = self.sym(symbol);
                (
                    M::Ctor {
                        record,
                        args: self.expr_slice(args),
                    },
                    self.placeholder(),
                )
            }
            raw::Expr::Variant {
                symbol,
                variant,
                args,
            } => {
                let record = self.sym(symbol);
                (
                    M::Variant {
                        record,
                        variant: *variant,
                        args: self.expr_slice(args),
                    },
                    self.placeholder(),
                )
            }
            raw::Expr::NullableCall(inner) => (
                M::NullableCall(inner.as_ref().map(|x| self.expr_ref(x))),
                self.placeholder(),
            ),
            raw::Expr::ClosureCall { target, args } => {
                let target = self.expr_ref(target);
                (
                    M::ClosureCall {
                        target,
                        args: self.expr_slice(args),
                    },
                    self.placeholder(),
                )
            }
            raw::Expr::Closure {
                captures,
                params,
                body,
            } => {
                let caps: Vec<(VarId, Ty<'tcx>)> = captures
                    .iter()
                    .map(|v| (VarId(*v), self.placeholder()))
                    .collect();
                let captures = self.tcx.alloc_slice(&caps);
                let ps: Vec<(VarId, Ty<'tcx>)> = params
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let mir_params = self.tcx.alloc_slice(&ps);
                let body = self.expr_ref(body);
                (
                    M::Closure(mir::ClosureExpr {
                        captures,
                        params: mir_params,
                        body,
                    }),
                    self.placeholder(),
                )
            }
            raw::Expr::Match(..) => {
                unreachable!("decision trees are not yet parsed")
            }
        };
        mir::Expr {
            kind,
            ty,
            span: None,
        }
    }
}

fn cap(c: raw::Cap) -> Capability {
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
    use crate::full::mono::monomorphize;
    use crate::full::print::Printer;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// Print a program, parse it back, re-print, and assert text equality
    /// (the text-faithful round-trip contract).
    fn roundtrip(source: &str) {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab);
            assert!(reports.is_empty(), "mono reports: {reports:#?}");

            let text = Printer::new(&elab.defs, elab.resolver).program(&full);
            let parsed = parse_program(tcx, &text).expect("re-parse");
            let text2 = Printer::new(&parsed.defs, &parsed.names).program(&parsed.program);
            assert_eq!(
                text, text2,
                "round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    #[test]
    fn roundtrips_a_scalar_function() {
        roundtrip(
            "pub fn fib(n: u64) -> u64 { \
             let m = n + 1; \
             if n <= 1 { m } else { fib(n - 1) + fib(n - 2) } }",
        );
    }
}
