//! Re-intern pass: rebuild the arena-allocated MIR from the owned [`ir_raw`] AST
//! the grammar produces.
//!
//! The rebuilt MIR is **value-sound modulo arena re-interning**: every node's
//! type is read from the printed `: ty` annotation (never invented — there is no
//! placeholder), so the result equals the original up to (a) fresh type-arena
//! pointers, which interning makes value-equal anyway, and (b) a consistent
//! relabeling of the session-local symbol/`DefId` tables. The textual form is
//! keyed by stable `@symbol` strings, not those ids, so it stays portable across
//! sessions and (eventually) files. `print(parse(t)) == t` therefore witnesses
//! type soundness, since every type is in the text.

use lasso::Rodeo;
use reussir_syntax::kind::{InternKey, Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::ir_lex::lex;
use crate::full::mir::{self, grammar as ir, raw};
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyCtxt, TyKind};
use crate::utils::string::StringToken;

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

    fn program(&mut self, raw: raw::Program) -> mir::Program<'tcx> {
        let records: Vec<mir::RecordInstance<'tcx>> = raw
            .records
            .iter()
            .map(|(s, t)| mir::RecordInstance {
                symbol: self.sym(s),
                ty: self.ty(t),
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

    // ----- decision trees -----

    fn tree_ref(&mut self, t: &raw::Tree) -> &'tcx mir::DecisionTree<'tcx> {
        let lowered = self.tree(t);
        self.tcx.alloc(lowered)
    }

    fn tree(&mut self, t: &raw::Tree) -> mir::DecisionTree<'tcx> {
        use mir::DecisionTree as D;
        match t {
            raw::Tree::Uncovered => D::Uncovered,
            raw::Tree::Unreachable => D::Unreachable,
            raw::Tree::Leaf { bindings, body } => D::Leaf {
                body: self.expr_ref(body),
                bindings: self.bindings(bindings),
            },
            raw::Tree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => D::Guard {
                bindings: self.bindings(bindings),
                guard: self.expr_ref(guard),
                success: self.tree_ref(success),
                failure: self.tree_ref(failure),
            },
            raw::Tree::Switch { scrutinee, cases } => D::Switch {
                scrutinee: self.tcx.alloc_slice(scrutinee),
                cases: self.cases(cases),
            },
        }
    }

    fn bindings(&mut self, bs: &[raw::Binding]) -> &'tcx [mir::Binding<'tcx>] {
        let v: Vec<mir::Binding<'tcx>> = bs
            .iter()
            .map(|(var, path)| (VarId(*var), self.tcx.alloc_slice(path)))
            .collect();
        self.tcx.alloc_slice(&v)
    }

    fn cases(&mut self, c: &raw::Cases) -> mir::SwitchCases<'tcx> {
        use mir::SwitchCases as S;
        match c {
            raw::Cases::Int { cases, default } => {
                let mut cs = Vec::with_capacity(cases.len());
                for (n, t) in cases {
                    cs.push((*n, self.tree(t)));
                }
                let default = self.tree_ref(default);
                S::Int {
                    cases: self.tcx.alloc_slice(&cs),
                    default,
                }
            }
            raw::Cases::Bool { if_true, if_false } => S::Bool {
                if_true: self.tree_ref(if_true),
                if_false: self.tree_ref(if_false),
            },
            raw::Cases::Ctor(arms) => {
                let mut v = Vec::with_capacity(arms.len());
                for t in arms {
                    v.push(self.tree(t));
                }
                S::Ctor(self.tcx.alloc_slice(&v))
            }
            raw::Cases::Str { cases, default } => {
                let mut cs = Vec::with_capacity(cases.len());
                for (w, t) in cases {
                    cs.push((StringToken::from_words(*w), self.tree(t)));
                }
                let default = self.tree_ref(default);
                S::String {
                    cases: self.tcx.alloc_slice(&cs),
                    default,
                }
            }
            raw::Cases::Nullable { non_null, null } => S::Nullable {
                non_null: self.tree_ref(non_null),
                null: self.tree_ref(null),
            },
        }
    }

    /// Lower a typed raw node. The type is **always** read from `e.ty` (printed
    /// by the serializer for every node) — never invented — so the result is
    /// value-sound modulo type re-interning.
    fn expr(&mut self, e: &raw::Expr) -> mir::Expr<'tcx> {
        use mir::ExprKind as M;
        let ty = self.ty(&e.ty);
        let kind: M<'tcx> = match &*e.kind {
            raw::Kind::ConstInt(n) => M::ConstInt(*n),
            raw::Kind::ConstFloat(f) => M::ConstFloat(*f),
            raw::Kind::ConstBool(b) => M::ConstBool(*b),
            raw::Kind::GlobalStr(words) => M::GlobalStr(StringToken::from_words(*words)),
            raw::Kind::Var(v) => M::Var(VarId(*v)),
            raw::Kind::Poison => M::Poison,
            raw::Kind::Negate(x) => M::Negate(self.expr_ref(x)),
            raw::Kind::Not(x) => M::Not(self.expr_ref(x)),
            raw::Kind::Arith(l, op, r) => M::Arith(self.expr_ref(l), arith(*op), self.expr_ref(r)),
            raw::Kind::Cmp(l, op, r) => M::Cmp(self.expr_ref(l), cmp(*op), self.expr_ref(r)),
            raw::Kind::Cast(x, t) => {
                let t = self.ty(t);
                M::Cast(self.expr_ref(x), t)
            }
            raw::Kind::If(c, t, f) => M::If(self.expr_ref(c), self.expr_ref(t), self.expr_ref(f)),
            raw::Kind::RegionRun(x) => M::RegionRun(self.expr_ref(x)),
            raw::Kind::Proj(base, path) => {
                let base = self.expr_ref(base);
                M::Proj(base, self.tcx.alloc_slice(path))
            }
            raw::Kind::Assign(dst, field, src) => {
                M::Assign(self.expr_ref(dst), *field, self.expr_ref(src))
            }
            raw::Kind::Let { var, name, value } => {
                let name = self.names.intern(name);
                M::Let {
                    var: VarId(*var),
                    name,
                    value: self.expr_ref(value),
                }
            }
            raw::Kind::Seq(items) => M::Seq(self.expr_slice(items)),
            raw::Kind::Call {
                regional,
                symbol,
                args,
            } => {
                let callee = self.sym(symbol);
                M::Call {
                    callee,
                    args: self.expr_slice(args),
                    regional: *regional,
                }
            }
            raw::Kind::Ctor { symbol, args } => {
                let record = self.sym(symbol);
                M::Ctor {
                    record,
                    args: self.expr_slice(args),
                }
            }
            raw::Kind::Variant {
                symbol,
                variant,
                args,
            } => {
                let record = self.sym(symbol);
                M::Variant {
                    record,
                    variant: *variant,
                    args: self.expr_slice(args),
                }
            }
            raw::Kind::NullableCall(inner) => {
                M::NullableCall(inner.as_ref().map(|x| self.expr_ref(x)))
            }
            raw::Kind::ClosureCall { target, args } => {
                let target = self.expr_ref(target);
                M::ClosureCall {
                    target,
                    args: self.expr_slice(args),
                }
            }
            raw::Kind::Closure {
                captures,
                params,
                body,
            } => {
                let caps: Vec<(VarId, Ty<'tcx>)> = captures
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let captures = self.tcx.alloc_slice(&caps);
                let ps: Vec<(VarId, Ty<'tcx>)> = params
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let mir_params = self.tcx.alloc_slice(&ps);
                let body = self.expr_ref(body);
                M::Closure(mir::ClosureExpr {
                    captures,
                    params: mir_params,
                    body,
                })
            }
            raw::Kind::Match(scrut, tree) => {
                let scrut = self.expr_ref(scrut);
                M::Match(scrut, self.tree(tree))
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
    use crate::full::mir::print::Printer;
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

    #[test]
    fn roundtrips_records_and_projection() {
        // Exercises `@Pair{..}` constructors, the `Pair` record type, and
        // `proj(.., 0)`.
        roundtrip(
            "struct Pair { a: i32, b: i32 } \
             pub fn mk(x: i32, y: i32) -> Pair { Pair { a: x, b: y } } \
             pub fn fst(p: Pair) -> i32 { p.a }",
        );
    }

    #[test]
    fn roundtrips_regional_flex_signature() {
        // Exercises a turbofished, capability-prefixed record type:
        // `[flex] Cell::<i32>`.
        roundtrip(
            "struct [regional] Cell<T> { v: T, next: [field] Cell<T> } \
             regional fn id(c: [flex] Cell<i32>) -> i32 { 0 }",
        );
    }

    #[test]
    fn roundtrips_a_match() {
        // Exercises `match`, a ctor `switch scrut { #0 => {..} #1 => {..} }`,
        // a pattern binding (`v1=scrut.0`), and a leaf body.
        roundtrip(
            "enum Opt { None, Some(i32) } \
             pub fn unwrap(o: Opt) -> i32 { \
             match o { Opt::None => 0, Opt::Some(x) => x } }",
        );
    }

    #[test]
    fn roundtrips_a_string_literal() {
        roundtrip("pub fn greet() -> str { \"hi\" }");
    }

    #[test]
    fn roundtrips_an_int_match() {
        // Exercises an int `switch` with a `_` default arm.
        roundtrip(
            "pub fn classify(n: i32) -> i32 { \
             match n { 0 => 10, 1 => 20, _ => 30 } }",
        );
    }

    #[test]
    fn roundtrips_regional_generics() {
        // Mono'd regional record type + flex capability in a signature.
        roundtrip(
            "struct [regional] Cell<T> { v: T, next: [field] Cell<T> } \
             regional fn foo<T>(bar: [flex] T) -> i32 { 0 } \
             regional fn use_ok(c: [flex] Cell<i32>) -> i32 { foo(c) }",
        );
    }
}
