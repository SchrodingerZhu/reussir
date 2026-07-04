//! Re-intern pass: rebuild the (owned) HIR from the `raw` AST the grammar
//! produces. The MIR twin is [`crate::full::mir::build`]; the HIR differs in that
//! expressions are owned (`Box`/`Vec`, no arena), calls resolve a `#path` to a
//! fresh [`DefId`], and functions/types carry generics (`$n`) but no inference
//! holes (a fully elaborated HIR has none).
//!
//! Value-sound modulo arena re-interning, exactly as on the MIR side: every
//! node's type is read from the printed annotation. Crucially the IR's cross
//! references live in the text as **qualified `#path`s**, not `DefId`s — a
//! `DefId` is only a session-local handle reconstructed here (resolve-or-declare,
//! so a definition and its forward call sites agree), which is what keeps the
//! representation stable as multi-file resolution arrives.

use reussir_syntax::kind::{InternKey, Resolver, TokenKey};
use reussir_syntax::source::FileId;
use rustc_hash::FxHashMap;

use crate::ir_lex::lex;
use crate::semi::ctxt::{DefaultCap, Record, RecordFields, TrampolineRoot, Variant};
use crate::semi::hir::grammar as hir_ir;
use crate::semi::hir::raw;
use crate::semi::hir::{
    ArithOp, ClosureExpr, CmpOp, DecisionTree, Expr, ExprId, ExprKind, Function, PatVarRef,
    SwitchCases, VarId,
};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{DefId, Flexivity, FpTy, GenericId, IntTy, Ty, TyCtxt, TyKind};
use crate::surface::RecordKind;
use crate::utils::string::StringToken;

/// A parsed HIR program plus the fresh tables needed to re-print and resume it.
/// Carries exactly the pieces monomorphization reads — the elaborated functions,
/// the record declarations, and the trampoline roots — so it can be fed to
/// [`crate::full::mono::monomorphize`] via a `MonoInput`.
pub struct Parsed<'tcx> {
    pub funcs: Vec<Function<'tcx>>,
    pub records: FxHashMap<DefId, Record<'tcx>>,
    pub trampolines: Vec<TrampolineRoot<'tcx>>,
    pub defs: DefTable,
    pub names: Names,
    /// The dump's source-file table, in id order: each file's display name.
    /// Item/expr spans are byte offsets into these files (by [`FileId`] =
    /// table index); a `<bracketed>` name is virtual (content not on disk).
    /// Empty for a dump printed without locations.
    pub files: Vec<String>,
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
    // The file table must be dense and in-order: spans index it positionally.
    let mut files = Vec::with_capacity(raw.files.len());
    for (i, f) in raw.files.iter().enumerate() {
        if f.id as usize != i {
            return Err(format!(
                "source-file table is not dense: entry {} declares id {}",
                i, f.id
            ));
        }
        files.push(f.name.clone());
    }
    for f in &raw.funcs {
        if let Some(id) = file_ref_check(&files, f.file) {
            return Err(id);
        }
    }
    for r in &raw.records {
        if let Some(id) = file_ref_check(&files, r.file) {
            return Err(id);
        }
    }
    let mut b = Builder {
        tcx,
        names: Names::default(),
        defs: DefTable::new(),
        next_expr_id: 0,
    };
    // Records first so their `DefId`s exist before function bodies reference them.
    let records: FxHashMap<DefId, Record<'tcx>> = raw.records.iter().map(|r| b.record(r)).collect();
    let funcs = raw.funcs.iter().map(|f| b.func(f)).collect();
    let trampolines = raw.trampolines.iter().map(|t| b.trampoline(t)).collect();
    Ok(Parsed {
        funcs,
        records,
        trampolines,
        defs: b.defs,
        names: b.names,
        files,
    })
}

struct Builder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    names: Names,
    defs: DefTable,
    /// Monotonic counter for fresh [`ExprId`]s during the rebuild.
    next_expr_id: u32,
}

impl<'tcx> Builder<'_, 'tcx> {
    fn fresh_expr_id(&mut self) -> ExprId {
        let id = ExprId(self.next_expr_id);
        self.next_expr_id += 1;
        id
    }

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

    /// Rebuild a generic binder list (`ty_params`) and its regional subset from
    /// the parsed `$n` / `regional $n` markers.
    fn generics(&mut self, gs: &[raw::Generic]) -> (Vec<(TokenKey, GenericId)>, Vec<GenericId>) {
        let ty_params = gs
            .iter()
            .map(|g| (self.names.intern(&format!("${}", g.id)), GenericId(g.id)))
            .collect();
        let regional = gs
            .iter()
            .filter(|g| g.regional)
            .map(|g| GenericId(g.id))
            .collect();
        (ty_params, regional)
    }

    /// Rebuild the `Record` metadata mono reads, including the ground field
    /// layout. A struct rebuilds as [`RecordFields::Named`] when its fields carry
    /// names (debug info reads them) and [`RecordFields::Unnamed`] for a tuple.
    fn record(&mut self, r: &raw::Record) -> (DefId, Record<'tcx>) {
        let def = self.record_def(&r.path);
        let name = self.names.intern(&r.path);
        let (ty_params, regional_generics) = self.generics(&r.generics);
        let kind = match r.kind {
            raw::RecordKind::Struct => RecordKind::StructKind,
            raw::RecordKind::Enum => RecordKind::EnumKind,
        };
        let default_cap = match r.default_cap {
            raw::DefaultCap::Value => DefaultCap::Value,
            raw::DefaultCap::Shared => DefaultCap::Shared,
            raw::DefaultCap::Regional => DefaultCap::Regional,
        };
        let fields = match &r.body {
            // Named when the fields carry names (a struct), unnamed for a tuple.
            raw::RecordBody::Compound(members) if members.iter().any(|m| m.name.is_some()) => {
                RecordFields::Named(
                    members
                        .iter()
                        .map(|m| {
                            let name = self.names.intern(m.name.as_deref().unwrap_or(""));
                            (name, self.ty(&m.ty), m.is_field)
                        })
                        .collect(),
                )
            }
            raw::RecordBody::Compound(members) => RecordFields::Unnamed(
                members
                    .iter()
                    .map(|m| (self.ty(&m.ty), m.is_field))
                    .collect(),
            ),
            raw::RecordBody::Variant(variants) => RecordFields::Variants(
                variants
                    .iter()
                    .map(|v| Variant {
                        name: self.names.intern(&v.name),
                        fields: v.fields.iter().map(|t| self.ty(t)).collect(),
                    })
                    .collect(),
            ),
        };
        let record = Record {
            def,
            name,
            ty_params,
            kind,
            default_cap,
            fields: Some(fields),
            regional_generics,
            span: span_of(r.span),
            file: file_of(r.file),
        };
        (def, record)
    }

    fn trampoline(&mut self, t: &raw::Tramp) -> TrampolineRoot<'tcx> {
        TrampolineRoot {
            name: t.name.clone(),
            abi: t.abi.clone(),
            target: self.function_def(&t.target),
            ty_args: self.tys(&t.ty_args),
        }
    }

    fn func(&mut self, f: &raw::Func) -> Function<'tcx> {
        let def = self.function_def(&f.path);
        let name = self.names.intern(&f.path);
        let (generics, regional_generics) = self.generics(&f.generics);
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
            span: span_of(f.span),
            file: file_of(f.file),
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
            raw::Ty::Nullable(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_nullable(inner)
            }
            raw::Ty::Record { cap, path, args } => {
                let def = self.record_def(path);
                let args: Vec<Ty<'tcx>> = args.iter().map(|a| self.ty(a)).collect();
                self.tcx.mk_record(def, &args, flexivity(*cap))
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

    /// Lower a typed raw node, always reading its type from `e.ty` (never
    /// invented), so the rebuilt HIR is value-sound modulo type re-interning.
    fn expr(&mut self, e: &raw::Expr) -> Expr<'tcx> {
        let ty = self.ty(&e.ty);
        let kind: ExprKind<'tcx> = match &*e.kind {
            raw::Kind::ConstInt(n) => ExprKind::ConstInt(self.tcx.alloc(n.clone())),
            raw::Kind::ConstFloat(f) => ExprKind::ConstFloat(self.tcx.alloc(f.clone())),
            raw::Kind::ConstBool(b) => ExprKind::ConstBool(*b),
            raw::Kind::GlobalStr(words) => ExprKind::GlobalStr(StringToken::from_words(*words)),
            raw::Kind::Var(v) => ExprKind::Var(VarId(*v)),
            raw::Kind::Poison => ExprKind::Poison,
            raw::Kind::Negate(x) => ExprKind::Negate(self.boxed(x)),
            raw::Kind::Not(x) => ExprKind::Not(self.boxed(x)),
            raw::Kind::Arith(l, op, r) => ExprKind::Arith(self.boxed(l), arith(*op), self.boxed(r)),
            raw::Kind::Cmp(l, op, r) => ExprKind::Cmp(self.boxed(l), cmp(*op), self.boxed(r)),
            raw::Kind::Cast(x, t) => {
                let t = self.ty(t);
                ExprKind::Cast(self.boxed(x), t)
            }
            raw::Kind::If(c, t, f) => ExprKind::If(self.boxed(c), self.boxed(t), self.boxed(f)),
            raw::Kind::RegionRun(x) => ExprKind::RegionRun(self.boxed(x)),
            raw::Kind::Proj(base, path) => ExprKind::Proj(self.boxed(base), path.clone()),
            raw::Kind::Assign(dst, field, src) => {
                ExprKind::Assign(self.boxed(dst), *field, self.boxed(src))
            }
            raw::Kind::Let {
                var,
                name,
                name_span,
                value,
            } => {
                let name = self.names.intern(name);
                ExprKind::Let {
                    var: VarId(*var),
                    name,
                    span: span_of(*name_span),
                    value: self.boxed(value),
                }
            }
            raw::Kind::Seq(items) => ExprKind::Seq(self.exprs(items)),
            raw::Kind::FuncCall {
                regional,
                path,
                ty_args,
                args,
            } => {
                let target = self.function_def(path);
                ExprKind::FuncCall {
                    target,
                    ty_args: self.tys(ty_args),
                    args: self.exprs(args),
                    regional: *regional,
                }
            }
            raw::Kind::CompoundCall {
                path,
                ty_args,
                args,
            } => {
                let target = self.record_def(path);
                ExprKind::CompoundCall {
                    target,
                    ty_args: self.tys(ty_args),
                    args: self.exprs(args),
                }
            }
            raw::Kind::VariantCall {
                path,
                ty_args,
                variant,
                args,
            } => {
                let target = self.record_def(path);
                ExprKind::VariantCall {
                    target,
                    ty_args: self.tys(ty_args),
                    variant: *variant,
                    args: self.exprs(args),
                }
            }
            raw::Kind::NullableCall(inner) => {
                ExprKind::NullableCall(inner.as_ref().map(|x| self.boxed(x)))
            }
            raw::Kind::ClosureCall { target, args } => {
                let target = self.boxed(target);
                ExprKind::ClosureCall {
                    target,
                    args: self.exprs(args),
                }
            }
            raw::Kind::Closure {
                captures,
                params,
                body,
            } => {
                let captures = captures
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let params = params
                    .iter()
                    .map(|(v, t)| (VarId(*v), self.ty(t)))
                    .collect();
                let body = self.boxed(body);
                ExprKind::Closure(ClosureExpr {
                    captures,
                    params,
                    body,
                })
            }
            raw::Kind::Match(scrut, tree) => {
                let scrut = self.boxed(scrut);
                ExprKind::Match(scrut, self.tree(tree))
            }
        };
        Expr {
            kind,
            ty,
            span: span_of(e.span),
            id: self.fresh_expr_id(),
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
                let cases = cases
                    .iter()
                    .map(|(n, t)| (self.tcx.alloc(n.clone()), self.tree(t)))
                    .collect();
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
            raw::Cases::Str { cases, default } => {
                let cases = cases
                    .iter()
                    .map(|(w, t)| (StringToken::from_words(*w), self.tree(t)))
                    .collect();
                SwitchCases::String {
                    cases,
                    default: Box::new(self.tree(default)),
                }
            }
            raw::Cases::Nullable { non_null, null } => SwitchCases::Nullable {
                non_null: Box::new(self.tree(non_null)),
                null: Box::new(self.tree(null)),
            },
        }
    }
}

fn flexivity(c: raw::Cap) -> Flexivity {
    match c {
        raw::Cap::None => Flexivity::Irrelevant,
        raw::Cap::Flex => Flexivity::Flex,
        raw::Cap::Rigid => Flexivity::Rigid,
        raw::Cap::Regional => Flexivity::Regional,
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

/// A raw `[start..end]` span into the owning item's file.
fn span_of(sp: Option<raw::Span>) -> Option<crate::surface::Span> {
    sp.map(|(start, end)| crate::surface::Span { start, end })
}

/// A raw `in <id>` file reference; a dump printed without locations has none
/// and everything indexes the primary file.
fn file_of(f: Option<u32>) -> FileId {
    f.map_or(FileId::ROOT, FileId::from_index)
}

/// Reject an `in <id>` reference past the end of the dump's file table.
fn file_ref_check(files: &[String], file: Option<u32>) -> Option<String> {
    match file {
        Some(id) if id as usize >= files.len() => Some(format!(
            "function references file {id}, but the source-file table has {} entr(y/ies)",
            files.len()
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::semi::elaborate;
    use crate::semi::hir::print::Printer;
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

            let text = Printer::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &elab.records,
                &elab.trampolines,
            );
            let parsed = parse_program(tcx, &text).expect("re-parse");
            let text2 = Printer::new(&parsed.defs, &parsed.names).program(
                &parsed.funcs,
                &parsed.records,
                &parsed.trampolines,
            );
            assert_eq!(
                text, text2,
                "round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    /// The lossless form: print WITH the source cache (file table + spans),
    /// parse back, re-print against a cache rebuilt from the dump's own file
    /// table, and assert text equality — so files, item locations, node
    /// spans, and `let`-name spans all survive the trip.
    fn roundtrip_with_locations(source: &str) {
        use reussir_syntax::source::SourceCache;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let cache = SourceCache::single("<test>", source);
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache).program(
                &elab.elaborated,
                &elab.records,
                &elab.trampolines,
            );
            assert!(
                text.contains("0 = \"<test>\";"),
                "file table missing:\n{text}"
            );
            assert!(text.contains(" in 0"), "item location missing:\n{text}");
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.files, vec!["<test>".to_string()]);
            // Rebuild the render cache exactly as a driver would from the table.
            let mut cache2 = SourceCache::new();
            for name in &parsed.files {
                cache2.add_unavailable(name);
            }
            let text2 = Printer::with_sources(&parsed.defs, &parsed.names, &cache2).program(
                &parsed.funcs,
                &parsed.records,
                &parsed.trampolines,
            );
            assert_eq!(
                text, text2,
                "lossless round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    #[test]
    fn locations_survive_the_textual_round_trip() {
        roundtrip_with_locations(
            "struct Pair { a: i64, b: i64 }\n\
             pub fn f(n: i64) -> i64 { let p = Pair { a: n, b: 1 }; p.a + p.b }",
        );
    }

    #[test]
    fn locations_survive_for_control_flow_and_matches() {
        roundtrip_with_locations(
            "enum Opt { None, Some(i64) }\n\
             pub fn g(o: Opt) -> i64 { match o { Opt::None => 0, Opt::Some(x) => { let y = x; y } } }",
        );
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
    fn roundtrips_a_string_literal() {
        roundtrip("pub fn greet() -> str { \"hi\" }");
    }

    #[test]
    fn roundtrips_a_match() {
        roundtrip(
            "enum Opt { None, Some(i32) } \
             pub fn unwrap(o: Opt) -> i32 { match o { Opt::None => 0, Opt::Some(x) => x } }",
        );
    }

    #[test]
    fn roundtrips_a_nullable_match() {
        roundtrip(
            r#"
            struct RcBox<T> { value: T }

            pub fn unwrap_or(n: Nullable<RcBox<i32>>, d: i32) -> i32 {
                match n {
                    Nullable::NonNull(b) => b.value,
                    Nullable::Null => d
                }
            }
            "#,
        );
    }

    #[test]
    fn roundtrips_nested_scrutinee_paths() {
        // A nested pattern produces depth-2 scrutinee paths (`scrut.1.0`),
        // whose adjacent indices lex as one float-shaped token — the grammar
        // must split it back (regression: this used to fail to re-parse).
        roundtrip(
            "enum L { C(i32, L), N } \
             pub fn two(l: L) -> i32 { \
             match l { L::C(x, L::C(y, N)) => x + y, L::C(x, t) => x, L::N => 0 } }",
        );
    }

    #[test]
    fn roundtrips_extreme_and_radix_literals() {
        // Arbitrary-precision literals survive the text round trip at full
        // width; radix forms normalize to decimal; floats stay exact decimals.
        roundtrip(
            "pub fn big() -> u64 { 18446744073709551615 } \
             pub fn hex() -> u32 { 0xDEAD_BEEF } \
             pub fn neg() -> i64 { -9223372036854775808 } \
             pub fn tenth() -> f64 { 0.1 } \
             pub fn tiny() -> f64 { 2.5e-300 }",
        );
    }

    #[test]
    fn roundtrips_regional_generics() {
        // A generic `[flex]`-bound function (so `regional_generics` is non-empty)
        // and a turbofished regional call.
        roundtrip(
            "struct [regional] Cell<T> { v: T, next: [field] Cell<T> } \
             regional fn foo<T>(bar: [flex] T) -> i32 { 0 } \
             regional fn use_ok(c: [flex] Cell<i32>) -> i32 { foo(c) }",
        );
    }
}
