//! Re-intern pass: rebuild the arena-allocated MIR from the owned [`raw`] AST
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

use crate::full::mir::{self, grammar as ir, raw};
use crate::ir_lex::lex;
use crate::semi::ctxt::{DefaultCap, TransformScript};
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Flexivity, FpTy, IntTy, Ty, TyCtxt, TyKind};
use crate::utils::string::StringToken;

/// A parsed program plus the fresh tables needed to re-print it.
pub struct Parsed<'tcx> {
    pub program: mir::Program<'tcx>,
    pub defs: DefTable,
    pub names: Names,
    /// The dump's source-file table, in id order: each file's display name.
    /// Function/expr spans are byte offsets into these files (by `FileId` =
    /// table index); a `<bracketed>` name is virtual (content not on disk).
    /// Empty for a dump printed without locations.
    pub files: Vec<String>,
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
    for t in &raw.transforms {
        if let Some(id) = file_ref_check(&files, t.file) {
            return Err(id);
        }
    }
    let string_literals = raw::string_entries(&raw.strings)?;
    let mut b = Builder {
        tcx,
        symbols: Rodeo::default(),
        names: Names::default(),
        defs: DefTable::new(),
        ids: mir::ExprIdGen::default(),
    };
    let program = b.program(raw, string_literals);
    Ok(Parsed {
        program,
        defs: b.defs,
        names: b.names,
        files,
    })
}

struct Builder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    symbols: Rodeo,
    names: Names,
    defs: DefTable,
    /// Fresh [`mir::ExprId`] anchors, regenerated deterministically on each
    /// parse. Ids are not part of the textual form, so a freshly re-interned
    /// tree re-numbers from zero without affecting round-trip text equality.
    ids: mir::ExprIdGen,
}

impl<'tcx> Builder<'_, 'tcx> {
    fn sym(&mut self, s: &str) -> mir::Symbol {
        mir::Symbol(self.symbols.get_or_intern(s))
    }

    /// The `DefId` for a record path, declaring it on first sight.
    /// Multi-segment paths (`pkg::m::Cell`) split back into module + item
    /// name, so re-prints display the same qualified path.
    fn record_def(&mut self, path: &str) -> crate::semi::ty::DefId {
        let segs: Vec<_> = path.split("::").map(|s| self.names.intern(s)).collect();
        if let Some(id) = self.defs.lookup_record(&segs) {
            return id;
        }
        let (name, module) = segs.split_last().expect("paths are never empty");
        self.defs.set_module(module.to_vec());
        self.defs.declare_record(*name).expect("fresh record decl")
    }

    /// Rebuild a record instance (symbol, nominal type, default capability, and
    /// ground layout) from its parsed declaration.
    fn record(&mut self, r: &raw::RecordDecl) -> mir::RecordInstance<'tcx> {
        let layout = match &r.body {
            raw::RecordBody::Compound(members) => {
                let ms: Vec<mir::Member<'tcx>> = members
                    .iter()
                    .map(|m| mir::Member {
                        ty: self.ty(&m.ty),
                        is_field: m.is_field,
                        name: m.name.as_deref().map(|s| self.sym(s)),
                    })
                    .collect();
                mir::RecordLayout::Compound(self.tcx.alloc_slice(&ms))
            }
            raw::RecordBody::Variant(variants) => {
                let vs: Vec<mir::VariantDef<'tcx>> = variants
                    .iter()
                    .map(|v| {
                        let fields: Vec<Ty<'tcx>> = v.fields.iter().map(|t| self.ty(t)).collect();
                        mir::VariantDef {
                            name: self.sym(&v.name),
                            symbol: self.sym(&v.symbol),
                            fields: self.tcx.intern_tys(&fields),
                        }
                    })
                    .collect();
                mir::RecordLayout::Variant(self.tcx.alloc_slice(&vs))
            }
            raw::RecordBody::Opaque {
                rust_name,
                drop_hook,
            } => mir::RecordLayout::Opaque {
                rust_name: self.sym(rust_name),
                drop_hook: self.sym(drop_hook),
            },
        };
        mir::RecordInstance {
            symbol: self.sym(&r.symbol),
            ty: self.ty(&r.ty),
            default_cap: def_cap(r.default_cap),
            repr_fixed: r.repr_fixed,
            layout,
        }
    }

    fn program(
        &mut self,
        raw: raw::Program,
        string_literals: Vec<(StringToken, String)>,
    ) -> mir::Program<'tcx> {
        let records: Vec<mir::RecordInstance<'tcx>> =
            raw.records.iter().map(|r| self.record(r)).collect();
        let trampolines: Vec<mir::Trampoline> = raw
            .trampolines
            .iter()
            .map(|t| mir::Trampoline {
                export: self.sym(&t.export),
                abi: t.abi.clone(),
                target: self.sym(&t.target),
                import: t.import,
            })
            .collect();
        let ffi_imports: Vec<mir::FfiImport> = raw
            .ffi_imports
            .iter()
            .map(|f| mir::FfiImport {
                symbol: self.sym(&f.symbol),
                boundary: self.sym(&f.boundary),
                texture: f.texture.clone(),
            })
            .collect();
        let ffi_textures: Vec<mir::FfiTexture> = raw
            .ffi_textures
            .iter()
            .map(|t| mir::FfiTexture {
                anchor: self.sym(&t.anchor),
                texture: t.texture.clone(),
            })
            .collect();
        let ffi_rc_glue: Vec<mir::FfiRcGlue<'tcx>> = raw
            .ffi_rc_glue
            .iter()
            .map(|g| mir::FfiRcGlue {
                ty: self.ty(&g.ty),
                acquire: self.sym(&g.acquire),
                release: self.sym(&g.release),
            })
            .collect();
        let functions: Vec<mir::Function<'tcx>> = raw.funcs.iter().map(|f| self.func(f)).collect();
        let transform_scripts = raw
            .transforms
            .into_iter()
            .map(|script| TransformScript {
                body: script.body,
                span: script
                    .span
                    .map(|(start, end)| crate::surface::Span { start, end }),
                file: script
                    .file
                    .map_or(reussir_syntax::source::FileId::ROOT, |index| {
                        reussir_syntax::source::FileId::from_index(index)
                    }),
            })
            .collect();

        // Move the freshly-built symbol interner into the program.
        let symbols = std::mem::take(&mut self.symbols);
        mir::Program {
            functions,
            records,
            trampolines,
            string_literals,
            transform_scripts,
            ffi_imports,
            ffi_textures,
            ffi_rc_glue,
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
            transform_anchor: f.transform_anchor,
            visibility: if f.is_pub {
                crate::surface::Visibility::Public
            } else {
                crate::surface::Visibility::Private
            },
            // Not serialized: the set is recomputed from HIR bodies wherever
            // they exist (elaboration, or a loaded .hir); a re-entered .mir
            // has no generic bodies left to witness reachability.
            mono_exported: false,
            is_regional: f.regional,
            params,
            return_ty,
            body,
            // `in <id>` when the dump carries locations; else the primary file.
            file: f.file.map_or(reussir_syntax::source::FileId::ROOT, |i| {
                reussir_syntax::source::FileId::from_index(i)
            }),
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
            raw::Ty::Char => self.tcx.mk_char(),
            raw::Ty::Unit => self.tcx.mk_unit(),
            raw::Ty::Bottom => self.tcx.mk(TyKind::Bottom),
            raw::Ty::Nullable(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_nullable(inner)
            }
            raw::Ty::Cell { inner, kind } => {
                let inner = self.ty(inner);
                self.tcx.mk_cell(inner, *kind)
            }
            raw::Ty::Arc(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_arc(inner)
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
            raw::Ty::Array { elem, dims } => {
                let elem = self.ty(elem);
                self.tcx.mk_array(elem, dims)
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
                    cs.push((self.tcx.alloc(n.clone()), self.tree(t)));
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
            raw::Cases::Char { cases, default } => {
                let mut cs = Vec::with_capacity(cases.len());
                for (c, t) in cases {
                    cs.push((*c, self.tree(t)));
                }
                S::Char {
                    cases: self.tcx.alloc_slice(&cs),
                    default: self.tree_ref(default),
                }
            }
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
            raw::Kind::ConstInt(n) => M::ConstInt(self.tcx.alloc(n.clone())),
            raw::Kind::ConstFloat(f) => M::ConstFloat(self.tcx.alloc(f.clone())),
            raw::Kind::ConstBool(b) => M::ConstBool(*b),
            raw::Kind::ConstChar(c) => M::ConstChar(*c),
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
            raw::Kind::Intrinsic {
                family,
                name,
                imm,
                args,
            } => M::Intrinsic {
                op: crate::intrinsic::IntrinsicOp::parse(family, name, *imm)
                    .expect("known intrinsic in machine-emitted IR"),
                args: self.expr_slice(args),
            },
            raw::Kind::ArrayOp { op, args } => M::ArrayOp {
                op: crate::intrinsic::ArrayFn::parse(op)
                    .expect("known array op in machine-emitted IR"),
                args: self.expr_slice(args),
            },
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
            id: self.ids.fresh(),
            kind,
            ty,
            span: e
                .span
                .map(|(start, end)| crate::surface::Span { start, end }),
        }
    }
}

fn cap(c: raw::Cap) -> Flexivity {
    match c {
        raw::Cap::None => Flexivity::Irrelevant,
        raw::Cap::Flex => Flexivity::Flex,
        raw::Cap::Rigid => Flexivity::Rigid,
        raw::Cap::Regional => Flexivity::Regional,
    }
}

fn def_cap(c: raw::DefCap) -> DefaultCap {
    match c {
        raw::DefCap::Value => DefaultCap::Value,
        raw::DefCap::Shared => DefaultCap::Shared,
        raw::DefCap::Regional => DefaultCap::Regional,
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

/// Reject an `in <id>` reference past the end of the dump's file table.
fn file_ref_check(files: &[String], file: Option<u32>) -> Option<String> {
    match file {
        Some(id) if id as usize >= files.len() => Some(format!(
            "item references file {id}, but the source-file table has {} entr(y/ies)",
            files.len()
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::full::mir::print::Printer;
    use crate::full::mir::raw;
    use crate::full::mono::monomorphize;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// Print a program, parse it back, re-print, and assert text equality
    /// (the text-faithful round-trip contract).
    fn roundtrip(source: &str) {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");

            let text = Printer::new(&elab.defs, elab.resolver).program(&full);
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(
                parsed
                    .program
                    .functions
                    .iter()
                    .filter(|function| function.transform_anchor)
                    .count(),
                full.functions
                    .iter()
                    .filter(|function| function.transform_anchor)
                    .count()
            );
            assert_eq!(
                parsed
                    .program
                    .transform_scripts
                    .iter()
                    .map(|script| &script.body)
                    .collect::<Vec<_>>(),
                full.transform_scripts
                    .iter()
                    .map(|script| &script.body)
                    .collect::<Vec<_>>()
            );
            let text2 = Printer::new(&parsed.defs, &parsed.names).program(&parsed.program);
            assert_eq!(
                text, text2,
                "round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    /// The lossless form: print WITH the source cache (file table, per-fn
    /// `in <file>`, node spans), parse back, re-print against a cache rebuilt
    /// from the dump's own table, and assert text equality.
    fn roundtrip_with_locations(source: &str) {
        use reussir_syntax::source::SourceCache;
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");

            let cache = SourceCache::single("<test>", source);
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache).program(&full);
            assert!(
                text.contains("0 = \"<test>\";"),
                "file table missing:\n{text}"
            );
            assert!(text.contains(" in 0"), "fn file reference missing:\n{text}");
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.files, vec!["<test>".to_string()]);
            assert_eq!(
                parsed
                    .program
                    .transform_scripts
                    .iter()
                    .map(|script| (&script.body, script.file, script.span))
                    .collect::<Vec<_>>(),
                full.transform_scripts
                    .iter()
                    .map(|script| (&script.body, script.file, script.span))
                    .collect::<Vec<_>>()
            );
            let mut cache2 = SourceCache::new();
            for name in &parsed.files {
                cache2.add_unavailable(name);
            }
            let text2 = Printer::with_sources(&parsed.defs, &parsed.names, &cache2)
                .program(&parsed.program);
            assert_eq!(
                text, text2,
                "lossless round-trip mismatch\n=== printed ===\n{text}\n=== reparsed ===\n{text2}"
            );
        });
    }

    #[test]
    fn source_file_table_escapes_debug_quoted_names() {
        use reussir_syntax::source::SourceCache;
        let source = "fn id(x: i32) -> i32 { x }";
        let file_name = r#"<quoted"path\name>"#;

        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");

            let mut cache = SourceCache::new();
            cache.add_virtual(file_name, source);
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache).program(&full);
            assert!(
                text.contains(r#"0 = "<quoted\"path\\name>";"#),
                "file table was not debug-escaped:\n{text}"
            );
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.files, vec![file_name.to_string()]);
        });
    }

    #[test]
    fn locations_survive_the_textual_round_trip() {
        roundtrip_with_locations(
            "pub fn fib(n: u64) -> u64 { \
             let m = n + 1; \
             if n <= 1 { m } else { fib(n - 1) + fib(n - 2) } }",
        );
    }

    #[test]
    fn roundtrips_array_types() {
        // The `[elem; extents…]` type in parameter and return position;
        // rank-1 and rank-2. The array *ops* have their own round-trip test.
        roundtrip(
            "pub fn id(a: [f64; 8]) -> [f64; 8] { a } \
             pub fn fst(m: [i32; 4, 4], n: [i32; 4, 4]) -> [i32; 4, 4] { m }",
        );
    }

    #[test]
    fn roundtrips_arrays() {
        // All five `array#…` ops; the tabulate/fold kernels are ordinary
        // closures (literal and named) riding the closure grammar.
        roundtrip(
            r#"
            pub fn make() -> [f64; 8] { core::intrinsic::array::tabulate<[f64; 8]>(|i| i as f64) }
            pub fn ones() -> [f64; 8] { core::intrinsic::array::splat<[f64; 8]>(1.0) }
            pub fn s(a: [f64; 8]) -> f64 { core::intrinsic::array::fold(a, 0.0, |acc, x| acc + x) }
            pub fn n(f: (f64, f64) -> f64, a: [f64; 8]) -> f64 { core::intrinsic::array::fold(a, 0.0, f) }
            pub fn b(a: [f64; 8], i: i64, v: f64) -> [f64; 8] {
                core::intrinsic::array::set(a, i, core::intrinsic::array::get(a, i) + v)
            }
            "#,
        );
    }

    #[test]
    fn roundtrips_ffi_items() {
        // The MIR-side FFI carriage: opaque record layouts, import
        // trampolines, wrapper/hook textures, and rc glue (the shared
        // record element forces a transparent wrapper + acquire/release).
        roundtrip(
            r#"
            extern "rust" [{ use reussir_rt::collections::vec::Vec as RVec; }];

            #[ffi(rust = "::reussir_rt::collections::vec::Vec")]
            pub struct Vec<T>;

            pub struct Item { value: i64 }

            #[ffi(import)]
            pub fn new<T>() -> Vec<T> [{ RVec::new() }];

            #[ffi(import)]
            pub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ RVec::push(v, x) }];

            pub fn use_it() -> Vec<Item> { push(new<Item>(), Item{1}) }
            "#,
        );
    }

    #[test]
    fn roundtrips_cells() {
        roundtrip(
            r#"
            struct [value] Boxed<T> { value: Cell<T> }
            pub fn use_cell(c: Cell<i64>) -> Cell<i64> {
                c
            }
            pub fn nested(c: Cell<Cell<i64>>) -> Cell<Cell<i64>> {
                c
            }
            pub fn use_refcell(c: RefCell<i64>) -> RefCell<i64> {
                c
            }
            pub fn mixed(c: Cell<RefCell<i64>>) -> Cell<RefCell<i64>> {
                c
            }
            "#,
        );
    }

    #[test]
    fn roundtrips_arcs() {
        roundtrip(
            r#"
            pub struct Data { value: i64 }
            pub enum Opt<T> { None, Some(T) }
            fn generic<T>(a: T) -> T { a }
            pub fn use_arc(a: Arc<Data>) -> Arc<Data> {
                generic(a)
            }
            pub fn maybe(a: Nullable<Arc<Data>>) -> Nullable<Arc<Data>> {
                a
            }
            pub fn make(v: i64) -> Arc<Data> {
                Arc<Data> { value: v }
            }
            pub fn make_some() -> Arc<Opt<i64>> {
                Arc<Opt<i64>>::Some{1}
            }
            pub fn read(a: Arc<Data>) -> i64 {
                a.value
            }
            pub fn get(o: Arc<Opt<i64>>) -> i64 {
                match o {
                    Opt::Some(x) => x,
                    Opt::None => 0
                }
            }
            "#,
        );
    }

    #[test]
    fn roundtrips_transform_metadata() {
        roundtrip_with_locations(
            "#[transform_anchor]\n\
             pub fn transform(transform_anchor: i32) -> i32 { transform_anchor }\n\
             transform [{\n  transform.yield\n}];",
        );
    }

    #[test]
    fn roundtrips_math_intrinsics() {
        roundtrip(
            "pub fn f(x: f64) -> f64 { \
             core::intrinsic::math::sqrt(core::intrinsic::math::powf(x, 2.0, 127), 0) }",
        );
    }

    #[test]
    fn roundtrips_a_scalar_function() {
        roundtrip(
            "pub fn fib(n: u64) -> u64 { \
             let m = n + 1; \
             if n <= 1 { m } else { fib(n - 1) + fib(n - 2) } }",
        );
    }

    /// The empty else-block synthesized for an `if` without `else` survives
    /// the MIR textual round-trip.
    #[test]
    fn roundtrips_if_without_else() {
        roundtrip("pub fn noop() { } pub fn tick(n: u64) { if n > 1 { noop() } }");
    }

    #[test]
    fn roundtrips_records_and_projection() {
        // Exercises `@Pair{..}` constructors, the `Pair` record type, and
        // `proj(.., 0)`.
        roundtrip(
            "pub struct Pair { a: i32, b: i32 } \
             pub fn mk(x: i32, y: i32) -> Pair { Pair { a: x, b: y } } \
             pub fn fst(p: Pair) -> i32 { p.a }",
        );
    }

    #[test]
    fn roundtrips_regional_flex_signature() {
        // Exercises a turbofished, capability-prefixed record type:
        // `[flex] TestCell::<i32>`.
        roundtrip(
            "pub struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> } \
             regional fn id(c: [flex] TestCell<i32>) -> i32 { 0 }",
        );
    }

    #[test]
    fn roundtrips_a_match() {
        // Exercises `match`, a ctor `switch scrut { #0 => {..} #1 => {..} }`,
        // a pattern binding (`v1=scrut.0`), and a leaf body. Each variant prints
        // with its mangled payload symbol (`@_RNv… Name(..)`), which must re-parse.
        roundtrip(
            "pub enum Opt { None, Some(i32) } \
             pub fn unwrap(o: Opt) -> i32 { \
             match o { Opt::None => 0, Opt::Some(x) => x } }",
        );
    }

    #[test]
    fn roundtrips_a_generic_enum_construction() {
        // A monomorphized generic enum: its instantiated variant payload symbols
        // carry the type arguments on the record (`Opt<i32>::Some` →
        // `_RNvI…lE…`), and the `@sym Name(..)` form must survive print → parse.
        roundtrip(
            "pub enum Opt<T> { None, Some(T) } \
             pub fn wrap(x: i32) -> Opt<i32> { Opt::Some{x} }",
        );
    }

    #[test]
    fn roundtrips_a_string_literal() {
        roundtrip("pub fn greet() -> str { \"hi\" }");
    }

    #[test]
    fn roundtrips_char_and_string_patterns() {
        roundtrip(
            r#"pub fn char_pick(c: char) -> i32 { match c { 'x' => 1, '\n' => 2, _ => 0 } }
               pub fn str_pick(s: str) -> i32 { match s { "hi" => 1, "bye" => 2, _ => 0 } }"#,
        );
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
            "pub struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> } \
             regional fn foo<T>(bar: [flex] T) -> i32 { 0 } \
             regional fn use_ok(c: [flex] TestCell<i32>) -> i32 { foo(c) }",
        );
    }

    #[test]
    fn roundtrips_an_integral_float() {
        // `1.0` must print with its point (else it re-lexes as an int).
        roundtrip("pub fn f() -> f64 { 1.0 }");
    }

    #[test]
    fn roundtrips_a_unicode_identifier() {
        // Source identifiers can be Unicode (XID); they print verbatim and the
        // lexer must accept them.
        roundtrip("pub fn f(\u{3b1}: i32) -> i32 { \u{3b1} }");
    }

    #[test]
    fn roundtrips_a_zero_arg_closure() {
        // A parameterless closure has type `() -> ret`; the closure-type grammar
        // must accept zero parameters.
        roundtrip("pub fn f(n: i32) -> i32 { (|| n)() }");
    }

    #[test]
    #[should_panic(expected = "Unicode scalar value")]
    fn rejects_surrogate_char_atoms() {
        // `char#<n>` in machine-emitted IR must be a valid Unicode scalar
        // value; U+D800 is a surrogate.
        raw::char_scalar(&crate::literal::Integer::from(0xD800u32));
    }

    #[test]
    fn rejects_string_table_token_mismatch() {
        let entry = raw::StringEntry {
            token: [0; 4],
            payload: "hi".into(),
        };
        let err = raw::string_entries(&[entry]).unwrap_err();
        assert!(err.contains("does not match payload"), "{err}");
    }
}
