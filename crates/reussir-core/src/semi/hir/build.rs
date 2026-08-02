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
use crate::semi::ctxt::{
    DefaultCap, FfiImport, FfiPrelude, Record, RecordFields, TrampolineRoot, TransformScript,
    Variant,
};
use crate::semi::hir::grammar as hir_ir;
use crate::semi::hir::raw;
use crate::semi::hir::{
    ArithOp, ClosureExpr, CmpOp, DecisionTree, Expr, ExprId, ExprKind, Function, PatVarRef,
    SwitchCases, VarId,
};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{DefId, Flexivity, FpTy, GenericId, IntTy, Ty, TyCtxt, TyKind};
use crate::surface::{self, RecordKind};
use crate::utils::string::StringToken;

/// A parsed HIR program plus the fresh tables needed to re-print and resume it.
/// Carries exactly the pieces monomorphization reads — the elaborated functions,
/// the record declarations, and the trampoline roots — so it can be fed to
/// [`crate::full::mono::monomorphize`] via a `MonoInput`.
pub struct Parsed<'tcx> {
    /// The `.rri` interface header when the dump is a package interface;
    /// `None` for a plain HIR program. Gating (format/producer/package
    /// checks, and rejecting a header where a program is expected) is the
    /// caller's job.
    pub header: Option<raw::InterfaceHeader>,
    pub funcs: Vec<Function<'tcx>>,
    pub transform_anchors: Vec<DefId>,
    pub transform_scripts: Vec<TransformScript>,
    pub strings: Vec<(StringToken, String)>,
    pub records: FxHashMap<DefId, Record<'tcx>>,
    pub trampolines: Vec<TrampolineRoot<'tcx>>,
    pub ffi_imports: FxHashMap<DefId, FfiImport>,
    pub ffi_preludes: Vec<FfiPrelude>,
    pub defs: DefTable,
    pub names: Names,
    /// Serialized trait-bound names per generic binder (`$0 (T): Num`),
    /// keyed by the dump's `$n` ids — machine emissions number generics
    /// globally (one elaborator table), so the map is collision-free;
    /// a hand-written dump that reuses an id across items with different
    /// bounds is last-write-wins. Empty entries are omitted.
    pub generic_bounds: FxHashMap<GenericId, Vec<String>>,
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
    /// Every interned string, densely indexed by key — the extern declare
    /// pass re-interns the whole table into the consumer's interner (see
    /// [`crate::semi::externs`]).
    pub fn entries(&self) -> &[String] {
        &self.strings
    }

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
    for t in &raw.transforms {
        if let Some(id) = file_ref_check(&files, t.file) {
            return Err(id);
        }
    }
    let strings = raw::string_entries(&raw.strings)?;
    let mut b = Builder {
        tcx,
        names: Names::default(),
        defs: DefTable::new(),
        generic_bounds: FxHashMap::default(),
        next_expr_id: 0,
    };
    // Records first so their `DefId`s exist before function bodies reference them.
    let records: FxHashMap<DefId, Record<'tcx>> = raw.records.iter().map(|r| b.record(r)).collect();
    let funcs: Vec<Function<'tcx>> = raw.funcs.iter().map(|f| b.func(f)).collect();
    let transform_anchors = raw
        .funcs
        .iter()
        .zip(&funcs)
        .filter_map(|(raw, func)| raw.transform_anchor.then_some(func.def))
        .collect();
    let ffi_imports: FxHashMap<DefId, FfiImport> = raw
        .funcs
        .iter()
        .zip(&funcs)
        .filter_map(|(raw, func)| match &raw.body {
            raw::FuncBody::Ffi(body) => Some((
                func.def,
                FfiImport {
                    body: body.clone(),
                    span: span_of(raw.span),
                    file: file_of(raw.file),
                },
            )),
            _ => None,
        })
        .collect();
    let ffi_preludes: Vec<FfiPrelude> = raw
        .ffi_preludes
        .iter()
        .map(|p| FfiPrelude {
            body: p.body.clone(),
            span: span_of(p.span),
            file: file_of(p.file),
        })
        .collect();
    let transform_scripts = raw
        .transforms
        .iter()
        .map(|script| TransformScript {
            body: script.body.clone(),
            span: span_of(script.span),
            file: file_of(script.file),
        })
        .collect();
    let trampolines = raw.trampolines.iter().map(|t| b.trampoline(t)).collect();
    Ok(Parsed {
        header: raw.header.clone(),
        funcs,
        transform_anchors,
        transform_scripts,
        strings,
        records,
        trampolines,
        ffi_imports,
        ffi_preludes,
        defs: b.defs,
        names: b.names,
        generic_bounds: b.generic_bounds,
        files,
    })
}

struct Builder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    names: Names,
    defs: DefTable,
    /// Bound names collected off every parsed binder (see [`Parsed::generic_bounds`]).
    generic_bounds: FxHashMap<GenericId, Vec<String>>,
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
    /// definition and its call sites, in any order). Multi-segment paths
    /// (`pkg::m::f`) split back into module + item name, so mangling sees the
    /// same segments the original elaboration produced.
    fn function_def(&mut self, path: &str) -> DefId {
        let segs: Vec<_> = path.split("::").map(|s| self.names.intern(s)).collect();
        if let Some(id) = self.defs.lookup_function(&segs) {
            return id;
        }
        let (name, module) = segs.split_last().expect("paths are never empty");
        self.defs.set_module(module.to_vec());
        self.defs.declare_function(*name).expect("fresh fn decl")
    }

    fn record_def(&mut self, path: &str) -> DefId {
        let segs: Vec<_> = path.split("::").map(|s| self.names.intern(s)).collect();
        if let Some(id) = self.defs.lookup_record(&segs) {
            return id;
        }
        let (name, module) = segs.split_last().expect("paths are never empty");
        self.defs.set_module(module.to_vec());
        self.defs.declare_record(*name).expect("fresh record decl")
    }

    /// Rebuild a generic binder list (`ty_params`) and its regional subset from
    /// the parsed `$n` / `regional $n` markers.
    fn generics(&mut self, gs: &[raw::Generic]) -> (Vec<(TokenKey, GenericId)>, Vec<GenericId>) {
        let ty_params = gs
            .iter()
            .map(|g| {
                let key = match &g.name {
                    Some(name) => self.names.intern(name),
                    None => self.names.intern(&format!("${}", g.id)),
                };
                if !g.bounds.is_empty() {
                    self.generic_bounds
                        .insert(GenericId(g.id), g.bounds.clone());
                }
                (key, GenericId(g.id))
            })
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
                            (name, self.ty(&m.ty), m.is_field, member_vis(m))
                        })
                        .collect(),
                )
            }
            raw::RecordBody::Compound(members) => RecordFields::Unnamed(
                members
                    .iter()
                    .map(|m| (self.ty(&m.ty), m.is_field, member_vis(m)))
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
            raw::RecordBody::Opaque(_) => RecordFields::Opaque,
        };
        let ffi = match &r.body {
            raw::RecordBody::Opaque(path) => Some(path.clone()),
            _ => None,
        };
        let record = Record {
            def,
            name,
            visibility: if r.is_pub {
                crate::surface::Visibility::Public
            } else {
                crate::surface::Visibility::Private
            },
            ty_params,
            kind,
            default_cap,
            repr_fixed: r.repr_fixed,
            ffi,
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
        let body = match &f.body {
            raw::FuncBody::Expr(b) => Some(self.expr(b)),
            raw::FuncBody::None | raw::FuncBody::Ffi(_) => None,
        };
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
            raw::Ty::Char => self.tcx.mk_char(),
            raw::Ty::Unit => self.tcx.mk_unit(),
            raw::Ty::Bottom => self.tcx.mk(TyKind::Bottom),
            raw::Ty::Generic(g) => self.tcx.mk_generic(GenericId(*g)),
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
            raw::Ty::Array { elem, dims } => {
                let elem = self.ty(elem);
                self.tcx.mk_array(elem, dims)
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
            raw::Kind::ConstChar(c) => ExprKind::ConstChar(*c),
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
            raw::Kind::Intrinsic {
                family,
                name,
                imm,
                args,
            } => ExprKind::Intrinsic {
                op: crate::intrinsic::IntrinsicOp::parse(family, name, *imm)
                    .expect("known intrinsic in machine-emitted IR"),
                args: self.exprs(args),
            },
            raw::Kind::ArrayOp { op, args } => ExprKind::ArrayOp {
                op: crate::intrinsic::ArrayFn::parse(op)
                    .expect("known array op in machine-emitted IR"),
                args: self.exprs(args),
            },
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
            raw::Cases::Char { cases, default } => {
                let cases = cases.iter().map(|(c, t)| (*c, self.tree(t))).collect();
                SwitchCases::Char {
                    cases,
                    default: Box::new(self.tree(default)),
                }
            }
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

/// A member's `pub` marker as the surface visibility it round-trips.
fn member_vis(m: &raw::Member) -> surface::Visibility {
    if m.is_pub {
        surface::Visibility::Public
    } else {
        surface::Visibility::Private
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
            "item references file {id}, but the source-file table has {} entr(y/ies)",
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

            let strings = elab.strings.entries();
            let bounds = elab.bound_names();
            let text = Printer::new(&elab.defs, elab.resolver)
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .with_bounds(&bounds)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.transform_anchors.len(), elab.transform_anchors.len());
            assert_eq!(
                parsed
                    .transform_scripts
                    .iter()
                    .map(|script| &script.body)
                    .collect::<Vec<_>>(),
                elab.transform_scripts
                    .iter()
                    .map(|script| &script.body)
                    .collect::<Vec<_>>()
            );
            let text2 = Printer::new(&parsed.defs, &parsed.names)
                .with_transform_metadata(&parsed.transform_anchors, &parsed.transform_scripts)
                .with_ffi_metadata(&parsed.ffi_preludes, &parsed.ffi_imports)
                .with_bounds(&parsed.generic_bounds)
                .program(
                    &parsed.funcs,
                    &parsed.strings,
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
            let strings = elab.strings.entries();
            let bounds = elab.bound_names();
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache)
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .with_bounds(&bounds)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            assert!(
                text.contains("0 = \"<test>\";"),
                "file table missing:\n{text}"
            );
            assert!(text.contains(" in 0"), "item location missing:\n{text}");
            let parsed = parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.files, vec!["<test>".to_string()]);
            assert_eq!(parsed.transform_anchors.len(), elab.transform_anchors.len());
            assert_eq!(
                parsed
                    .transform_scripts
                    .iter()
                    .map(|script| (&script.body, script.file, script.span))
                    .collect::<Vec<_>>(),
                elab.transform_scripts
                    .iter()
                    .map(|script| (&script.body, script.file, script.span))
                    .collect::<Vec<_>>()
            );
            // Rebuild the render cache exactly as a driver would from the table.
            let mut cache2 = SourceCache::new();
            for name in &parsed.files {
                cache2.add_unavailable(name);
            }
            let text2 = Printer::with_sources(&parsed.defs, &parsed.names, &cache2)
                .with_transform_metadata(&parsed.transform_anchors, &parsed.transform_scripts)
                .with_ffi_metadata(&parsed.ffi_preludes, &parsed.ffi_imports)
                .with_bounds(&parsed.generic_bounds)
                .program(
                    &parsed.funcs,
                    &parsed.strings,
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
    fn source_file_table_escapes_debug_quoted_names() {
        use reussir_syntax::source::SourceCache;
        let source = "fn id(x: i32) -> i32 { x }";
        let file_name = r#"<quoted"path\name>"#;

        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let mut cache = SourceCache::new();
            cache.add_virtual(file_name, source);
            let strings = elab.strings.entries();
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            );
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
            "pub struct Pair { a: i64, b: i64 }\n\
             pub fn f(n: i64) -> i64 { let p = Pair { a: n, b: 1 }; p.a + p.b }",
        );
    }

    #[test]
    fn locations_survive_for_control_flow_and_matches() {
        roundtrip_with_locations(
            "pub enum Opt { None, Some(i64) }\n\
             pub fn g(o: Opt) -> i64 { match o { Opt::None => 0, Opt::Some(x) => { let y = x; y } } }",
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
        // The three FFI item forms: a foreign prelude, an opaque record, and
        // foreign-bodied imports (with an escape-y body to exercise quoting).
        roundtrip(
            r#"
            extern "rust" [{ use reussir_rt::collections::vec::Vec as RVec; }];

            #[ffi(rust = "::reussir_rt::collections::vec::Vec")]
            pub struct Vec<T>;

            #[ffi(import)]
            pub fn new<T>() -> Vec<T> [{ RVec::new() }];

            #[ffi(import)]
            pub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ RVec::push(v, "\"".len() as i64; x) }];

            pub fn use_it() -> Vec<f64> { push(new<f64>(), 1.0) }
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
            pub fn use_arc(a: Arc<Data>) -> Arc<Data> {
                a
            }
            pub fn maybe(a: Nullable<Arc<Data>>) -> Nullable<Arc<Data>> {
                a
            }
            pub fn generic<T>(a: Arc<T>) -> Arc<T> {
                a
            }
            pub fn make(v: i64) -> Arc<Data> {
                Arc<Data> { value: v }
            }
            pub fn make_some() -> Arc<Opt<i64>> {
                Arc<Opt<i64>>::Some{1}
            }
            pub fn make_none() -> Arc<Opt<i64>> {
                Arc<Opt<i64>>::None
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
    fn roundtrips_math_intrinsics() {
        roundtrip(
            "fn f(x: f64) -> bool { \
             core::intrinsic::math::isnan(core::intrinsic::math::fma(x, x, x, 127), 0) }",
        );
        roundtrip_with_locations(
            "pub fn g(x: f64) -> f64 { core::intrinsic::math::fpowi(x, 3, 1) }",
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
    fn roundtrips_if_without_else() {
        roundtrip("pub fn noop() { } pub fn tick(n: u64) { if n > 1 { noop() } }");
    }

    /// The sugar is pure syntax: an omitted `else` elaborates to exactly the
    /// HIR of an explicit empty `else {}`.
    #[test]
    fn if_without_else_matches_explicit_empty_else() {
        fn printed_hir(source: &str) -> String {
            with_tcx(|tcx| {
                let parse = reussir_syntax::parse(source);
                assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
                let prog = surface::program(&parse.root);
                let elab = elaborate(tcx, &prog, parse.resolver());
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
                let strings = elab.strings.entries();
                Printer::new(&elab.defs, elab.resolver).program(
                    &elab.elaborated,
                    &strings,
                    &elab.records,
                    &elab.trampolines,
                )
            })
        }

        let sugar = printed_hir("pub fn noop() { } pub fn tick(n: u64) { if n > 1 { noop() } }");
        let explicit =
            printed_hir("pub fn noop() { } pub fn tick(n: u64) { if n > 1 { noop() } else { } }");
        assert_eq!(sugar, explicit);
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
    fn roundtrips_a_match() {
        roundtrip(
            "pub enum Opt { None, Some(i32) } \
             pub fn unwrap(o: Opt) -> i32 { match o { Opt::None => 0, Opt::Some(x) => x } }",
        );
    }

    /// Record visibility survives the textual trip: `pub` prints on the
    /// record head and parses back to `Public`; an unmarked record stays
    /// `Private`. The marker is what lets a package interface (`--emit rri`)
    /// distinguish nameable surface records from private ones shipped only
    /// for layout.
    #[test]
    fn record_visibility_roundtrips() {
        use reussir_syntax::kind::Resolver as _;
        with_tcx(|tcx| {
            let source = "pub struct Data { value: i64 } \
                          struct Hidden { value: i64 } \
                          pub fn get(d: Data) -> i64 { d.value } \
                          fn peek(h: Hidden) -> i64 { h.value }";
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let strings = elab.strings.entries();
            let text = Printer::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            );
            assert!(text.contains("pub [shared] struct #Data"), "{text}");
            assert!(text.contains("\n[shared] struct #Hidden"), "{text}");

            let parsed = parse_program(tcx, &text).expect("re-parse");
            let vis_of = |name: &str| {
                parsed
                    .records
                    .values()
                    .find(|r| parsed.names.resolve(r.name) == name)
                    .expect("record present")
                    .visibility
            };
            assert_eq!(vis_of("Data"), surface::Visibility::Public);
            assert_eq!(vis_of("Hidden"), surface::Visibility::Private);
        });
        roundtrip(
            "pub struct Data { value: i64 } \
             struct Hidden { value: i64 } \
             pub fn get(d: Data) -> i64 { d.value } \
             fn peek(h: Hidden) -> i64 { h.value }",
        );
    }

    #[test]
    fn roundtrips_a_nullable_match() {
        roundtrip(
            r#"
            pub struct RcBox<T> { value: T }

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
            "pub enum L { C(i32, L), N } \
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
            "pub struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> } \
             regional fn foo<T>(bar: [flex] T) -> i32 { 0 } \
             regional fn use_ok(c: [flex] TestCell<i32>) -> i32 { foo(c) }",
        );
    }

    #[test]
    fn bounded_generics_roundtrip() {
        let source = "struct M<T : Num> { v: T }\n\
                      fn f<T : Integral + Sync>(x: T) -> T { x }\n\
                      fn g<T>(x: T) -> T { x }";
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let strings = elab.strings.entries();
            let bounds = elab.bound_names();
            let text = Printer::new(&elab.defs, elab.resolver)
                .with_bounds(&bounds)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            assert!(text.contains("struct #M<$0 (T): Num>"), "{text}");
            assert!(text.contains(": Integral + Sync>"), "{text}");
            // The unbounded generic prints bare.
            assert!(text.contains("fn #g<$2 (T)>"), "{text}");

            let parsed = parse_program(tcx, &text).expect("re-parse");
            let mut entries: Vec<(u32, Vec<String>)> = parsed
                .generic_bounds
                .iter()
                .map(|(g, bs)| (g.0, bs.clone()))
                .collect();
            entries.sort();
            assert_eq!(
                entries,
                [
                    (0, vec!["Num".to_string()]),
                    (1, vec!["Integral".to_string(), "Sync".to_string()]),
                ]
            );
        });
        // And byte-stability through both round-trip forms.
        roundtrip(source);
        roundtrip_with_locations(source);
    }

    /// A pre-bounds dump (bound-less binder) still parses, with empty bounds.
    #[test]
    fn boundless_binder_still_parses() {
        with_tcx(|tcx| {
            let text = "fn #id<$0 (T)>(v0 (x): $0) -> $0 {\n    v0 : $0\n}\n";
            let parsed = parse_program(tcx, text).expect("parse");
            assert!(parsed.generic_bounds.is_empty());
            assert_eq!(parsed.funcs.len(), 1);
        });
    }

    #[test]
    fn roundtrips_field_visibility() {
        with_tcx(|tcx| {
            let source = "pub struct [value] S { pub x: i64, y: i64 }\n\
                          pub struct T(pub i64, i64)\n\
                          pub struct [regional] R { pub c: [field] R, d: i64 }";
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);

            let strings = elab.strings.entries();
            let text = Printer::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            );
            // `pub` precedes the name, and the `field` marker; a private
            // member carries no marker.
            assert!(
                text.contains("pub [value] struct #S { pub \"x\": i64, \"y\": i64 };"),
                "{text}"
            );
            assert!(
                text.contains("pub [shared] struct #T { pub i64, i64 };"),
                "{text}"
            );
            assert!(text.contains("pub \"c\": field "), "{text}");
        });
        // And the marker survives print -> parse -> print byte-identically.
        roundtrip(
            "pub struct [value] S { pub x: i64, y: i64 }\n\
             pub struct T(pub i64, i64)\n\
             pub struct [regional] R { pub c: [field] R, d: i64 }",
        );
    }
}
