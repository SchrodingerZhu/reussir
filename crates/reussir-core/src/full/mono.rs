//! Monomorphization: the Semi → Full lowering driver.
//!
//! Elaboration leaves functions polymorphic. Monomorphization specializes them
//! and **lowers** them into the [`crate::full::mir`]: starting from a set of
//! **roots**, it instantiates each reachable function at every concrete
//! type-argument tuple it is used with, grounds its types (see
//! [`crate::full::subst`]), resolves every callee to its v0 [`mir::Symbol`] (see
//! [`crate::full::mangle`]), and **erases the generic apparatus** — the MIR has
//! no type arguments on any node and dispatches purely by interned symbol.
//!
//! # Roots
//!
//! Every **local** non-generic function (emitted even if uncalled —
//! whole-program DCE is the backend's job) plus every trampoline target.
//! Generic functions are emitted once per distinct instantiation discovered
//! from a reachable body. Imported functions ([`MonoExterns`]) are never
//! roots: they are reached only through calls, instantiating from their
//! shipped bodies or lowering to declarations from their prototypes.
//!
//! # Worklist
//!
//! An [`Instance`] is `(DefId, &[Ty])`; its interned type-argument slice compares
//! by element pointer-identity, so structurally-equal instantiations collapse to
//! one entry. Popping an instance lowers its body to MIR; each `FuncCall` found
//! there resolves to a callee symbol *and* enqueues that callee instance, so
//! discovery and symbol resolution happen in the same pass.
//!
//! # Regional check at the call boundary
//!
//! Some generics must be instantiated *regional* — only regional records can be
//! flex or occupy a `[field]` link. The elaborator records these requirements
//! (the `[flex]`/`[field]` colorings are dropped on a bare generic, as in the
//! reference, but the implied requirement is kept):
//!
//! * `Function::regional_generics` — a generic used at a `[flex]` position
//!   (`bar: [flex] T`) or assigned into a flex link in the body;
//! * `Record::regional_generics` — a generic at the head of a `[field]` link
//!   (`inner: [field] T`).
//!
//! Mono verifies them here — when a function instance is popped, and when a
//! record instance is finalized — reporting any non-regional (value/shared)
//! instantiation. The *flexivity* rules (flex capture/return/assign) are a
//! separate concern enforced at the Semi stage on concrete records, not here.
//!
//! # `Arc` inner check at the instantiation boundary
//!
//! `ty_eval` rejects a concrete `Arc` inner that is not a shared rc box (a
//! `[shared]` record, an array, or a closure) but lets a generic inner
//! (`Arc<T>`) through. An instantiation — e.g. a trampoline's explicit type
//! arguments — is the first point where such a type grounds, so mono re-walks
//! every substituted type and reports any `Arc<i32>`/`Arc<Cell<…>>` it finds
//! (see [`arc_inner_rejection`]).

use std::collections::VecDeque;

use lasso::Rodeo;
use rustc_hash::{FxHashMap, FxHashSet};

use reussir_syntax::kind::{Resolver, TokenKey};
use reussir_syntax::source::FileId;

use crate::full::ffi::{FfiCtx, WrapperDecl, WrapperParam};
use crate::full::mangle::Mangler;
use crate::full::mir;
use crate::full::subst::{Subst, subst_ty};
use crate::full::{ffi as ffi_render};
use crate::literal;
use crate::semi::ctxt::{
    DefaultCap, Elaborator, FfiImport, FfiPrelude, Record, RecordFields, Report, Severity,
    TrampolineRoot, TransformScript,
};
use crate::semi::hir::{self, DecisionTree, Expr, ExprKind, Function, SwitchCases};
use crate::semi::resolve::DefTable;
use crate::semi::traits::sync::{SyncEnv, SyncVerdict, wf_arc};
use crate::semi::ty::{DefId, Flexivity, Ty, TyCtxt, TyKind};
use crate::semi::ty::{FpTy, IntTy};
use crate::semi::ty_eval::arc_inner_rejection;
use crate::surface::Span;
use crate::utils::string::StringToken;

/// Exactly the part of an [`Elaborator`] that monomorphization reads. Taking
/// this rather than the whole elaborator lets a program **reconstructed from the
/// textual HIR** — which has no traits/inference/builtins, only the elaborated
/// functions, records, and trampoline roots — be monomorphized identically.
pub struct MonoInput<'a, 'tcx> {
    pub tcx: &'a TyCtxt<'tcx>,
    pub defs: &'a DefTable,
    pub resolver: &'a dyn Resolver<TokenKey>,
    pub elaborated: &'a [Function<'tcx>],
    pub records: &'a FxHashMap<DefId, Record<'tcx>>,
    pub trampolines: &'a [TrampolineRoot<'tcx>],
    pub transform_anchors: &'a [DefId],
    pub transform_scripts: &'a [TransformScript],
    pub ffi_imports: &'a FxHashMap<DefId, FfiImport>,
    pub ffi_preludes: &'a [FfiPrelude],
    pub strings: Vec<(StringToken, String)>,
    /// The dependency-interface tables (`--extern`); empty defaults for a
    /// single-package compilation.
    pub externs: MonoExterns<'a, 'tcx>,
}

/// What loaded dependency interfaces contribute to monomorphization: the
/// remapped extern tables an [`Elaborator`] collected via
/// `declare_extern_package`.
///
/// Imported functions join instantiation lookup (`by_def`) but are **never
/// roots** — a ground import compiles in its own package and only its symbol
/// crosses the boundary — and never seed the export closure
/// ([`super::interface::export_closure`] walks `elaborated` only: a consumer
/// does not re-export its dependency). An enqueued instance whose imported
/// function carries a body instantiates here exactly like a local one (its
/// `_RI` symbol dedups `weak_odr` against any other emitter); a bodyless
/// prototype lowers to a `mir::Function` declaration the link resolves
/// against the dependency's artifact.
#[derive(Default)]
pub struct MonoExterns<'a, 'tcx> {
    pub functions: &'a [Function<'tcx>],
    /// String literals imported bodies reference. Tokens are
    /// content-addressed, so they merge into the program's table by token.
    pub strings: &'a [(StringToken, String)],
    /// Foreign bodies of imported `#[ffi(import)]` functions, keyed by their
    /// consumer defs; their textures render per instance like local ones.
    pub ffi_imports: Option<&'a FxHashMap<DefId, FfiImport>>,
    /// Preludes of the imported ffi bodies' files (file ids already remapped
    /// into the consumer's cache, so the per-file pairing holds).
    pub ffi_preludes: &'a [FfiPrelude],
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Borrow this elaborator as the [`MonoInput`] mono consumes.
    pub fn mono_input(&self) -> MonoInput<'_, 'tcx> {
        MonoInput {
            tcx: self.tcx,
            defs: &self.defs,
            resolver: self.resolver,
            elaborated: &self.elaborated,
            records: &self.records,
            trampolines: &self.trampolines,
            transform_anchors: &self.transform_anchors,
            transform_scripts: &self.transform_scripts,
            ffi_imports: &self.ffi_imports,
            ffi_preludes: &self.ffi_preludes,
            strings: self.strings.entries(),
            externs: MonoExterns {
                functions: &self.extern_functions,
                strings: &self.extern_strings,
                ffi_imports: Some(&self.extern_ffi_imports),
                ffi_preludes: &self.extern_ffi_preludes,
            },
        }
    }
}

/// A ground instantiation of a top-level item: its [`DefId`] applied to an
/// interned tuple of concrete type arguments (empty for a non-generic item).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: DefId,
    pub ty_args: &'tcx [Ty<'tcx>],
}

/// The private ground functions whose symbols must stay externally linkable:
/// everything reachable from a generic body that a *foreign* compilation may
/// instantiate.
///
/// A `pub` generic's body is compiled wherever it is instantiated — under a
/// package HIR index, in another package's objects entirely. Any ground call
/// inside such a body then resolves at link time against *this* package's
/// artifacts, so an `internal` callee would leave the foreign instance with an
/// unresolvable symbol. The set is seeded with every `pub` generic body and
/// closed transitively through the (private) generics those bodies call —
/// their bodies travel and instantiate the same way. Ground `pub` functions
/// are not seeds: their code is compiled here and only their (already
/// external) symbol crosses the boundary.
pub fn mono_exports(input: &MonoInput<'_, '_>) -> FxHashSet<DefId> {
    // Derived from the interface export closure — the prototypes it ships are
    // exactly the ground functions foreign instantiations can reach, and the
    // private ones among them are the symbols that must not go `internal`.
    // One traversal, so the linkage promise and the `.rri` contents cannot
    // drift.
    let by_def: FxHashMap<DefId, &Function<'_>> =
        input.elaborated.iter().map(|f| (f.def, f)).collect();
    super::interface::export_closure(input)
        .protos
        .into_iter()
        .filter(|def| {
            by_def
                .get(def)
                .is_some_and(|f| f.visibility == crate::surface::Visibility::Private)
        })
        .collect()
}

/// Visit every expression node under `expr` — through closure bodies, match
/// arm bodies, and guards — in pre-order. The single canonical body walk:
/// derive per-fact walkers (call targets, mentioned types, string literals)
/// from it instead of re-encoding the `ExprKind` shape.
pub(crate) fn for_each_expr<'e, 'tcx>(expr: &'e Expr<'tcx>, f: &mut impl FnMut(&'e Expr<'tcx>)) {
    use ExprKind::*;
    f(expr);
    match &expr.kind {
        GlobalStr(_) | ConstChar(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_)
        | Poison => {}
        Negate(e) | Not(e) | Cast(e, _) | RegionRun(e) | Proj(e, _) => {
            for_each_expr(e, f);
        }
        Arith(a, _, b) | Cmp(a, _, b) | Assign(a, _, b) => {
            for_each_expr(a, f);
            for_each_expr(b, f);
        }
        If(c, t, e) => {
            for_each_expr(c, f);
            for_each_expr(t, f);
            for_each_expr(e, f);
        }
        Match(scrutinee, tree) => {
            for_each_expr(scrutinee, f);
            tree_exprs(tree, f);
        }
        Let { value, .. } => for_each_expr(value, f),
        Seq(es) => es.iter().for_each(|e| for_each_expr(e, f)),
        FuncCall { args, .. } => {
            args.iter().for_each(|e| for_each_expr(e, f));
        }
        // Compound/variant targets name records, not functions; only their
        // arguments contain expressions.
        CompoundCall { args, .. }
        | VariantCall { args, .. }
        | Intrinsic { args, .. }
        | ArrayOp { args, .. } => {
            args.iter().for_each(|e| for_each_expr(e, f));
        }
        NullableCall(inner) => {
            if let Some(e) = inner {
                for_each_expr(e, f);
            }
        }
        Closure(c) => for_each_expr(&c.body, f),
        ClosureCall { target, args } => {
            for_each_expr(target, f);
            args.iter().for_each(|e| for_each_expr(e, f));
        }
    }
}

fn tree_exprs<'e, 'tcx>(tree: &'e DecisionTree<'tcx>, f: &mut impl FnMut(&'e Expr<'tcx>)) {
    use DecisionTree::*;
    match tree {
        Uncovered | Unreachable => {}
        Leaf { body, .. } => for_each_expr(body, f),
        Guard {
            guard,
            success,
            failure,
            ..
        } => {
            for_each_expr(guard, f);
            tree_exprs(success, f);
            tree_exprs(failure, f);
        }
        Switch { cases, .. } => match cases {
            SwitchCases::Int { cases, default } => {
                cases.iter().for_each(|(_, t)| tree_exprs(t, f));
                tree_exprs(default, f);
            }
            SwitchCases::Bool { if_true, if_false } => {
                tree_exprs(if_true, f);
                tree_exprs(if_false, f);
            }
            SwitchCases::Char { cases, default } => {
                cases.iter().for_each(|(_, t)| tree_exprs(t, f));
                tree_exprs(default, f);
            }
            SwitchCases::Ctor(trees) => trees.iter().for_each(|t| tree_exprs(t, f)),
            SwitchCases::String { cases, default } => {
                cases.iter().for_each(|(_, t)| tree_exprs(t, f));
                tree_exprs(default, f);
            }
            SwitchCases::Nullable { non_null, null } => {
                tree_exprs(non_null, f);
                tree_exprs(null, f);
            }
        },
    }
}

/// Monomorphize an elaborated program into its ground Full MIR, alongside any
/// diagnostics raised by the call-boundary regional check.
pub fn monomorphize<'a, 'tcx>(input: &MonoInput<'a, 'tcx>) -> (mir::Program<'tcx>, Vec<Report>) {
    let tcx: &'a TyCtxt<'tcx> = input.tcx;
    // Instantiation looks up local *and* imported functions; only locals are
    // seeded as roots below.
    let by_def: FxHashMap<DefId, &Function<'tcx>> = input
        .elaborated
        .iter()
        .chain(input.externs.functions)
        .map(|f| (f.def, f))
        .collect();
    let exports = mono_exports(input);

    let mut driver = Driver {
        tcx,
        resolver: input.resolver,
        mangler: Mangler::new(input.defs, input.resolver),
        symbols: Rodeo::default(),
        queue: VecDeque::new(),
        seen: FxHashSet::default(),
        origins: FxHashMap::default(),
        records: FxHashSet::default(),
        record_defs: input.records,
        reported_arcs: FxHashSet::default(),
        reports: Vec::new(),
        cur_file: FileId::ROOT,
        ids: mir::ExprIdGen::default(),
    };

    // Seed roots: every *local* non-generic function, then every trampoline
    // target. Imported grounds must not seed — their definitions live in the
    // dependency's artifact and only calls reach them (as declarations).
    for f in input.elaborated {
        if f.generics.is_empty() {
            driver.enqueue(f.def, tcx.intern_tys(&[]), f.span);
        }
    }
    for t in input.trampolines {
        driver.enqueue(t.target, tcx.intern_tys(&t.ty_args), None);
    }

    // FFI accumulators. `ffi_mangler` is a second (stateless) mangler the
    // rendering closures borrow, keeping `driver` free for mutation.
    let ffi_mangler = Mangler::new(input.defs, input.resolver);
    let instance_symbol = |def: DefId, args: &'tcx [Ty<'tcx>]| ffi_mangler.mangle_instance(def, args);
    let mut ffi_imports_out: Vec<mir::FfiImport> = Vec::new();
    let mut ffi_textures: Vec<mir::FfiTexture> = Vec::new();
    let mut import_trampolines: Vec<mir::Trampoline> = Vec::new();
    // Shared-record wrappers referenced by any texture, keyed by instance
    // symbol (deterministic order) — each becomes one acquire/release pair.
    let mut glue: std::collections::BTreeMap<String, Ty<'tcx>> = std::collections::BTreeMap::new();

    let mut functions = Vec::new();
    while let Some(inst) = driver.queue.pop_front() {
        let Some(func) = by_def.get(&inst.def) else {
            // Neither a local definition nor an imported body/prototype: with
            // externs in play a silent skip would manufacture an undefined
            // symbol at link time, so report at the discovering call site. A
            // stale or incomplete dependency interface is the expected cause;
            // an item that failed to elaborate never reaches mono (the driver
            // stops on elaboration errors).
            let (file, span) = driver
                .origins
                .get(&inst)
                .copied()
                .unwrap_or((FileId::ROOT, None));
            driver.cur_file = file;
            let path = input.defs.path(inst.def).display(input.resolver);
            driver.error(
                span,
                format!(
                    "no body or prototype recorded for `{path}`: the definition is \
                     neither in this package nor in a loaded dependency interface \
                     (is a stale `.rri` being passed via `--extern`?)"
                ),
            );
            continue;
        };
        driver.cur_file = func.file;
        driver.reported_arcs.clear();

        // Bind this instance's generics, then ground the signature and lower the
        // body into the MIR.
        let mut subst = Subst::default();
        for ((_, gid), &ty) in func.generics.iter().zip(inst.ty_args.iter()) {
            subst.insert(*gid, ty);
            // Call-boundary check: a generic used at a `[flex]` position must be
            // instantiated with a regional record (only regional records can be
            // flex). Value/shared records and scalars are rejected.
            if func.regional_generics.contains(gid) && !is_regional_arg(ty) {
                driver.error(
                    func.span,
                    "a `[flex]` type parameter requires a regional record, but this \
                     instantiation supplied a non-regional (value/shared) type"
                        .to_string(),
                );
            }
        }
        let params: Vec<mir::Param<'tcx>> = func
            .params
            .iter()
            .map(|(name, var, ty)| {
                let ty = subst_ty(tcx, *ty, &subst);
                driver.note_records(ty);
                driver.check_arc_inners(ty, func.span);
                mir::Param {
                    name: *name,
                    var: *var,
                    ty,
                }
            })
            .collect();
        let return_ty = subst_ty(tcx, func.return_ty, &subst);
        driver.note_records(return_ty);
        driver.check_arc_inners(return_ty, func.span);
        let body = func.body.as_ref().map(|b| driver.lower_ref(b, &subst));

        let symbol = driver.symbol_of(inst.def, inst.ty_args);

        // An `#[ffi(import)]` instance (local or imported — a polyffi texture
        // inside a shipped body compiles in the consumer's pipeline): render
        // its boundary wrapper texture and register the import trampoline.
        // The function itself stays a bodyless declaration under `symbol`
        // (emitted below as usual).
        let ffi_import = input.ffi_imports.get(&inst.def).or_else(|| {
            input
                .externs
                .ffi_imports
                .and_then(|imports| imports.get(&inst.def))
        });
        if let Some(fimport) = ffi_import {
            let fctx = FfiCtx {
                records: input.records,
                instance_symbol: &instance_symbol,
            };
            let mut decls: std::collections::BTreeMap<String, WrapperDecl<'tcx>> =
                std::collections::BTreeMap::new();
            let mut wrapper_params = Vec::new();
            let mut ok = true;
            for p in &params {
                let name = input.resolver.resolve(p.name).to_owned();
                let ident = ffi_render::rust_ident(&name);
                let rust_ty = fctx.rust_name(p.ty, &mut decls);
                match (ident, rust_ty) {
                    (Ok(ident), Ok(rust_ty)) => {
                        wrapper_params.push(WrapperParam { ident, rust_ty });
                    }
                    (ident, rust_ty) => {
                        for err in [ident.err(), rust_ty.err()].into_iter().flatten() {
                            driver.error(
                                fimport.span,
                                format!("`#[ffi(import)]` parameter `{name}`: {err}"),
                            );
                        }
                        ok = false;
                    }
                }
            }
            let ret = match *return_ty.kind() {
                TyKind::Unit => None,
                _ => match fctx.rust_name(return_ty, &mut decls) {
                    Ok(r) => Some(r),
                    Err(err) => {
                        driver.error(
                            fimport.span,
                            format!("`#[ffi(import)]` return type: {err}"),
                        );
                        ok = false;
                        None
                    }
                },
            };
            if ok {
                let param_tys: Vec<Ty<'tcx>> = params.iter().map(|p| p.ty).collect();
                let trivial = fctx.classify_trivial(&param_tys, return_ty);
                let ret_direct = matches!(*return_ty.kind(), TyKind::Unit)
                    || fctx.integer_like(return_ty);
                // `[:T:]` placeholders in the body substitute to the
                // instance's Rust spellings.
                let mut placeholders: FxHashMap<&str, String> = FxHashMap::default();
                for ((gname, _), &ty) in func.generics.iter().zip(inst.ty_args.iter()) {
                    if let Ok(rendered) = fctx.rust_name(ty, &mut decls) {
                        placeholders.insert(input.resolver.resolve(*gname), rendered);
                    }
                }
                let body_text =
                    ffi_render::substitute_placeholders(&fimport.body, &placeholders);
                // File ids are one space (externs remap at declaration), so
                // the per-file prelude pairing holds across both tables.
                let preludes: Vec<&str> = input
                    .ffi_preludes
                    .iter()
                    .chain(input.externs.ffi_preludes)
                    .filter(|p| p.file == fimport.file)
                    .map(|p| p.body.as_str())
                    .collect();
                let boundary_name =
                    format!("{}_ffi", driver.symbols.resolve(&symbol.0));
                let texture = ffi_render::import_texture(
                    &preludes,
                    &decls,
                    &boundary_name,
                    &wrapper_params,
                    ret.as_deref(),
                    trivial,
                    ret_direct,
                    &body_text,
                );
                let boundary = mir::Symbol(driver.symbols.get_or_intern(&boundary_name));
                ffi_imports_out.push(mir::FfiImport {
                    symbol,
                    boundary,
                    texture,
                });
                import_trampolines.push(mir::Trampoline {
                    export: boundary,
                    abi: "C".to_owned(),
                    target: symbol,
                    import: true,
                });
                for (sym, decl) in decls {
                    glue.entry(sym).or_insert(decl.ty);
                }
            }
        }

        functions.push(mir::Function {
            symbol,
            transform_anchor: input.transform_anchors.contains(&func.def),
            visibility: func.visibility,
            mono_exported: exports.contains(&func.def),
            is_regional: func.is_regional,
            params,
            return_ty,
            body,
            file: func.file,
        });
    }

    // Close the record set over fields before resolving layouts: a record used
    // only as another record's field is otherwise never collected, leaving its
    // layout unresolved.
    driver.close_records_over_fields(input.records, tcx);

    // Reject inline-recursive record *instances* before resolving layouts: a
    // `[value]` record stored inside itself through a chain of inline members
    // has no finite layout (the LLVM conversion would recurse forever). Semi
    // rejects the def-level cycles it can see; grounding can create new ones
    // through generics (`[value] Wrap<T> { x: T }` instantiated at a value
    // record that contains `Wrap` back), so the closed instance set is walked
    // here with every member substituted.
    driver.reject_infinite_value_recursion(tcx);

    // Resolve the discovered record instances to symbols. Collect the keys first
    // so the interner can be borrowed mutably while we map them.
    let record_keys: Vec<(DefId, &'tcx [Ty<'tcx>])> = driver.records.iter().copied().collect();
    // Call-boundary regional check for records: a `[field] T` link requires `T`
    // regional, so a record instance must supply a regional type for each such
    // generic parameter.
    for &(def, args) in &record_keys {
        let Some(record) = input.records.get(&def) else {
            continue;
        };
        driver.cur_file = record.file;
        for &gid in &record.regional_generics {
            let Some(pos) = record.ty_params.iter().position(|(_, g)| *g == gid) else {
                continue;
            };
            if matches!(args.get(pos), Some(&ty) if !is_regional_arg(ty)) {
                driver.error(
                    record.span,
                    "a `[field]` link requires a regional record, but this record was \
                     instantiated at a non-regional (value/shared) type"
                        .to_string(),
                );
            }
        }
    }
    let mut records: Vec<mir::RecordInstance<'tcx>> = Vec::with_capacity(record_keys.len());
    for (def, args) in record_keys {
        let symbol = driver.symbol_of(def, args);
        // Layout is capability-independent, so canonicalize the coloring.
        let ty = tcx.mk_record(def, args, Flexivity::Irrelevant);
        // An opaque `#[ffi]` instance: render its identity string and drop
        // hook, and emit the hook's texture.
        if let Some(record) = input.records.get(&def).filter(|r| r.ffi.is_some()) {
            let fctx = FfiCtx {
                records: input.records,
                instance_symbol: &instance_symbol,
            };
            let mut decls: std::collections::BTreeMap<String, WrapperDecl<'tcx>> =
                std::collections::BTreeMap::new();
            let layout = match fctx.rust_name(ty, &mut decls) {
                Ok(rust_name) => {
                    let hook_name = format!("{}_ffi_drop", driver.symbols.resolve(&symbol.0));
                    let texture = ffi_render::drop_texture(&decls, &hook_name, &rust_name);
                    let layout = mir::RecordLayout::Opaque {
                        rust_name: mir::Symbol(driver.symbols.get_or_intern(&rust_name)),
                        drop_hook: mir::Symbol(driver.symbols.get_or_intern(&hook_name)),
                    };
                    ffi_textures.push(mir::FfiTexture {
                        anchor: symbol,
                        texture,
                    });
                    for (sym, decl) in decls {
                        glue.entry(sym).or_insert(decl.ty);
                    }
                    layout
                }
                Err(err) => {
                    driver.cur_file = record.file;
                    driver.error(
                        record.span,
                        format!(
                            "cannot instantiate the `#[ffi]` record at this type \
                             argument: {err}"
                        ),
                    );
                    mir::RecordLayout::Compound(&[])
                }
            };
            records.push(mir::RecordInstance {
                symbol,
                ty,
                default_cap: DefaultCap::Shared,
                repr_fixed: false,
                layout,
            });
            continue;
        }
        // A record whose definition is missing (it failed to elaborate) gets an
        // empty value layout so lowering still has a well-formed instance.
        let (default_cap, layout) = match input.records.get(&def) {
            Some(record) => resolve_layout(
                tcx,
                record,
                args,
                input.resolver,
                &driver.mangler,
                &mut driver.symbols,
            ),
            None => (DefaultCap::Value, mir::RecordLayout::Compound(&[])),
        };
        // `#[repr(fixed)]` is carried straight from the collected record; a
        // missing definition (elaboration failure) is treated as non-fixed.
        let repr_fixed = input
            .records
            .get(&def)
            .is_some_and(|record| record.repr_fixed);
        records.push(mir::RecordInstance {
            symbol,
            ty,
            default_cap,
            repr_fixed,
            layout,
        });
    }

    let mut trampolines: Vec<mir::Trampoline> = input
        .trampolines
        .iter()
        .map(|t| mir::Trampoline {
            export: mir::Symbol(driver.symbols.get_or_intern(&t.name)),
            abi: t.abi.clone(),
            target: driver.symbol_of(t.target, tcx.intern_tys(&t.ty_args)),
            import: false,
        })
        .collect();
    trampolines.extend(import_trampolines);

    // Reussir-side rc glue for every shared-record wrapper any texture
    // referenced. The wrapped instances were noted during rendering, so
    // their layouts are resolved above.
    let ffi_rc_glue: Vec<mir::FfiRcGlue<'tcx>> = glue
        .into_iter()
        .map(|(sym, ty)| mir::FfiRcGlue {
            ty,
            acquire: mir::Symbol(driver.symbols.get_or_intern(format!("{sym}_ffi_acquire"))),
            release: mir::Symbol(driver.symbols.get_or_intern(format!("{sym}_ffi_release"))),
        })
        .collect();

    // Deterministic output: sort by mangled text.
    functions.sort_by_cached_key(|f| driver.symbols.resolve(&f.symbol.0));
    records.sort_by_cached_key(|r| driver.symbols.resolve(&r.symbol.0));
    ffi_imports_out.sort_by_cached_key(|f| driver.symbols.resolve(&f.symbol.0).to_owned());
    ffi_textures.sort_by_cached_key(|t| driver.symbols.resolve(&t.anchor.0).to_owned());

    // Imported bodies' `GlobalStr` tokens must resolve in the program's
    // table. Tokens are content-addressed, so a literal both sides use is
    // one entry.
    let mut string_literals = input.strings.clone();
    let mut seen_strings: FxHashSet<StringToken> =
        string_literals.iter().map(|(token, _)| *token).collect();
    for (token, text) in input.externs.strings {
        if seen_strings.insert(*token) {
            string_literals.push((*token, text.clone()));
        }
    }

    let Driver {
        symbols, reports, ..
    } = driver;
    (
        mir::Program {
            functions,
            records,
            trampolines,
            string_literals,
            transform_scripts: input.transform_scripts.to_vec(),
            ffi_imports: ffi_imports_out,
            ffi_textures,
            ffi_rc_glue,
            symbols,
        },
        reports,
    )
}

/// Resolve a record instance to its ground layout: the field types with the
/// instance's generics substituted away (variants expanded), plus the
/// default capability that selects the lowering. Variant names are interned into
/// `symbols` (the program's symbol table).
fn resolve_layout<'tcx>(
    tcx: &TyCtxt<'tcx>,
    record: &Record<'tcx>,
    args: &'tcx [Ty<'tcx>],
    resolver: &dyn Resolver<TokenKey>,
    mangler: &Mangler<'_>,
    symbols: &mut Rodeo,
) -> (DefaultCap, mir::RecordLayout<'tcx>) {
    let mut subst = Subst::default();
    for ((_, gid), &ty) in record.ty_params.iter().zip(args.iter()) {
        subst.insert(*gid, ty);
    }
    let layout = match record.fields.as_ref() {
        Some(RecordFields::Named(fields)) => {
            let members: Vec<mir::Member<'tcx>> = fields
                .iter()
                .map(|(name, ty, is_mut)| mir::Member {
                    ty: subst_ty(tcx, *ty, &subst),
                    is_field: *is_mut,
                    name: Some(mir::Symbol(symbols.get_or_intern(resolver.resolve(*name)))),
                })
                .collect();
            mir::RecordLayout::Compound(tcx.alloc_slice(&members))
        }
        Some(RecordFields::Unnamed(fields)) => {
            let members: Vec<mir::Member<'tcx>> = fields
                .iter()
                .map(|(ty, is_mut)| mir::Member {
                    ty: subst_ty(tcx, *ty, &subst),
                    is_field: *is_mut,
                    name: None,
                })
                .collect();
            mir::RecordLayout::Compound(tcx.alloc_slice(&members))
        }
        Some(RecordFields::Variants(variants)) => {
            let vdefs: Vec<mir::VariantDef<'tcx>> = variants
                .iter()
                .map(|v| {
                    let fields: Vec<Ty<'tcx>> =
                        v.fields.iter().map(|&t| subst_ty(tcx, t, &subst)).collect();
                    let name = resolver.resolve(v.name);
                    let symbol = mangler.mangle_variant(record.def, name, args);
                    mir::VariantDef {
                        name: mir::Symbol(symbols.get_or_intern(name)),
                        symbol: mir::Symbol(symbols.get_or_intern(&symbol)),
                        fields: tcx.intern_tys(&fields),
                    }
                })
                .collect();
            mir::RecordLayout::Variant(tcx.alloc_slice(&vdefs))
        }
        // Opaque instances are resolved by the caller (they need the FFI
        // rendering context); a record that failed field population
        // elaborates with no fields. Both get an empty compound so lowering
        // still has a well-formed layout.
        Some(RecordFields::Opaque) | None => mir::RecordLayout::Compound(&[]),
    };
    (record.default_cap, layout)
}

/// The maximum type-argument nesting depth monomorphization will instantiate
/// before treating the chain as non-terminating — in the spirit of rustc's
/// `recursion_limit`. Deep but finite generic nesting stays well under it; the
/// unbounded type growth of polymorphic recursion blows straight past it.
const RECURSION_LIMIT: usize = 128;

/// The structural nesting depth of a ground type: `1` for a scalar, one more
/// than its deepest argument for a constructor. Polymorphic recursion grows this
/// without bound (each `f<T>` → `f<Wrap<T>>` step adds a level), which is what
/// the recursion-limit guard in [`Driver::enqueue`] watches.
fn ty_depth(ty: Ty<'_>) -> usize {
    match *ty.kind() {
        TyKind::Nullable(inner) => 1 + ty_depth(inner),
        TyKind::Cell { elem: inner, .. } => 1 + ty_depth(inner),
        TyKind::Arc(inner) => 1 + ty_depth(inner),
        TyKind::Array { elem, .. } => 1 + ty_depth(elem),
        TyKind::Record { args, .. } => 1 + args.iter().map(|&a| ty_depth(a)).max().unwrap_or(0),
        TyKind::Closure { params, ret } => {
            1 + params
                .iter()
                .copied()
                .chain(std::iter::once(ret))
                .map(ty_depth)
                .max()
                .unwrap_or(0)
        }
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Char
        | TyKind::Unit
        | TyKind::Bottom
        | TyKind::Generic(_)
        | TyKind::Hole(_) => 1,
    }
}

/// The surface spelling of an integer type, for diagnostics.
fn int_ty_name(ty: IntTy) -> String {
    match ty {
        IntTy::Signed(w) => format!("i{w}"),
        IntTy::Unsigned(w) => format!("u{w}"),
    }
}

/// The surface spelling of a float type, for diagnostics.
fn fp_ty_name(ty: FpTy) -> String {
    match ty {
        FpTy::Ieee(w) => format!("f{w}"),
        FpTy::BFloat16 => "bfloat16".to_string(),
        FpTy::Float8 => "float8".to_string(),
    }
}

/// Whether a ground type argument is a regional record. Only regional records
/// carry a `Regional`/`Flex`/`Rigid` capability; value/shared records are
/// `Irrelevant` and scalars carry none.
fn is_regional_arg(ty: Ty<'_>) -> bool {
    matches!(
        ty.flexivity(),
        Some(Flexivity::Regional | Flexivity::Flex | Flexivity::Rigid)
    )
}

/// [`SyncEnv`] over ground mono state: capabilities and members from the
/// elaborated record table, member types grounded with [`subst_ty`]. Every
/// type here is ground, so — unlike the semi-side env — nothing blocks on
/// incomplete fields or generics.
struct MonoSyncEnv<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    record_defs: &'a FxHashMap<DefId, Record<'tcx>>,
    resolver: &'a dyn Resolver<TokenKey>,
}

impl<'tcx> SyncEnv<'tcx> for MonoSyncEnv<'_, 'tcx> {
    fn default_cap(&self, def: DefId) -> Option<DefaultCap> {
        self.record_defs.get(&def).map(|r| r.default_cap)
    }

    fn members(&self, def: DefId, args: &'tcx [Ty<'tcx>]) -> Option<Vec<(String, Ty<'tcx>)>> {
        let rec = self.record_defs.get(&def)?;
        let fields = rec.fields.as_ref()?;
        let subst: Subst<'tcx> = rec
            .ty_params
            .iter()
            .map(|(_, g)| *g)
            .zip(args.iter().copied())
            .collect();
        let ground = |t: Ty<'tcx>| subst_ty(self.tcx, t, &subst);
        Some(match fields {
            RecordFields::Named(fs) => fs
                .iter()
                .map(|(n, t, _)| (self.resolver.resolve(*n).to_owned(), ground(*t)))
                .collect(),
            RecordFields::Unnamed(fs) => fs
                .iter()
                .enumerate()
                .map(|(i, (t, _))| (i.to_string(), ground(*t)))
                .collect(),
            RecordFields::Variants(vs) => vs
                .iter()
                .flat_map(|v| {
                    let vname = self.resolver.resolve(v.name);
                    v.fields
                        .iter()
                        .enumerate()
                        .map(move |(i, t)| (format!("{vname}.{i}"), *t))
                })
                .map(|(label, t)| (label, ground(t)))
                .collect(),
            // The foreign payload is invisible; `foreign` refutes before
            // members are ever consulted.
            RecordFields::Opaque => Vec::new(),
        })
    }

    fn member_record_defs(&self, def: DefId) -> Option<Vec<DefId>> {
        let rec = self.record_defs.get(&def)?;
        let fields = rec.fields.as_ref()?;
        let mut out = Vec::new();
        let mut walk = |t: Ty<'tcx>| crate::semi::traits::sync::collect_record_defs(t, &mut out);
        match fields {
            RecordFields::Named(fs) => fs.iter().for_each(|(_, t, _)| walk(*t)),
            RecordFields::Unnamed(fs) => fs.iter().for_each(|(t, _)| walk(*t)),
            RecordFields::Variants(vs) => vs
                .iter()
                .for_each(|v| v.fields.iter().for_each(|t| walk(*t))),
            RecordFields::Opaque => {}
        }
        Some(out)
    }

    fn foreign(&self, def: DefId) -> bool {
        self.record_defs.get(&def).is_some_and(|r| r.ffi.is_some())
    }
}

/// Mutable worklist + lowering state.
struct Driver<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    /// Symbol resolution for diagnostics (member labels in `Sync` witnesses).
    resolver: &'a dyn Resolver<TokenKey>,
    mangler: Mangler<'a>,
    /// Interner for every mangled symbol; moved into the final `mir::Program`.
    symbols: Rodeo,
    queue: VecDeque<Instance<'tcx>>,
    /// Function instances already enqueued (so each is emitted once).
    seen: FxHashSet<Instance<'tcx>>,
    /// Where each instance was first demanded — the discovering call's file
    /// and span — so an instantiation target with no recorded body or
    /// prototype reports at its call site.
    origins: FxHashMap<Instance<'tcx>, (FileId, Option<Span>)>,
    /// Ground record instances whose layout is needed (keyed flex-independently).
    records: FxHashSet<(DefId, &'tcx [Ty<'tcx>])>,
    /// Record definitions (for their default capability) — the deferred `Arc`
    /// inner check needs to tell a `[shared]` record from the rest.
    record_defs: &'a FxHashMap<DefId, Record<'tcx>>,
    /// `Arc` types already reported by the deferred inner check; cleared per
    /// function instance so an offending instantiation reports once, not once
    /// per mention in the body.
    reported_arcs: FxHashSet<Ty<'tcx>>,
    reports: Vec<Report>,
    /// The declaration file of the item currently being instantiated; stamped
    /// onto reports so multi-file diagnostics point into the right source.
    cur_file: FileId,
    /// Source of fresh [`mir::ExprId`] anchors for every lowered node. A single
    /// counter across the whole program: one semi expr may lower into many MIR
    /// exprs (once per instantiation), so semi ids are not reused.
    ids: mir::ExprIdGen,
}

impl<'a, 'tcx> Driver<'a, 'tcx> {
    /// Intern a mangled instance symbol (deduping so a callee and its definition
    /// share one [`mir::Symbol`]).
    fn symbol_of(&mut self, def: DefId, ty_args: &'tcx [Ty<'tcx>]) -> mir::Symbol {
        mir::Symbol(
            self.symbols
                .get_or_intern(self.mangler.mangle_instance(def, ty_args)),
        )
    }

    /// Record a ground record instance (for layout) and return its symbol.
    fn record_symbol(&mut self, def: DefId, args: &'tcx [Ty<'tcx>]) -> mir::Symbol {
        self.records.insert((def, args));
        self.symbol_of(def, args)
    }

    /// Push an `Error` diagnostic.
    fn error(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Error,
            message: message.into(),
            span,
            file: self.cur_file,
        });
    }

    /// Enqueue a function instance if it has not been seen, recording the
    /// demanding site (`span` in [`cur_file`](Self::cur_file)) for the
    /// missing-definition diagnostic.
    fn enqueue(&mut self, def: DefId, ty_args: &'tcx [Ty<'tcx>], span: Option<Span>) {
        let inst = Instance { def, ty_args };
        if self.seen.insert(inst) {
            self.origins.insert(inst, (self.cur_file, span));
            // Bound the queue by type-argument depth. The queue is the only thing
            // that grows, and polymorphic recursion makes each instance one level
            // deeper than the last, so an unbounded chain trips this rather than
            // looping forever. Ground types of depth <= the limit over a finite
            // set of `DefId`s are a finite set, so this guarantees termination.
            let depth = ty_args.iter().map(|&t| ty_depth(t)).max().unwrap_or(0);
            assert!(
                depth <= RECURSION_LIMIT,
                "monomorphize: type-argument nesting depth {depth} exceeds the \
                 recursion limit ({RECURSION_LIMIT}); this is almost certainly \
                 unbounded instantiation from polymorphic recursion"
            );
            self.queue.push_back(inst);
        }
    }

    /// Collect every record instance reachable from a ground type.
    fn note_records(&mut self, ty: Ty<'tcx>) {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => {
                self.records.insert((def, args));
                for &arg in args {
                    self.note_records(arg);
                }
            }
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.note_records(p);
                }
                self.note_records(ret);
            }
            TyKind::Nullable(inner) => self.note_records(inner),
            TyKind::Cell { elem: inner, .. } => self.note_records(inner),
            TyKind::Arc(inner) => self.note_records(inner),
            TyKind::Array { elem, .. } => self.note_records(elem),
            _ => {}
        }
    }

    /// The deferred half of the `Arc` inner check (see the module docs):
    /// `ty_eval` lets a generic inner (`Arc<T>`) through, so a ground
    /// instantiation is the first point where an `Arc` of a non-box —
    /// `Arc<i32>`, `Arc<Cell<…>>` — can exist. Walk a substituted type and
    /// report every such `Arc`, once per type per function instance (see
    /// [`reported_arcs`](Self::reported_arcs)).
    fn check_arc_inners(&mut self, ty: Ty<'tcx>, span: Option<Span>) {
        match *ty.kind() {
            TyKind::Arc(inner) => {
                let rejection = arc_inner_rejection(
                    |def| self.record_defs.get(&def).map(|r| r.default_cap),
                    inner,
                );
                if let Some(kind) = rejection {
                    if self.reported_arcs.insert(ty) {
                        self.error(
                            span,
                            format!(
                                "this instantiation grounds an `Arc` inner type that is not a \
                                 `[shared]` record, array, or closure ({kind})"
                            ),
                        );
                    }
                } else {
                    // The kind is admissible; the ground backstop for the
                    // structural half: every member must be `Sync`. Semi
                    // checks concrete annotations/ctors eagerly but defers
                    // anything a generic blocks — instantiation is where
                    // those ground.
                    let env = MonoSyncEnv {
                        tcx: self.tcx,
                        record_defs: self.record_defs,
                        resolver: self.resolver,
                    };
                    if let SyncVerdict::NotSync(w) = wf_arc(&env, inner)
                        && self.reported_arcs.insert(ty)
                    {
                        self.error(
                            span,
                            format!(
                                "this instantiation grounds an `Arc` whose contents are not \
                                 `Sync`: {}",
                                w.describe()
                            ),
                        );
                    }
                }
                self.check_arc_inners(inner, span);
            }
            TyKind::Record { args, .. } => {
                for &arg in args {
                    self.check_arc_inners(arg, span);
                }
            }
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.check_arc_inners(p, span);
                }
                self.check_arc_inners(ret, span);
            }
            TyKind::Nullable(inner) => self.check_arc_inners(inner, span),
            TyKind::Cell { elem: inner, kind } => {
                // The ground half of the cell element bounds (see semi's
                // `report_cell_wf`): a generic element defers to this point.
                use crate::semi::ty::CellKind;
                use crate::semi::ty_eval::{
                    atomic_cell_element_rejection, lock_cell_slot_rejection,
                };
                match kind {
                    CellKind::Plain | CellKind::Exclusive => {}
                    CellKind::Atomic => {
                        if let Some(what) = atomic_cell_element_rejection(inner)
                            && self.reported_arcs.insert(ty)
                        {
                            self.error(
                                span,
                                format!(
                                    "this instantiation grounds an `Atomic` cell \
                                     whose element is {what}"
                                ),
                            );
                        }
                    }
                    CellKind::Mutex | CellKind::Flatlock | CellKind::Rwlock => {
                        let slot = lock_cell_slot_rejection(
                            |def| self.record_defs.get(&def).map(|r| r.default_cap),
                            inner,
                        );
                        if let Some(what) = slot {
                            if self.reported_arcs.insert(ty) {
                                self.error(
                                    span,
                                    format!(
                                        "this instantiation grounds a `{}` cell \
                                         whose element is {what}",
                                        kind.surface_name()
                                    ),
                                );
                            }
                        } else {
                            let env = MonoSyncEnv {
                                tcx: self.tcx,
                                record_defs: self.record_defs,
                                resolver: self.resolver,
                            };
                            if let SyncVerdict::NotSync(w) =
                                crate::semi::traits::sync::sync_verdict(&env, inner)
                                && self.reported_arcs.insert(ty)
                            {
                                self.error(
                                    span,
                                    format!(
                                        "this instantiation grounds a `{}` cell whose \
                                         element is not `Sync`: {}",
                                        kind.surface_name(),
                                        w.describe()
                                    ),
                                );
                            }
                        }
                    }
                }
                self.check_arc_inners(inner, span);
            }
            TyKind::Array { elem, .. } => self.check_arc_inners(elem, span),
            _ => {}
        }
    }

    /// Collect the records reachable from a ground type, pushing each *newly*
    /// discovered instance onto `worklist` (and into [`records`](Self::records)).
    /// Used to close the record set over fields; bounds nesting depth like
    /// [`enqueue`](Self::enqueue) so polymorphic recursion through a field trips
    /// the limit rather than looping forever.
    fn discover_records(&mut self, ty: Ty<'tcx>, worklist: &mut Vec<(DefId, &'tcx [Ty<'tcx>])>) {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => {
                if self.records.insert((def, args)) {
                    let depth = args.iter().map(|&t| ty_depth(t)).max().unwrap_or(0);
                    assert!(
                        depth <= RECURSION_LIMIT,
                        "monomorphize: record field nesting depth {depth} exceeds the \
                         recursion limit ({RECURSION_LIMIT}); this is almost certainly \
                         unbounded instantiation from polymorphic recursion through a field"
                    );
                    worklist.push((def, args));
                }
                for &arg in args {
                    self.discover_records(arg, worklist);
                }
            }
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.discover_records(p, worklist);
                }
                self.discover_records(ret, worklist);
            }
            TyKind::Nullable(inner) => self.discover_records(inner, worklist),
            TyKind::Cell { elem: inner, .. } => self.discover_records(inner, worklist),
            TyKind::Arc(inner) => self.discover_records(inner, worklist),
            TyKind::Array { elem, .. } => self.discover_records(elem, worklist),
            _ => {}
        }
    }

    /// Close [`records`](Self::records) over record fields: a record reachable
    /// only as another record's field (never in a signature) still needs its
    /// layout. Each instance's field types are substituted under its type
    /// arguments — mirroring [`resolve_layout`] — and the records they mention are
    /// added, to a fixed point.
    fn close_records_over_fields(
        &mut self,
        records: &FxHashMap<DefId, Record<'tcx>>,
        tcx: &TyCtxt<'tcx>,
    ) {
        let mut worklist: Vec<(DefId, &'tcx [Ty<'tcx>])> = self.records.iter().copied().collect();
        while let Some((def, args)) = worklist.pop() {
            let Some(record) = records.get(&def) else {
                continue;
            };
            let mut subst = Subst::default();
            for ((_, gid), &ty) in record.ty_params.iter().zip(args.iter()) {
                subst.insert(*gid, ty);
            }
            let field_tys: Vec<Ty<'tcx>> = match record.fields.as_ref() {
                Some(RecordFields::Named(fields)) => fields.iter().map(|(_, ty, _)| *ty).collect(),
                Some(RecordFields::Unnamed(fields)) => fields.iter().map(|(ty, _)| *ty).collect(),
                Some(RecordFields::Variants(variants)) => variants
                    .iter()
                    .flat_map(|v| v.fields.iter().copied())
                    .collect(),
                Some(RecordFields::Opaque) | None => Vec::new(),
            };
            for field_ty in field_tys {
                let ground = subst_ty(tcx, field_ty, &subst);
                self.discover_records(ground, &mut worklist);
                // The member position is where an `Arc` can ground without
                // ever appearing in a signature or expression type: a direct
                // `Arc` slot is rejected at declaration, but one behind a
                // `Nullable`/`Cell` wrapper (`next: Nullable<Arc<T>>`) is
                // legal and semi's wf check blocks on the generic. This walk
                // is its only ground checkpoint.
                self.check_arc_inners(ground, record.span);
            }
        }
    }

    /// The instances stored *inline* in instance `(def, args)`'s layout:
    /// ground member heads that are `[value]`-capability records. Pointers
    /// (shared/regional records, `Arc`, `Nullable`, cells, arrays, closures)
    /// break an inline chain.
    fn inline_value_instances(
        &self,
        tcx: &'a TyCtxt<'tcx>,
        def: DefId,
        args: &'tcx [Ty<'tcx>],
    ) -> Vec<(DefId, &'tcx [Ty<'tcx>])> {
        let Some(record) = self.record_defs.get(&def) else {
            return Vec::new();
        };
        let mut subst = Subst::default();
        for ((_, gid), &ty) in record.ty_params.iter().zip(args.iter()) {
            subst.insert(*gid, ty);
        }
        let field_tys: Vec<Ty<'tcx>> = match record.fields.as_ref() {
            Some(RecordFields::Named(fields)) => fields.iter().map(|(_, ty, _)| *ty).collect(),
            Some(RecordFields::Unnamed(fields)) => fields.iter().map(|(ty, _)| *ty).collect(),
            Some(RecordFields::Variants(variants)) => variants
                .iter()
                .flat_map(|v| v.fields.iter().copied())
                .collect(),
            Some(RecordFields::Opaque) | None => Vec::new(),
        };
        field_tys
            .into_iter()
            .filter_map(|t| match *subst_ty(tcx, t, &subst).kind() {
                TyKind::Record { def, args, .. }
                    if self
                        .record_defs
                        .get(&def)
                        .is_some_and(|r| r.default_cap == DefaultCap::Value) =>
                {
                    Some((def, args))
                }
                _ => None,
            })
            .collect()
    }

    /// The ground half of the infinite-size check (see semi's
    /// `reject_infinite_value_recursion`): walk the closed record-instance
    /// set and report any `[value]` instance stored inline within itself.
    fn reject_infinite_value_recursion(&mut self, tcx: &'a TyCtxt<'tcx>) {
        type Key<'tcx> = (DefId, &'tcx [Ty<'tcx>]);
        // 1 = on the current DFS path, 2 = finished.
        let mut color: FxHashMap<Key<'tcx>, u8> = FxHashMap::default();
        let roots: Vec<Key<'tcx>> = self.records.iter().copied().collect();
        for root in roots {
            if color.contains_key(&root)
                || !self
                    .record_defs
                    .get(&root.0)
                    .is_some_and(|r| r.default_cap == DefaultCap::Value)
            {
                continue;
            }
            let mut stack: Vec<(Key<'tcx>, Vec<Key<'tcx>>)> =
                vec![(root, self.inline_value_instances(tcx, root.0, root.1))];
            color.insert(root, 1);
            while let Some((_, succs)) = stack.last_mut() {
                let Some(next) = succs.pop() else {
                    let (done, _) = stack.pop().expect("non-empty stack");
                    color.insert(done, 2);
                    continue;
                };
                match color.get(&next) {
                    Some(1) => {
                        let cycle: Vec<String> = stack
                            .iter()
                            .map(|(k, _)| k.0)
                            .skip_while(|&d| d != next.0)
                            .map(|d| self.resolver.resolve(self.record_defs[&d].name).to_owned())
                            .collect();
                        let span = self.record_defs[&next.0].span;
                        self.error(
                            span,
                            format!(
                                "this instantiation creates a recursive `[value]` \
                                 record of infinite size: `{}` is stored inline \
                                 within itself (through `{}`); break the cycle \
                                 with a boxed link",
                                cycle.first().cloned().unwrap_or_default(),
                                cycle.join("` → `"),
                            ),
                        );
                    }
                    Some(_) => {}
                    None => {
                        let succs = self.inline_value_instances(tcx, next.0, next.1);
                        color.insert(next, 1);
                        stack.push((next, succs));
                    }
                }
            }
        }
    }

    /// Ground a slice of type arguments and re-intern it.
    fn ground_args(&self, ty_args: &[Ty<'tcx>], subst: &Subst<'tcx>) -> &'tcx [Ty<'tcx>] {
        let grounded: Vec<Ty<'tcx>> = ty_args
            .iter()
            .map(|&t| subst_ty(self.tcx, t, subst))
            .collect();
        self.tcx.intern_tys(&grounded)
    }

    /// Lower an expression and arena-allocate it, returning a shared reference.
    fn lower_ref(&mut self, e: &Expr<'tcx>, subst: &Subst<'tcx>) -> &'tcx mir::Expr<'tcx> {
        let lowered = self.lower_expr(e, subst);
        self.tcx.alloc(lowered)
    }

    /// Lower a list of expressions into an arena slice.
    fn lower_slice(&mut self, es: &[Expr<'tcx>], subst: &Subst<'tcx>) -> &'tcx [mir::Expr<'tcx>] {
        let lowered: Vec<mir::Expr<'tcx>> = es.iter().map(|e| self.lower_expr(e, subst)).collect();
        self.tcx.alloc_slice(&lowered)
    }

    /// Lower one Semi expression into a ground MIR expression.
    fn lower_expr(&mut self, e: &Expr<'tcx>, subst: &Subst<'tcx>) -> mir::Expr<'tcx> {
        let ty = subst_ty(self.tcx, e.ty, subst);
        self.note_records(ty);
        self.check_arc_inners(ty, e.span);
        let kind = self.lower_kind(&e.kind, subst, e.span);
        self.check_literal_range(&kind, ty, e.span);
        mir::Expr {
            id: self.ids.fresh(),
            kind,
            ty,
            span: e.span,
        }
    }

    /// A numeric literal is arbitrary-precision until here; now that its type
    /// is ground, reject values the type cannot hold. Integers must be in
    /// range; a float that would round to infinity is an error (there is no
    /// literal syntax for `inf`, so it can only be a mistake). Checked on the
    /// *lowered* kind so folded negations (`-128 : i8`) see the signed value.
    fn check_literal_range(
        &mut self,
        kind: &mir::ExprKind<'tcx>,
        ty: Ty<'tcx>,
        span: Option<Span>,
    ) {
        match *kind {
            mir::ExprKind::ConstInt(n) => {
                if let TyKind::Int(int_ty) = *ty.kind()
                    && !literal::int_fits(n, int_ty)
                {
                    self.error(
                        span,
                        format!(
                            "integer literal `{n}` is out of range for `{}`",
                            int_ty_name(int_ty)
                        ),
                    );
                }
            }
            mir::ExprKind::ConstFloat(f) => {
                if let TyKind::Fp(fp) = *ty.kind()
                    && let Some((exp_bits, mant_bits)) = literal::ieee_params(fp)
                    && f.to_ieee_bits(exp_bits, mant_bits).is_err()
                {
                    self.error(
                        span,
                        format!(
                            "float literal `{f}` is out of range for `{}` (it would round to infinity)",
                            fp_ty_name(fp)
                        ),
                    );
                }
            }
            _ => {}
        }
    }

    fn lower_kind(
        &mut self,
        kind: &ExprKind<'tcx>,
        subst: &Subst<'tcx>,
        span: Option<Span>,
    ) -> mir::ExprKind<'tcx> {
        use mir::ExprKind as M;
        match kind {
            ExprKind::GlobalStr(s) => M::GlobalStr(*s),
            ExprKind::ConstChar(c) => M::ConstChar(*c),
            ExprKind::ConstInt(n) => M::ConstInt(n),
            ExprKind::ConstFloat(f) => M::ConstFloat(f),
            ExprKind::ConstBool(b) => M::ConstBool(*b),
            ExprKind::Var(v) => M::Var(*v),
            ExprKind::Poison => M::Poison,
            // Negation of a literal folds into the constant, so `-128 : i8`
            // (and `-9223372036854775808 : i64`) is a single in-range value
            // by the time the range check above sees it. The exception is
            // `-0.0`: an exact decimal has no negative zero (IEEE signed zero
            // is a binary-format artifact), so it stays a runtime negation —
            // folding it would turn `-0.0` into `+0.0` and flip, e.g., the
            // sign of `1.0 / -0.0`.
            ExprKind::Negate(e) => match &e.kind {
                ExprKind::ConstInt(n) => M::ConstInt(self.tcx.alloc(-(*n).clone())),
                ExprKind::ConstFloat(f) if *f.mantissa() != 0 => {
                    M::ConstFloat(self.tcx.alloc(f.neg()))
                }
                _ => M::Negate(self.lower_ref(e, subst)),
            },
            ExprKind::Not(e) => M::Not(self.lower_ref(e, subst)),
            ExprKind::Arith(l, op, r) => {
                M::Arith(self.lower_ref(l, subst), *op, self.lower_ref(r, subst))
            }
            ExprKind::Cmp(l, op, r) => {
                M::Cmp(self.lower_ref(l, subst), *op, self.lower_ref(r, subst))
            }
            ExprKind::Cast(e, t) => {
                let t = subst_ty(self.tcx, *t, subst);
                self.note_records(t);
                M::Cast(self.lower_ref(e, subst), t)
            }
            ExprKind::If(c, t, f) => M::If(
                self.lower_ref(c, subst),
                self.lower_ref(t, subst),
                self.lower_ref(f, subst),
            ),
            ExprKind::RegionRun(e) => M::RegionRun(self.lower_ref(e, subst)),
            ExprKind::Proj(e, idx) => {
                let base = self.lower_ref(e, subst);
                M::Proj(base, self.tcx.alloc_slice(idx))
            }
            ExprKind::Assign(d, i, s) => {
                M::Assign(self.lower_ref(d, subst), *i, self.lower_ref(s, subst))
            }
            ExprKind::Let {
                var, name, value, ..
            } => M::Let {
                var: *var,
                name: *name,
                value: self.lower_ref(value, subst),
            },
            ExprKind::Seq(es) => M::Seq(self.lower_slice(es, subst)),
            ExprKind::FuncCall {
                target,
                ty_args,
                args,
                regional,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                self.enqueue(*target, ty_args, span);
                let callee = self.symbol_of(*target, ty_args);
                M::Call {
                    callee,
                    args: self.lower_slice(args, subst),
                    regional: *regional,
                }
            }
            ExprKind::Intrinsic { op, args } => M::Intrinsic {
                op: *op,
                args: self.lower_slice(args, subst),
            },
            ExprKind::CompoundCall {
                target,
                ty_args,
                args,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                let record = self.record_symbol(*target, ty_args);
                M::Ctor {
                    record,
                    args: self.lower_slice(args, subst),
                }
            }
            ExprKind::VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => {
                let ty_args = self.ground_args(ty_args, subst);
                let record = self.record_symbol(*target, ty_args);
                M::Variant {
                    record,
                    variant: *variant,
                    args: self.lower_slice(args, subst),
                }
            }
            ExprKind::NullableCall(opt) => {
                M::NullableCall(opt.as_ref().map(|e| self.lower_ref(e, subst)))
            }
            ExprKind::ClosureCall { target, args } => M::ClosureCall {
                target: self.lower_ref(target, subst),
                args: self.lower_slice(args, subst),
            },
            ExprKind::Closure(c) => M::Closure(self.lower_closure(c, subst)),
            ExprKind::ArrayOp { op, args } => M::ArrayOp {
                op: *op,
                args: self.lower_slice(args, subst),
            },
            ExprKind::Match(scrut, tree) => {
                M::Match(self.lower_ref(scrut, subst), self.lower_tree(tree, subst))
            }
        }
    }

    fn lower_closure(
        &mut self,
        c: &hir::ClosureExpr<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::ClosureExpr<'tcx> {
        // Flex-escape rules (a flex value may not be captured/returned) are a
        // *flexivity* concern and are enforced at the Semi stage on concrete
        // records; monomorphization only verifies the regional requirement at the
        // call boundary (see the worklist loop).
        let captures = self.lower_var_tys(&c.captures, subst);
        let params = self.lower_var_tys(&c.params, subst);
        let body = self.lower_ref(&c.body, subst);
        mir::ClosureExpr {
            captures,
            params,
            body,
        }
    }

    /// Ground a `(var, type)` list (closure captures/params) into an arena slice.
    fn lower_var_tys(
        &mut self,
        vars: &[(hir::VarId, Ty<'tcx>)],
        subst: &Subst<'tcx>,
    ) -> &'tcx [(hir::VarId, Ty<'tcx>)] {
        let grounded: Vec<(hir::VarId, Ty<'tcx>)> = vars
            .iter()
            .map(|(v, t)| {
                let t = subst_ty(self.tcx, *t, subst);
                self.note_records(t);
                (*v, t)
            })
            .collect();
        self.tcx.alloc_slice(&grounded)
    }

    /// Lower a decision tree and arena-allocate it.
    fn lower_tree_ref(
        &mut self,
        tree: &DecisionTree<'tcx>,
        subst: &Subst<'tcx>,
    ) -> &'tcx mir::DecisionTree<'tcx> {
        let lowered = self.lower_tree(tree, subst);
        self.tcx.alloc(lowered)
    }

    /// Copy pattern bindings into the arena (each scrutinee path is its own slice).
    fn lower_bindings(
        &self,
        bindings: &[(hir::VarId, hir::PatVarRef)],
    ) -> &'tcx [mir::Binding<'tcx>] {
        let copied: Vec<mir::Binding<'tcx>> = bindings
            .iter()
            .map(|(var, path)| (*var, self.tcx.alloc_slice(&path.0)))
            .collect();
        self.tcx.alloc_slice(&copied)
    }

    fn lower_tree(
        &mut self,
        tree: &DecisionTree<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::DecisionTree<'tcx> {
        use mir::DecisionTree as M;
        match tree {
            DecisionTree::Uncovered => M::Uncovered,
            DecisionTree::Unreachable => M::Unreachable,
            DecisionTree::Leaf { body, bindings } => M::Leaf {
                body: self.lower_ref(body, subst),
                bindings: self.lower_bindings(bindings),
            },
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => {
                let bindings = self.lower_bindings(bindings);
                M::Guard {
                    bindings,
                    guard: self.lower_ref(guard, subst),
                    success: self.lower_tree_ref(success, subst),
                    failure: self.lower_tree_ref(failure, subst),
                }
            }
            DecisionTree::Switch { scrutinee, cases } => {
                let scrutinee = self.tcx.alloc_slice(&scrutinee.0);
                M::Switch {
                    scrutinee,
                    cases: self.lower_cases(cases, subst),
                }
            }
        }
    }

    /// Lower a list of `(key, sub-tree)` arms into an arena slice.
    fn lower_keyed<K: Copy>(
        &mut self,
        arms: &[(K, DecisionTree<'tcx>)],
        subst: &Subst<'tcx>,
    ) -> &'tcx [(K, mir::DecisionTree<'tcx>)] {
        let lowered: Vec<(K, mir::DecisionTree<'tcx>)> = arms
            .iter()
            .map(|(k, t)| (*k, self.lower_tree(t, subst)))
            .collect();
        self.tcx.alloc_slice(&lowered)
    }

    fn lower_cases(
        &mut self,
        cases: &SwitchCases<'tcx>,
        subst: &Subst<'tcx>,
    ) -> mir::SwitchCases<'tcx> {
        use mir::SwitchCases as M;
        match cases {
            SwitchCases::Int { cases, default } => M::Int {
                cases: self.lower_keyed(cases, subst),
                default: self.lower_tree_ref(default, subst),
            },
            SwitchCases::Bool { if_true, if_false } => M::Bool {
                if_true: self.lower_tree_ref(if_true, subst),
                if_false: self.lower_tree_ref(if_false, subst),
            },
            SwitchCases::Char { cases, default } => {
                let lowered = self.lower_keyed(cases, subst);
                M::Char {
                    cases: self.tcx.alloc_slice(&lowered),
                    default: self.lower_tree_ref(default, subst),
                }
            }
            SwitchCases::Ctor(arms) => {
                let lowered: Vec<mir::DecisionTree<'tcx>> =
                    arms.iter().map(|t| self.lower_tree(t, subst)).collect();
                M::Ctor(self.tcx.alloc_slice(&lowered))
            }
            SwitchCases::String { cases, default } => M::String {
                cases: self.lower_keyed(cases, subst),
                default: self.lower_tree_ref(default, subst),
            },
            SwitchCases::Nullable { non_null, null } => M::Nullable {
                non_null: self.lower_tree_ref(non_null, subst),
                null: self.lower_tree_ref(null, subst),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// Elaborate `source`, monomorphize it (asserting no diagnostics), and hand
    /// the program to `f`.
    fn with_full<R>(source: &str, f: impl FnOnce(&mir::Program<'_>) -> R) -> R {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(
                reports.is_empty(),
                "unexpected mono diagnostics: {reports:#?}"
            );
            f(&full)
        })
    }

    fn symbols(full: &mir::Program<'_>) -> Vec<String> {
        full.functions
            .iter()
            .map(|f| full.symbol(f.symbol).to_string())
            .collect()
    }

    /// Elaborate (asserting success) and monomorphize, returning the mono
    /// diagnostics' messages.
    fn mono_reports(source: &str) -> Vec<String> {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (_, reports) = monomorphize(&elab.mono_input());
            reports.into_iter().map(|r| r.message).collect()
        })
    }

    /// The mono-export set: private ground functions reachable from `pub`
    /// generic bodies — transitively through private generics — and nothing
    /// else. `pub` ground bodies do not seed (their code never leaves this
    /// package), and unreachable privates stay unexported.
    #[test]
    fn mono_exports_reach_through_generic_bodies() {
        let source = "
            fn deep(x: i64) -> i64 { x + 1 }
            fn mid<T : Num>(x: T) -> i64 { deep(2) }
            fn from_ground(x: i64) -> i64 { x + 3 }
            fn unreachable_helper(x: i64) -> i64 { x + 4 }
            pub fn api<T : Num>(x: T) -> i64 { mid(x) }
            pub fn ground(x: i64) -> i64 { from_ground(x) }
        ";
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let exported: Vec<&str> = full
                .functions
                .iter()
                .filter(|f| f.mono_exported)
                .map(|f| full.symbol(f.symbol))
                .collect();
            // `deep` crosses the boundary: `api<T>` → `mid<T>` → `deep`.
            // `from_ground` does not (`pub` ground bodies stay home), and the
            // unreachable helper stays plain private.
            assert_eq!(exported, ["_RC4deep"], "{:?}", symbols(&full));
        });
    }

    /// Literals are arbitrary-precision end to end: extremes that the old
    /// `i64`-pinned parse panicked on (a full-range `u64`, `i64::MIN`) now
    /// survive, and folded negation makes `-128 : i8` a single in-range value.
    #[test]
    fn extreme_literals_fit_their_types() {
        with_full(
            "pub fn big() -> u64 { 18446744073709551615 } \
             pub fn min() -> i64 { -9223372036854775808 } \
             pub fn neg() -> i8 { -128 } \
             pub fn hex() -> u32 { 0xFFFF_FFFF } \
             pub fn bin() -> u8 { 0b1111_1111 }",
            |_| (),
        );
    }

    /// `-0.0` must stay a *runtime* negation: `FloatLit` is an exact decimal
    /// with no signed zero, so folding it would produce `+0.0` and flip, e.g.,
    /// the sign of `1.0 / -0.0`. Non-zero literals do fold.
    #[test]
    fn negative_zero_is_not_folded() {
        use crate::full::mir::print::Printer as MirPrinter;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(
                "pub fn nz() -> f64 { -0.0 } \
                 pub fn nn() -> f64 { -1.5 }",
            );
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "{:#?}", elab.reports);
            let (mir, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "{reports:#?}");
            let text = MirPrinter::new(&elab.defs, elab.resolver).program(&mir);
            assert!(
                text.contains("-(0.0 : f64)"),
                "-0.0 must stay a runtime negation:\n{text}"
            );
            assert!(
                text.contains("-1.5 : f64") && !text.contains("-(1.5"),
                "non-zero literals must fold:\n{text}"
            );
        });
    }

    #[test]
    fn out_of_range_literals_are_diagnosed_not_panics() {
        let r = mono_reports("pub fn f() -> i8 { 128 }");
        assert!(
            r.iter().any(|m| m.contains("out of range for `i8`")),
            "{r:?}"
        );
        let r = mono_reports("pub fn f() -> u8 { -1 }");
        assert!(
            r.iter().any(|m| m.contains("out of range for `u8`")),
            "{r:?}"
        );
        let r = mono_reports("pub fn f() -> u64 { 18446744073709551616 }");
        assert!(
            r.iter().any(|m| m.contains("out of range for `u64`")),
            "{r:?}"
        );
        let r = mono_reports("pub fn f() -> f64 { 1e400 }");
        assert!(
            r.iter()
                .any(|m| m.contains("out of range for `f64`") && m.contains("infinity")),
            "{r:?}"
        );
        let r = mono_reports("pub fn f() -> f16 { 65520.0 }");
        assert!(
            r.iter().any(|m| m.contains("out of range for `f16`")),
            "{r:?}"
        );
        // Well inside the range: no diagnostics.
        assert!(mono_reports("pub fn f() -> f16 { 65504.0 }").is_empty());
    }

    /// **Resumability**: serialize the elaborated HIR, parse it back into a
    /// `Parsed` (which carries no `Elaborator`, only functions + records +
    /// trampoline roots), monomorphize *that*, and check it yields exactly the
    /// MIR the original elaboration does. This is what "the parsed HIR can be
    /// monomorphized" means.
    #[test]
    fn parsed_hir_monomorphizes_to_the_same_mir() {
        use crate::full::mir::print::Printer as MirPrinter;
        use crate::semi::hir::build::parse_program;
        use crate::semi::hir::print::Printer as HirPrinter;

        let source = "pub struct Pair { a: i32, b: i32 } \
                      fn id<T>(x: T) -> T { x } \
                      pub fn mk(x: i32, y: i32) -> Pair { id(Pair { a: x, b: y }) }";
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "{:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "{:#?}", elab.reports);

            // Monomorphize the original elaboration.
            let (mir0, r0) = monomorphize(&elab.mono_input());
            assert!(r0.is_empty(), "{r0:#?}");
            let text0 = MirPrinter::new(&elab.defs, elab.resolver).program(&mir0);

            // Serialize the HIR, parse it back, and monomorphize *that*.
            let strings = elab.strings.entries();
            let hir_text = HirPrinter::new(&elab.defs, elab.resolver)
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            let parsed = parse_program(tcx, &hir_text).expect("re-parse HIR");
            let input = MonoInput {
                tcx,
                defs: &parsed.defs,
                resolver: &parsed.names,
                elaborated: &parsed.funcs,
                records: &parsed.records,
                trampolines: &parsed.trampolines,
                transform_anchors: &parsed.transform_anchors,
                transform_scripts: &parsed.transform_scripts,
                ffi_imports: &parsed.ffi_imports,
                ffi_preludes: &parsed.ffi_preludes,
                strings: parsed.strings.clone(),
                externs: MonoExterns::default(),
            };
            let (mir1, r1) = monomorphize(&input);
            assert!(r1.is_empty(), "{r1:#?}");
            let text1 = MirPrinter::new(&parsed.defs, &parsed.names).program(&mir1);

            assert_eq!(
                text0, text1,
                "resumed MIR differs from the original:\n=== original ===\n{text0}\n=== resumed ===\n{text1}"
            );
        });
    }

    /// **End-to-end pipeline over a large program.** A single source covering
    /// value records, projections, a recursive `enum` + `match`, polymorphic
    /// functions, and the full modality machinery (a `[regional]` record with an
    /// in-place `[field]` link, `regional` functions, `[flex]` results, and a
    /// `regional { .. }` region) is driven through every textual stage:
    ///
    ///   parse → semi-elaborate → print HIR → parse HIR
    ///        → resume into full elaboration (monomorphize) → print MIR → parse MIR
    ///
    /// Each textual IR is round-tripped (print → parse → print) to prove it is
    /// stable, and the MIR reached by *resuming a parsed HIR* is checked to be
    /// byte-for-byte identical to the MIR from the original elaboration.
    #[test]
    fn large_program_survives_the_full_pipeline() {
        use crate::full::mir::build::parse_program as parse_mir;
        use crate::full::mir::print::Printer as MirPrinter;
        use crate::semi::hir::build::parse_program as parse_hir;
        use crate::semi::hir::print::Printer as HirPrinter;

        let source = r#"
            pub struct Pair { a: i32, b: i32 }
            pub struct Point { x: i32, y: i32 }

            fn id<T>(x: T) -> T { x }
            fn sum(p: Pair) -> i32 { p.a + p.b }
            fn shift(p: Point, d: i32) -> Point { Point { x: p.x + d, y: p.y + d } }

            enum List<T> { Nil, Cons(T, List<T>) }
            fn head_or<T>(xs: List<T>, d: T) -> T {
                match xs {
                    List::Nil => d,
                    List::Cons(x, rest) => x
                }
            }

            struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }

            regional fn fresh<T>(x: T) -> [flex] TestCell<T> { TestCell { v: x, next: Nullable::Null } }

            regional fn loop_back(seed: i32) -> i32 {
                let c = TestCell { v: seed, next: Nullable::Null };
                c->next := Nullable::NonNull{c};
                c.v
            }

            pub fn mk(x: i32, y: i32) -> Pair { id(Pair { a: x, b: y }) }

            pub fn run(n: i32) -> i32 {
                let p = mk(n, n);
                let q = shift(Point { x: n, y: n }, 1);
                let s = head_or(List::Cons{n, List::Nil}, 0);
                regional { loop_back(sum(p) + q.x + s) }
            }"#;

        with_tcx(|tcx| {
            // ----- parse + semi-elaborate -----
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );

            // ----- print HIR + round-trip it through the parser -----
            let strings = elab.strings.entries();
            let hir_text = HirPrinter::new(&elab.defs, elab.resolver)
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            let hir = parse_hir(tcx, &hir_text).expect("re-parse HIR");
            let hir_text2 = HirPrinter::new(&hir.defs, &hir.names)
                .with_transform_metadata(&hir.transform_anchors, &hir.transform_scripts)
                .with_ffi_metadata(&hir.ffi_preludes, &hir.ffi_imports)
                .program(&hir.funcs, &hir.strings, &hir.records, &hir.trampolines);
            assert_eq!(
                hir_text, hir_text2,
                "HIR round-trip mismatch\n=== printed ===\n{hir_text}\n=== reparsed ===\n{hir_text2}"
            );

            // ----- resume the *parsed* HIR into full elaboration -----
            let resumed_input = MonoInput {
                tcx,
                defs: &hir.defs,
                resolver: &hir.names,
                elaborated: &hir.funcs,
                records: &hir.records,
                trampolines: &hir.trampolines,
                transform_anchors: &hir.transform_anchors,
                transform_scripts: &hir.transform_scripts,
                ffi_imports: &hir.ffi_imports,
                ffi_preludes: &hir.ffi_preludes,
                strings: hir.strings.clone(),
                externs: MonoExterns::default(),
            };
            let (mir_resumed, r_resumed) = monomorphize(&resumed_input);
            assert!(r_resumed.is_empty(), "resumed mono reports: {r_resumed:#?}");
            let mir_text = MirPrinter::new(&hir.defs, &hir.names).program(&mir_resumed);

            // The resumed MIR must equal the MIR of the original elaboration.
            let (mir_orig, r_orig) = monomorphize(&elab.mono_input());
            assert!(r_orig.is_empty(), "original mono reports: {r_orig:#?}");
            let mir_text_orig = MirPrinter::new(&elab.defs, elab.resolver).program(&mir_orig);
            assert_eq!(
                mir_text_orig, mir_text,
                "resuming a parsed HIR produced different MIR\n=== from original elab ===\n{mir_text_orig}\n=== from parsed HIR ===\n{mir_text}"
            );

            // ----- print MIR + round-trip it through the parser -----
            let mir = parse_mir(tcx, &mir_text).expect("re-parse MIR");
            let mir_text2 = MirPrinter::new(&mir.defs, &mir.names).program(&mir.program);
            assert_eq!(
                mir_text, mir_text2,
                "MIR round-trip mismatch\n=== printed ===\n{mir_text}\n=== reparsed ===\n{mir_text2}"
            );

            // Sanity: the public roots and the eagerly-emitted non-generic
            // functions all made it into the final MIR.
            let syms: Vec<String> = mir_resumed
                .functions
                .iter()
                .map(|f| mir_resumed.symbol(f.symbol).to_string())
                .collect();
            for root in ["mk", "run", "sum", "shift", "loop_back"] {
                assert!(
                    syms.iter().any(|s| s.contains(root)),
                    "expected `{root}` in the emitted MIR, got: {syms:#?}"
                );
            }
        });
    }

    fn is_ground(ty: Ty<'_>) -> bool {
        match *ty.kind() {
            TyKind::Generic(_) | TyKind::Hole(_) => false,
            TyKind::Record { args, .. } => args.iter().all(|&a| is_ground(a)),
            TyKind::Closure { params, ret } => {
                params.iter().all(|&p| is_ground(p)) && is_ground(ret)
            }
            TyKind::Nullable(inner) => is_ground(inner),
            TyKind::Cell { elem: inner, .. } => is_ground(inner),
            TyKind::Arc(inner) => is_ground(inner),
            TyKind::Array { elem, .. } => is_ground(elem),
            _ => true,
        }
    }

    /// The immediate sub-expressions of a MIR node (decision-tree arms aside).
    fn children<'tcx>(e: &mir::Expr<'tcx>) -> Vec<&'tcx mir::Expr<'tcx>> {
        use mir::ExprKind::*;
        match e.kind {
            ArrayOp { args, .. } => args.iter().collect(),
            GlobalStr(_) | ConstChar(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_)
            | Poison => vec![],
            Negate(x) | Not(x) | Cast(x, _) | RegionRun(x) | Proj(x, _) => vec![x],
            Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => vec![l, r],
            If(c, t, f) => vec![c, t, f],
            Let { value, .. } => vec![value],
            Seq(es) => es.iter().collect(),
            Call { args, .. }
            | Ctor { args, .. }
            | Variant { args, .. }
            | Intrinsic { args, .. } => args.iter().collect(),
            NullableCall(opt) => opt.into_iter().collect(),
            ClosureCall { target, args } => std::iter::once(target).chain(args.iter()).collect(),
            Closure(c) => vec![c.body],
            Match(scrut, _) => vec![scrut],
        }
    }

    fn assert_ground(e: &mir::Expr<'_>) {
        assert!(is_ground(e.ty), "non-ground expr type: {:?}", e.ty);
        for child in children(e) {
            assert_ground(child);
        }
    }

    #[test]
    fn emits_every_non_generic_function_even_if_uncalled() {
        // `never_called` is dead, yet still emitted (DCE is the backend's job).
        let src = r#"
            pub fn used(n: i32) -> i32 { n }
            fn never_called(n: i32) -> i32 { n }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RC4used".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC12never_called".to_string()), "{syms:?}");
        });
    }

    #[test]
    fn instantiates_a_generic_at_each_use() {
        // `id<T>` is emitted once per distinct instantiation; calls are by symbol.
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
            pub fn use_bool(b: bool) -> bool { id(b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC2idlE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC2idbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC7use_i32".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RC8use_bool".to_string()), "{syms:?}");
            let mut sorted = syms.clone();
            sorted.dedup();
            assert_eq!(sorted.len(), syms.len(), "duplicate instances: {syms:?}");
        });
    }

    #[test]
    fn transform_anchor_marks_each_generic_instance() {
        let src = r#"
            #[transform_anchor]
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
            pub fn use_bool(b: bool) -> bool { id(b) }
            transform [{ transform.yield }];
        "#;
        with_full(src, |full| {
            assert_eq!(
                full.functions
                    .iter()
                    .filter(|function| function.transform_anchor)
                    .count(),
                2
            );
            assert_eq!(full.transform_scripts.len(), 1);
            assert_eq!(full.transform_scripts[0].body, "{ transform.yield }");
        });
    }

    #[test]
    fn bodies_are_ground_after_monomorphization() {
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
        "#;
        with_full(src, |full| {
            for func in &full.functions {
                if let Some(body) = func.body {
                    assert_ground(body);
                }
                for p in &func.params {
                    assert!(
                        is_ground(p.ty),
                        "non-ground param in {}",
                        full.symbol(func.symbol)
                    );
                }
            }
        });
    }

    #[test]
    fn calls_are_resolved_to_callee_symbols() {
        // The MIR dispatches by interned symbol: the body of `use_i32` calls the
        // ground `id<i32>` instance directly by its mangled name.
        let src = r#"
            fn id<T>(x: T) -> T { x }
            pub fn use_i32(n: i32) -> i32 { id(n) }
        "#;
        with_full(src, |full| {
            let user = full
                .functions
                .iter()
                .find(|f| full.symbol(f.symbol) == "_RC7use_i32")
                .expect("use_i32 emitted");
            let body = user.body.expect("use_i32 has a body");
            // A one-statement block collapses to the statement, so the body is
            // the `id(n)` call directly.
            let mir::ExprKind::Call { callee, .. } = body.kind else {
                panic!("expected a Call body, got {:?}", body.kind);
            };
            assert_eq!(full.symbol(callee), "_RIC2idlE");
        });
    }

    #[test]
    fn multiple_type_params_instantiate_per_ordering() {
        let src = r#"
            fn pick<A, B>(a: A, b: B) -> A { a }
            fn use_ib(n: i32, b: bool) -> i32 { pick(n, b) }
            fn use_bi(b: bool, n: i32) -> bool { pick(b, n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4picklbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4pickblE".to_string()), "{syms:?}");
            let picks = syms.iter().filter(|s| s.contains("pick")).count();
            assert_eq!(picks, 2, "{syms:?}");
        });
    }

    #[test]
    fn permuting_type_params_terminates() {
        let src = r#"
            fn swap<A, B>(a: A, b: B) -> i32 { swap(b, a) }
            fn main(n: i32, b: bool) -> i32 { swap(n, b) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4swaplbE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4swapblE".to_string()), "{syms:?}");
            let swaps = syms.iter().filter(|s| s.contains("swap")).count();
            assert_eq!(swaps, 2, "{syms:?}");
        });
    }

    #[test]
    fn mutual_recursion_at_fixed_type_terminates() {
        let src = r#"
            fn ping<T>(x: T) -> i32 { pong(x) }
            fn pong<T>(x: T) -> i32 { ping(x) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RIC4pinglE".to_string()), "{syms:?}");
            assert!(syms.contains(&"_RIC4ponglE".to_string()), "{syms:?}");
        });
    }

    #[test]
    fn monomorphic_recursion_terminates() {
        let src = r#"
            fn rec(n: i32) -> i32 { rec(n) }
            fn main(n: i32) -> i32 { rec(n) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.contains(&"_RC3rec".to_string()), "{syms:?}");
        });
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_polymorphic_recursion() {
        let src = r#"
            struct Wrap<T> { value: T }
            fn rec<T>(x: T) -> i32 { rec(Wrap { value: x }) }
            fn main(n: i32) -> i32 { rec(n) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_polymorphic_recursion_in_one_of_several_params() {
        let src = r#"
            struct Wrap<T> { value: T }
            fn rec2<A, B>(a: A, b: B) -> i32 { rec2(Wrap { value: a }, b) }
            fn main(n: i32, b: bool) -> i32 { rec2(n, b) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    #[should_panic(expected = "recursion limit")]
    fn detects_mutual_polymorphic_recursion() {
        let src = r#"
            struct Wrap<T> { value: T }
            fn ping<T>(x: T) -> i32 { pong(Wrap { value: x }) }
            fn pong<T>(x: T) -> i32 { ping(Wrap { value: x }) }
            fn main(n: i32) -> i32 { ping(n) }
        "#;
        with_full(src, |_| {});
    }

    #[test]
    fn flex_generic_accepts_regional_instantiation() {
        // `foo<T>(bar: [flex] T)` requires T to be regional; instantiating it at a
        // regional record is accepted (no diagnostic).
        let src = r#"
            struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }
            regional fn foo<T>(bar: [flex] T) -> i32 { 0 }
            regional fn use_ok(c: [flex] TestCell<i32>) -> i32 { foo(c) }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.iter().any(|s| s.contains("foo")), "{syms:?}");
        });
    }

    #[test]
    fn flex_generic_rejects_non_regional_instantiation() {
        // The same `[flex] T` parameter instantiated at a value record is rejected
        // at the call boundary — only regional records may be flex.
        let src = r#"
            struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }
            struct Pair { a: i32 }
            regional fn foo<T>(bar: [flex] T) -> i32 { 0 }
            regional fn use_bad(p: Pair) -> i32 { foo(p) }
        "#;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (_full, reports) = monomorphize(&elab.mono_input());
            assert!(
                reports
                    .iter()
                    .any(|r| r.message.contains("regional record")),
                "expected a regionality diagnostic, got {reports:#?}"
            );
        });
    }

    #[test]
    fn arc_generic_rejects_non_box_instantiation() {
        // `ty_eval` defers a generic `Arc` inner; a trampoline's explicit type
        // arguments are the first point where it grounds. `Arc<i32>` and
        // `Arc<Cell<i32>>` must be rejected here — `Arc` takes only a shared
        // rc box (a `[shared]` record, an array, or a closure).
        let src = r#"
            fn passthrough<T>(a: Arc<T>) -> Arc<T> { a }
            extern "C" trampoline "bad_scalar" = passthrough<i32>;
            extern "C" trampoline "bad_cell" = passthrough<Cell<i32>>;
        "#;
        let reports = mono_reports(src);
        assert!(
            reports.iter().any(|m| m.contains(
                "grounds an `Arc` inner type that is not a `[shared]` record, \
                 array, or closure (not an rc box)"
            )),
            "{reports:#?}"
        );
        assert!(
            reports.iter().any(|m| m.contains(
                "grounds an `Arc` inner type that is not a `[shared]` record, \
                 array, or closure (a cell)"
            )),
            "{reports:#?}"
        );
    }

    #[test]
    fn arc_generic_rejects_non_sync_instantiation() {
        // The kind is admissible (a `[shared]` record) but the contents are
        // not `Sync`: `S` holds a bare shared `Q` *outside its recursive
        // group*, so §3.3 promotion does not apply and the plain box
        // refutes. Semi defers the generic inner; grounding it here is the
        // backstop. A recursive instantiation, by contrast, promotes and is
        // accepted.
        let src = r#"
            struct Q { x: i64 }
            struct S { p: Q }
            enum List<T> { Nil, Cons(T, List<T>) }
            fn passthrough<T>(a: Arc<T>) -> Arc<T> { a }
            extern "C" trampoline "bad_s" = passthrough<S>;
            extern "C" trampoline "ok_list" = passthrough<List<i32>>;
        "#;
        let reports = mono_reports(src);
        assert!(
            reports.iter().any(
                |m| m.contains("grounds an `Arc` whose contents are not `Sync`")
                    && m.contains("member `p`")
            ),
            "{reports:#?}"
        );
        assert!(
            !reports.iter().any(|m| m.contains("Cons")),
            "the recursive instantiation must promote, not refute: {reports:#?}"
        );
    }

    #[test]
    fn arc_behind_a_generic_member_is_checked_at_instantiation() {
        // A member-position `Arc` can ground without ever appearing in a
        // signature or expression type: `link: Nullable<Arc<T>>` is a legal
        // member (only a *direct* `Arc` slot is rejected at declaration) and
        // semi's wf check blocks on the generic — closing the record set
        // over fields is its only ground checkpoint, for both halves of the
        // check.
        let src = r#"
            struct Q { x: i64 }
            struct S { p: Q }
            struct Holder<T> { link: Nullable<Arc<T>> }
            fn keep<T>(h: Holder<T>) -> Holder<T> { h }
            extern "C" trampoline "unsync_member" = keep<S>;
            extern "C" trampoline "non_box_member" = keep<i32>;
        "#;
        let reports = mono_reports(src);
        assert!(
            reports.iter().any(
                |m| m.contains("grounds an `Arc` whose contents are not `Sync`")
                    && m.contains("member `p`")
            ),
            "{reports:#?}"
        );
        assert!(
            reports
                .iter()
                .any(|m| m.contains("not a `[shared]` record, array, or closure")),
            "{reports:#?}"
        );
    }

    #[test]
    fn ground_value_recursion_is_rejected_at_instantiation() {
        // The def-level check cannot see a cycle closed through a generic:
        // `Wrap<T>` stores `T` inline, and instantiating it at a record that
        // contains `Wrap<A>` back closes the loop only once ground.
        let src = r#"
            struct [value] Wrap<T> { x: T }
            struct [value] A { w: Wrap<A> }
            fn f(a: A) -> i64 { 0 }
            extern "C" trampoline "f_ffi" = f;
        "#;
        let reports = mono_reports(src);
        assert!(
            reports
                .iter()
                .any(|m| m.contains("recursive `[value]` record of infinite size")),
            "{reports:#?}"
        );
    }

    #[test]
    fn ground_cell_element_bounds_are_rechecked() {
        // Semi defers a generic cell element; grounding is where the bounds
        // land: a bare shared record fails the lock kinds' Sync bound, a
        // record fails Atomic's primitive bound, while Sync instantiations
        // pass.
        let src = r#"
            struct Pair { a: i32 }
            fn hold<T>(m: Mutex<T>) -> Mutex<T> { m }
            fn count<T>(a: Atomic<T>) -> Atomic<T> { a }
            extern "C" trampoline "bad_mutex" = hold<Pair>;
            extern "C" trampoline "ok_mutex" = hold<i64>;
            extern "C" trampoline "bad_atomic" = count<Pair>;
            extern "C" trampoline "ok_atomic" = count<i64>;
        "#;
        let reports = mono_reports(src);
        assert!(
            reports
                .iter()
                .any(|m| m.contains("grounds a `Mutex` cell whose element is not `Sync`")),
            "{reports:#?}"
        );
        assert!(
            reports
                .iter()
                .any(|m| m.contains("grounds an `Atomic` cell whose element is")),
            "{reports:#?}"
        );
        assert_eq!(reports.len(), 2, "{reports:#?}");
    }

    #[test]
    fn arc_generic_accepts_shared_box_instantiations() {
        // A `[shared]` record, an array, and a closure are all shared rc
        // boxes, so grounding `Arc<T>` at them is fine.
        let src = r#"
            struct Data { value: i64 }
            fn passthrough<T>(a: Arc<T>) -> Arc<T> { a }
            extern "C" trampoline "ok_record" = passthrough<Data>;
            extern "C" trampoline "ok_array" = passthrough<[f64; 8]>;
            extern "C" trampoline "ok_closure" = passthrough<(i64) -> i64>;
        "#;
        let reports = mono_reports(src);
        assert!(reports.is_empty(), "{reports:#?}");
    }

    #[test]
    fn field_link_generic_accepts_regional() {
        // `Wrapper<T>` with `[field] T` instantiated at a regional record is fine.
        let src = r#"
            struct [regional] TestCell<T> { v: T, next: [field] TestCell<T> }
            struct [regional] Wrapper<T> { inner: [field] T }
            regional fn use_ok(w: [flex] Wrapper<TestCell<i32>>) -> i32 { 0 }
        "#;
        with_full(src, |full| {
            let syms = symbols(full);
            assert!(syms.iter().any(|s| s.contains("use_ok")), "{syms:?}");
        });
    }

    #[test]
    fn field_link_generic_must_be_regional() {
        // `struct [regional] Wrapper<T> { inner: [field] T }` requires `T` to be
        // regional. Instantiating `Wrapper` at a value record is rejected at the
        // call boundary, even with no function-body assignment.
        let src = r#"
            struct Pair { a: i32 }
            struct [regional] Wrapper<T> { inner: [field] T }
            regional fn use_bad(w: [flex] Wrapper<Pair>) -> i32 { 0 }
        "#;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (_full, reports) = monomorphize(&elab.mono_input());
            assert!(
                reports.iter().any(|r| r.message.contains("`[field]` link")),
                "expected a `[field]` regionality diagnostic, got {reports:#?}"
            );
        });
    }

    // ----- cross-package monomorphization -----

    /// Elaborate `dep_src` as package `dep`, reduce it to its export closure,
    /// print the interface, re-parse, and declare it into a consumer
    /// elaborating `app_src` as package `app` — the round trip `--extern`
    /// performs — then monomorphize the consumer. `edit` mutates the printed
    /// interface text in between (identity for the well-formed cases).
    fn mono_with_dep(
        dep_src: &str,
        app_src: &str,
        edit: impl Fn(String) -> String,
        f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>, &mir::Program<'tcx>, &[Report]),
    ) {
        use std::sync::Arc;

        use reussir_syntax::Interner as _;
        use reussir_syntax::source::FileId;

        use crate::full::interface::{RRI_FORMAT, export_closure};
        use crate::semi::externs::ExternPackage;
        use crate::semi::hir::print::{InterfaceEmit, Printer as HirPrinter};
        use crate::semi::{PackageFile, elaborate_package};

        with_tcx(|tcx| {
            let dep_interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut dep_keys = dep_interner.clone();
            let dep = dep_keys.get_or_intern("dep");
            let parse = reussir_syntax::parse_with_interner(dep_src, dep_interner.clone());
            assert!(parse.ok(), "dep parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let files = [PackageFile {
                file: FileId::ROOT,
                module: vec![dep],
                program: &prog,
            }];
            let dep_elab = elaborate_package(tcx, &files, &dep_interner);
            assert!(
                !dep_elab.has_errors(),
                "dep elab errors: {:#?}",
                dep_elab.reports
            );
            let closure = export_closure(&dep_elab.mono_input());
            let strings: Vec<_> = dep_elab
                .strings
                .entries()
                .into_iter()
                .filter(|(token, _)| closure.strings.contains(token))
                .collect();
            let text = HirPrinter::new(&dep_elab.defs, dep_elab.resolver)
                .with_interface(InterfaceEmit {
                    format: RRI_FORMAT,
                    package: "dep",
                    producer: "t",
                    bodies: &closure.bodies,
                    protos: &closure.protos,
                    records: &closure.records,
                    file_root: None,
                })
                .program(&dep_elab.elaborated, &strings, &dep_elab.records, &[]);
            let text = edit(text);
            let parsed = crate::semi::hir::build::parse_program(tcx, &text)
                .expect("interface re-parses");

            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let app = keys.get_or_intern("app");
            let parse = reussir_syntax::parse_with_interner(app_src, interner.clone());
            assert!(parse.ok(), "app parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let files = [PackageFile {
                file: FileId::ROOT,
                module: vec![app],
                program: &prog,
            }];
            let mut elab = Elaborator::new(tcx, &interner);
            elab.declare_extern_package(
                &mut keys,
                &ExternPackage {
                    name: "dep",
                    parsed: &parsed,
                    files: &[],
                },
            );
            elab.run_package(&files);
            assert!(!elab.has_errors(), "app elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            f(&elab, &full, &reports);
        });
    }

    /// The printed text of one function item (`fn @sym… {…}` / `fn @sym…;`)
    /// out of a whole-program MIR dump.
    fn mir_item(program_text: &str, symbol: &str) -> String {
        let needle = format!("fn @{symbol}(");
        program_text
            .split("\n\n")
            .find(|item| item.contains(&needle))
            .unwrap_or_else(|| panic!("`{symbol}` not in:\n{program_text}"))
            .to_owned()
    }

    /// An imported generic instantiates in the consumer byte-for-byte like
    /// the same instantiation made inside the dependency: same `_RI` symbol
    /// (path-based v0 mangling), same lowered body, callee resolved to the
    /// dependency's ground helper — which itself lowers to a bodyless
    /// declaration (its definition lives in the dep's artifact).
    #[test]
    fn imported_generic_instantiates_like_the_dep_itself() {
        use crate::full::mir::print::Printer as MirPrinter;

        const DEP: &str = "fn helper(x: i64) -> i64 { x + 1 }\n\
                           pub fn twice<T : Num>(x: T) -> T { helper(0); x + x }";

        // The dependency's own instantiation of `twice<i64>`.
        let dep_side = with_tcx(|tcx| {
            use std::sync::Arc;

            use reussir_syntax::Interner as _;
            use reussir_syntax::source::FileId;

            use crate::semi::{PackageFile, elaborate_package};

            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let dep = keys.get_or_intern("dep");
            let src = format!("{DEP}\npub fn local_use(n: i64) -> i64 {{ twice(n) }}");
            let parse = reussir_syntax::parse_with_interner(&src, interner.clone());
            assert!(parse.ok(), "{:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let files = [PackageFile {
                file: FileId::ROOT,
                module: vec![dep],
                program: &prog,
            }];
            let elab = elaborate_package(tcx, &files, &interner);
            assert!(!elab.has_errors(), "{:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "{reports:#?}");
            let text = MirPrinter::new(&elab.defs, elab.resolver).program(&full);
            mir_item(&text, "_RINvC3dep5twicexE")
        });

        mono_with_dep(
            DEP,
            "pub fn use_it(n: i64) -> i64 { dep::twice(n) }",
            |text| text,
            |elab, full, reports| {
                assert!(reports.is_empty(), "{reports:#?}");
                let text = MirPrinter::new(&elab.defs, elab.resolver).program(full);
                let instance = mir_item(&text, "_RINvC3dep5twicexE");
                assert_eq!(
                    instance, dep_side,
                    "the consumer's instance must match the dep's own"
                );
                assert!(
                    instance.contains('{'),
                    "the instance is a definition:\n{instance}"
                );
                // The private ground callee crossed as a prototype only.
                let helper = mir_item(&text, "_RNvC3dep6helper");
                assert!(helper.trim_end().ends_with(';'), "a declaration:\n{helper}");
                // The consumer's own root calls the instance by symbol.
                let user = mir_item(&text, "_RNvC3app6use_it");
                assert!(user.contains("_RINvC3dep5twicexE"), "{user}");
            },
        );
    }

    /// An imported ground `pub` function lowers to a declaration when called
    /// and is not emitted at all otherwise: imported functions are never mono
    /// roots (the dependency compiled their bodies; a consumer root would
    /// re-emit a definition it does not have).
    #[test]
    fn imported_grounds_declare_when_called_and_never_root() {
        mono_with_dep(
            "pub fn ground(x: i64) -> i64 { x + 1 }\n\
             pub fn unused(x: i64) -> i64 { x + 2 }",
            "pub fn go(n: i64) -> i64 { dep::ground(n) }",
            |text| text,
            |_, full, reports| {
                assert!(reports.is_empty(), "{reports:#?}");
                let called = full
                    .functions
                    .iter()
                    .find(|f| full.symbol(f.symbol) == "_RNvC3dep6ground")
                    .expect("called import declared");
                assert!(called.body.is_none(), "declaration, not definition");
                assert!(
                    !symbols(full).iter().any(|s| s.contains("unused")),
                    "uncalled imports must not seed: {:?}",
                    symbols(full)
                );
            },
        );
    }

    /// Imported items never join the consumer's export surface: the closure
    /// seeds from the consumer's own items only, and no dep-derived MIR
    /// function is `mono_exported`.
    #[test]
    fn imported_functions_never_seed_exports() {
        mono_with_dep(
            "fn helper(x: i64) -> i64 { x }\n\
             pub fn api<T : Num>(x: T) -> T { helper(0); x }",
            "fn local_helper(x: i64) -> i64 { x }\n\
             pub fn wrap<T : Num>(x: T) -> T { local_helper(0); dep::api(x) }",
            |text| text,
            |elab, full, reports| {
                assert!(reports.is_empty(), "{reports:#?}");
                let closure = crate::full::interface::export_closure(&elab.mono_input());
                for def in closure.bodies.iter().chain(&closure.protos) {
                    let path = elab.defs.path(*def).display(elab.resolver);
                    assert!(
                        path.starts_with("app::"),
                        "a consumer does not re-export its dependency: {path}"
                    );
                }
                for f in &full.functions {
                    if full.symbol(f.symbol).contains("dep") {
                        assert!(
                            !f.mono_exported,
                            "{} must not be mono-exported",
                            full.symbol(f.symbol)
                        );
                    }
                }
                // The consumer's own reachable private ground still exports.
                let local = full
                    .functions
                    .iter()
                    .find(|f| full.symbol(f.symbol) == "_RNvC3app12local_helper")
                    .expect("local helper emitted");
                assert!(local.mono_exported);
            },
        );
    }

    /// Strings referenced by imported bodies merge into the program table by
    /// their content-addressed token — shared literals collapse to one entry.
    #[test]
    fn extern_strings_merge_by_token() {
        mono_with_dep(
            "pub fn tagged<T : Num>(x: T) -> i64 { let s = \"from-dep\"; 0 }",
            "pub fn go(n: i64) -> i64 { let t = \"from-dep\"; dep::tagged(n) }",
            |text| text,
            |_, full, reports| {
                assert!(reports.is_empty(), "{reports:#?}");
                let hits = full
                    .string_literals
                    .iter()
                    .filter(|(_, payload)| payload == "from-dep")
                    .count();
                assert_eq!(hits, 1, "{:?}", full.string_literals);
            },
        );
    }

    /// A record reachable only through an imported body still resolves a
    /// ground layout (the `note_records` path reads the shared record table,
    /// which holds imported records).
    #[test]
    fn records_used_only_by_imported_bodies_get_layouts() {
        mono_with_dep(
            "struct Hidden { v: i64 }\n\
             pub fn boxed<T : Num>(x: T) -> i64 { let h = Hidden { v: 1 }; h.v }",
            "pub fn go(n: i64) -> i64 { dep::boxed(n) }",
            |text| text,
            |_, full, reports| {
                assert!(reports.is_empty(), "{reports:#?}");
                let hidden = full
                    .records
                    .iter()
                    .find(|r| full.symbol(r.symbol) == "_RNvC3dep6Hidden")
                    .expect("imported record instance collected");
                let mir::RecordLayout::Compound(members) = hidden.layout else {
                    panic!("compound layout expected, got {:?}", hidden.layout);
                };
                assert_eq!(members.len(), 1, "fields resolved, not defaulted");
            },
        );
    }

    /// An instantiation target that is neither local nor imported is a
    /// reported error, not a silent skip: with externs in play the skip would
    /// manufacture an undefined symbol at link time. Simulated by stripping a
    /// shipped generic out of the interface (a stale/incomplete `.rri`).
    #[test]
    fn unknown_instantiation_target_reports() {
        mono_with_dep(
            "fn helper<T>(x: T) -> T { x }\n\
             pub fn api<T>(x: T) -> T { helper(x) }",
            "pub fn go(n: i64) -> i64 { dep::api(n) }",
            |text| {
                // Drop the `helper` item wholesale (items are separated by
                // blank lines), leaving `api`'s shipped body calling a def
                // the interface no longer carries.
                text.split("\n\n")
                    .filter(|item| !item.starts_with("fn #dep::helper"))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            },
            |_, _, reports| {
                assert!(
                    reports.iter().any(|r| {
                        r.message
                            .contains("no body or prototype recorded for `dep::helper`")
                            && r.message.contains("stale")
                    }),
                    "{reports:#?}"
                );
            },
        );
    }

    #[test]
    fn generic_assigned_into_link_must_be_regional() {
        // Assigning a `Nullable<T>` into a flex link records that `T` must be
        // regional (a body-discovered requirement, not just `[flex] T` params).
        // Instantiating at a non-regional type is rejected at the call boundary.
        let src = r#"
            struct Pair { a: i32 }
            struct [regional] Box<T> { item: [field] T }
            regional fn store<T>(b: [flex] Box<T>, z: Nullable<T>) -> i32 { b->item := z; 0 }
            regional fn use_bad(b: [flex] Box<Pair>, z: Nullable<Pair>) -> i32 { store(b, z) }
        "#;
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let (_full, reports) = monomorphize(&elab.mono_input());
            assert!(
                reports
                    .iter()
                    .any(|r| r.message.contains("regional record")),
                "expected a regionality diagnostic, got {reports:#?}"
            );
        });
    }
}
