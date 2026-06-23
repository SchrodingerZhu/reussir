//! The elaboration context: collected items, global + per-function state, and
//! the scan/collect driver.

use reussir_syntax::kind::{Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::semi::infer::InferCtxt;
use crate::semi::resolve::DefTable;
use crate::semi::traits::builtins::Builtins;
use crate::semi::traits::{TraitDb, TraitId};
use crate::semi::ty::{Capability, DefId, GenericId, Ty, TyCtxt, TyKind};
use crate::surface::{self, Span};
use crate::utils::string::StringUniqifier;

use super::fulfill::FulfillCtxt;
use super::hir::{ExprId, Function, VarId};

/// The capability a record declares by default. (Per-use [`crate::semi::ty::Capability`]
/// flexivity is derived from this during coloring.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefaultCap {
    Value,
    Shared,
    Regional,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

/// A diagnostic.
#[derive(Clone, Debug)]
pub struct Report {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
}

/// Per-generic metadata: its name and the trait bounds declared on it.
#[derive(Clone, Debug)]
pub struct GenericInfo {
    pub name: TokenKey,
    pub bounds: Vec<TraitId>,
}

/// A record's fields, with concrete field types resolved.
#[derive(Clone, Debug)]
pub enum RecordFields<'tcx> {
    /// `(name, type, is_mutable)`.
    Named(Vec<(TokenKey, Ty<'tcx>, bool)>),
    /// `(type, is_mutable)`.
    Unnamed(Vec<(Ty<'tcx>, bool)>),
    Variants(Vec<Variant<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct Variant<'tcx> {
    pub name: TokenKey,
    pub fields: Vec<Ty<'tcx>>,
}

/// A collected record (struct or enum). Fields are populated in a second pass.
#[derive(Clone, Debug)]
pub struct Record<'tcx> {
    pub def: DefId,
    pub name: TokenKey,
    pub ty_params: Vec<(TokenKey, GenericId)>,
    pub kind: surface::RecordKind,
    pub default_cap: DefaultCap,
    pub fields: Option<RecordFields<'tcx>>,
    /// Generics that must be instantiated regional because they appear as the
    /// element of a `[field]` link (e.g. `inner: [field] T`). Checked at the
    /// monomorphization call boundary, like a function's `regional_generics`.
    pub regional_generics: Vec<GenericId>,
    pub span: Option<Span>,
}

/// A collected function prototype (signature only).
#[derive(Clone, Debug)]
pub struct FuncProto<'tcx> {
    pub def: DefId,
    pub name: TokenKey,
    pub visibility: surface::Visibility,
    pub generics: Vec<(TokenKey, GenericId)>,
    /// Generics used at a `[flex]` position (e.g. `bar: [flex] T`). A `[flex]`
    /// use of a bare generic requires the instantiating type to be a regional
    /// record — checked at the monomorphization call boundary. (The `[flex]`
    /// coloring itself is dropped on a generic, exactly as in the reference; this
    /// records the *requirement* it implies.)
    pub regional_generics: Vec<GenericId>,
    /// `(name, colored type)`.
    pub params: Vec<(TokenKey, Ty<'tcx>)>,
    pub return_ty: Ty<'tcx>,
    pub is_regional: bool,
    pub span: Option<Span>,
}

/// Records the generic at the head of a `[flex]`-annotated type (peeling
/// `Nullable`): a `[flex]` use of a bare generic means that generic must be
/// instantiated with a regional record.
fn collect_regional_generic(t: Ty<'_>, out: &mut Vec<GenericId>) {
    match *t.kind() {
        TyKind::Generic(id) => {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        TyKind::Nullable(inner) => collect_regional_generic(inner, out),
        _ => {}
    }
}

/// A resolved `extern "<abi>" trampoline "<name>" = target::func<TyArgs>;`.
///
/// A trampoline exports a stable C-ABI symbol aliasing a concrete (ground)
/// instantiation of an internal function. It is a **monomorphization root**: the
/// future mono worklist must seed `(target, ty_args)` so the aliased function is
/// emitted. Recorded here so Semi output can carry the seed list.
#[derive(Clone, Debug)]
pub struct TrampolineRoot<'tcx> {
    /// The exported C symbol name.
    pub name: String,
    /// The C ABI string (e.g. `"C"`).
    pub abi: String,
    /// The resolved internal target function.
    pub target: DefId,
    /// The concrete (ground) type arguments to instantiate the target at.
    pub ty_args: Vec<Ty<'tcx>>,
}

/// A local variable binding.
#[derive(Clone, Debug)]
pub struct VarDef<'tcx> {
    pub name: TokenKey,
    pub ty: Ty<'tcx>,
    pub span: Option<Span>,
}

/// A lexically-scoped variable environment. `VarId`s are stable for a function;
/// only visibility is scoped.
#[derive(Default)]
pub struct VarEnv<'tcx> {
    defs: Vec<VarDef<'tcx>>,
    scope: Vec<(TokenKey, VarId)>,
}

impl<'tcx> VarEnv<'tcx> {
    pub fn fresh(&mut self, name: TokenKey, ty: Ty<'tcx>, span: Option<Span>) -> VarId {
        let id = VarId(self.defs.len() as u32);
        self.defs.push(VarDef { name, ty, span });
        self.scope.push((name, id));
        id
    }

    pub fn lookup(&self, name: TokenKey) -> Option<(VarId, Ty<'tcx>)> {
        self.scope
            .iter()
            .rev()
            .find(|(n, _)| *n == name)
            .map(|(_, id)| (*id, self.defs[id.0 as usize].ty))
    }

    pub fn def(&self, id: VarId) -> &VarDef<'tcx> {
        &self.defs[id.0 as usize]
    }

    /// A marker for the current scope depth; pass to [`VarEnv::restore`].
    pub fn mark(&self) -> usize {
        self.scope.len()
    }

    /// Drop bindings introduced since `mark` (their `VarId`s stay allocated).
    pub fn restore(&mut self, mark: usize) {
        self.scope.truncate(mark);
    }

    fn reset(&mut self) {
        self.defs.clear();
        self.scope.clear();
    }
}

/// The whole elaborator: global collected state plus per-function working state.
pub struct Elaborator<'a, 'tcx> {
    pub tcx: &'a TyCtxt<'tcx>,
    /// Resolves the surface AST's interned token keys back into source text.
    pub resolver: &'a dyn Resolver<TokenKey>,
    pub traits: TraitDb<'tcx>,
    pub builtins: Builtins,
    pub trait_names: FxHashMap<&'static str, TraitId>,
    /// Resolution registry: item `DefId`s and their fully-qualified paths.
    pub defs: DefTable,
    pub records: FxHashMap<DefId, Record<'tcx>>,
    pub functions: FxHashMap<DefId, FuncProto<'tcx>>,
    /// Resolved extern-trampoline roots (mono seeds). See [`TrampolineRoot`].
    pub trampolines: Vec<TrampolineRoot<'tcx>>,
    pub generics: Vec<GenericInfo>,
    pub strings: StringUniqifier,
    pub elaborated: Vec<Function<'tcx>>,
    pub reports: Vec<Report>,

    // ----- per-function working state (reset by `enter_function`) -----
    pub infer: InferCtxt<'a, 'tcx>,
    pub vars: VarEnv<'tcx>,
    pub generic_names: FxHashMap<TokenKey, GenericId>,
    pub inside_region: bool,
    /// Generics required to be regional, accumulated while checking the current
    /// function body (e.g. a generic assigned into a flex link). Seeded from the
    /// prototype's `[flex]`-position generics and folded into the elaborated
    /// [`Function`]. See [`FuncProto::regional_generics`].
    pub regional_generics: Vec<GenericId>,
    pub fulfill: FulfillCtxt<'tcx>,
    expr_counter: u32,
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'tcx>, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        let mut traits = TraitDb::new();
        let builtins = Builtins::register(&mut traits, tcx);
        let trait_names = [
            ("Num", builtins.num),
            ("Integral", builtins.integral),
            ("FloatingPoint", builtins.floating_point),
            ("PtrLike", builtins.ptr_like),
            ("Send", builtins.send),
            ("Sync", builtins.sync),
        ]
        .into_iter()
        .collect();
        Elaborator {
            tcx,
            resolver,
            defs: DefTable::new(),
            traits,
            builtins,
            trait_names,
            trampolines: Vec::new(),
            records: FxHashMap::default(),
            functions: FxHashMap::default(),
            generics: Vec::new(),
            strings: StringUniqifier::new(),
            elaborated: Vec::new(),
            reports: Vec::new(),
            infer: InferCtxt::new(tcx),
            vars: VarEnv::default(),
            generic_names: FxHashMap::default(),
            inside_region: false,
            regional_generics: Vec::new(),
            fulfill: FulfillCtxt::default(),
            expr_counter: 0,
        }
    }

    // ----- diagnostics -----

    pub fn error(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Error,
            message: message.into(),
            span,
        });
    }

    pub fn warn(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Warning,
            message: message.into(),
            span,
        });
    }

    pub fn has_errors(&self) -> bool {
        self.reports.iter().any(|r| r.severity == Severity::Error)
    }

    /// Resolve a surface token key into its source text. The returned slice is
    /// tied to the parse, not to `self`, so it composes with `&mut self` calls.
    pub fn sym(&self, key: TokenKey) -> &'a str {
        self.resolver.resolve(key)
    }

    // ----- id / generic allocation -----

    pub fn fresh_expr_id(&mut self) -> ExprId {
        let id = ExprId(self.expr_counter);
        self.expr_counter += 1;
        id
    }

    /// Allocate a fresh generic parameter with the given bounds.
    pub fn fresh_generic(&mut self, name: TokenKey, bounds: Vec<TraitId>) -> GenericId {
        let id = GenericId(self.generics.len() as u32);
        self.generics.push(GenericInfo { name, bounds });
        id
    }

    pub fn generic_bounds(&self, generic: GenericId) -> &[TraitId] {
        &self.generics[generic.0 as usize].bounds
    }

    /// Resolve a bound path (by basename) to a built-in trait.
    pub fn resolve_bound(&mut self, path: &surface::Path, span: Option<Span>) -> Option<TraitId> {
        let name = self.sym(path.basename);
        match self.trait_names.get(name) {
            Some(&id) => Some(id),
            None => {
                self.error(span, format!("unknown trait bound `{name}`"));
                None
            }
        }
    }

    fn resolve_bounds(&mut self, bounds: &[surface::Path], span: Option<Span>) -> Vec<TraitId> {
        bounds
            .iter()
            .filter_map(|b| self.resolve_bound(b, span))
            .collect()
    }

    /// Reset per-function state and install the function's generics in scope.
    pub(super) fn enter_function(&mut self, generics: &[(TokenKey, GenericId)]) {
        self.infer = InferCtxt::new(self.tcx);
        self.vars.reset();
        self.generic_names.clear();
        self.inside_region = false;
        self.fulfill = FulfillCtxt::default();
        for (name, id) in generics {
            self.generic_names.insert(*name, *id);
        }
    }

    // ----- driver -----

    /// Run the three collection passes over a program: scan record stubs and
    /// function prototypes, populate record fields, then check function bodies.
    pub fn run(&mut self, program: &surface::Program) {
        // Project each item once — `kind()` re-walks the node and is not
        // memoized — then run the passes over the typed collections. Record
        // stubs and function prototypes must all exist before field types and
        // bodies are resolved (mutual recursion / forward references).
        let mut records = Vec::new();
        let mut functions = Vec::new();
        let mut trampolines = Vec::new();
        for stmt in program {
            let span = span_of(stmt);
            match stmt.kind() {
                surface::StmtKind::Record(rec) => records.push((rec, span)),
                surface::StmtKind::Function(func) => functions.push((func, span)),
                surface::StmtKind::ExternTrampoline(t) => trampolines.push((t, span)),
                // `mod` is a no-op while every program is one flat module.
                surface::StmtKind::Mod(..) => {}
            }
        }
        for (rec, span) in &records {
            self.scan_record(rec, *span);
        }
        for (func, span) in &functions {
            self.scan_function(func, *span);
        }
        for (rec, _) in &records {
            self.populate_record(rec);
        }
        for (func, span) in &functions {
            self.check_function(func, *span);
        }
        for (tramp, span) in &trampolines {
            self.collect_trampoline(tramp, *span);
        }
    }

    fn scan_record(&mut self, rec: &surface::Record, span: Option<Span>) {
        let ty_params = self.collect_generics(&rec.ty_params, span);
        let default_cap = match rec.default_cap {
            surface::Capability::Value => DefaultCap::Value,
            surface::Capability::Shared => DefaultCap::Shared,
            surface::Capability::Regional => DefaultCap::Regional,
            other => {
                self.error(
                    span,
                    format!("`{other:?}` is not a valid record capability"),
                );
                DefaultCap::Shared
            }
        };
        let name = rec.name;
        let Some(def) = self.defs.declare_record(name) else {
            self.error(
                span,
                format!("record `{}` is defined more than once", self.sym(name)),
            );
            return;
        };
        let record = Record {
            def,
            name,
            ty_params,
            kind: rec.kind,
            default_cap,
            fields: None,
            regional_generics: Vec::new(),
            span,
        };
        self.records.insert(def, record);
    }

    fn scan_function(&mut self, func: &surface::Function, span: Option<Span>) {
        let generics = self.collect_generics(&func.generics, span);
        self.generic_names = generics.iter().map(|(n, id)| (*n, *id)).collect();

        // A `[flex]` annotation on a bare generic (its coloring is dropped, as in
        // the reference) records that the generic must be instantiated regional.
        let mut regional_generics: Vec<GenericId> = Vec::new();
        let mut params = Vec::new();
        for (name, ty, flex) in &func.params {
            let t = self.eval_type_flex(ty, *flex);
            if *flex {
                collect_regional_generic(t, &mut regional_generics);
            }
            params.push((*name, t));
        }
        let return_ty = match &func.return_type {
            Some((ty, flex)) => {
                let t = self.eval_type_flex(ty, *flex);
                if *flex {
                    collect_regional_generic(t, &mut regional_generics);
                }
                t
            }
            None => self.tcx.mk_unit(),
        };
        self.generic_names.clear();

        let name = func.name;
        let Some(def) = self.defs.declare_function(name) else {
            self.error(
                span,
                format!("function `{}` is defined more than once", self.sym(name)),
            );
            return;
        };
        let proto = FuncProto {
            def,
            name,
            visibility: func.visibility,
            generics,
            regional_generics,
            params,
            return_ty,
            is_regional: func.is_regional,
            span,
        };
        self.functions.insert(def, proto);
    }

    /// Allocate generics for a list of `(name, bounds)` declarations.
    fn collect_generics(
        &mut self,
        decls: &[(TokenKey, Vec<surface::Path>)],
        span: Option<Span>,
    ) -> Vec<(TokenKey, GenericId)> {
        decls
            .iter()
            .map(|(name, bounds)| {
                let bounds = self.resolve_bounds(bounds, span);
                let id = self.fresh_generic(*name, bounds);
                (*name, id)
            })
            .collect()
    }

    fn populate_record(&mut self, rec: &surface::Record) {
        let Some(def) = self.defs.resolve_record(rec.name) else {
            return;
        };
        let Some(record) = self.records.get(&def) else {
            return;
        };
        let ty_params = record.ty_params.clone();
        let span = record.span;
        let default_cap = record.default_cap;
        self.generic_names = ty_params.iter().map(|(n, id)| (*n, *id)).collect();

        // A generic at the head of a `[field]` link (`inner: [field] T`) must be
        // instantiated regional; record the requirement for the mono check.
        let mut regional_generics: Vec<GenericId> = Vec::new();
        let fields = match &rec.fields {
            surface::RecordFields::Named(fs) => RecordFields::Named(
                fs.iter()
                    .map(|f| {
                        let (name, ty, mutable) = &f.value;
                        let fty = self.field_ty(ty, *mutable);
                        if *mutable {
                            self.note_link_element(fty, &mut regional_generics, span);
                        }
                        (*name, fty, *mutable)
                    })
                    .collect(),
            ),
            surface::RecordFields::Unnamed(fs) => RecordFields::Unnamed(
                fs.iter()
                    .map(|f| {
                        let (ty, mutable) = &f.value;
                        let fty = self.field_ty(ty, *mutable);
                        if *mutable {
                            self.note_link_element(fty, &mut regional_generics, span);
                        }
                        (fty, *mutable)
                    })
                    .collect(),
            ),
            surface::RecordFields::Variants(vs) => RecordFields::Variants(
                vs.iter()
                    .map(|v| {
                        let (name, tys) = &v.value;
                        Variant {
                            name: *name,
                            fields: tys.iter().map(|t| self.eval_type(t)).collect(),
                        }
                    })
                    .collect(),
            ),
        };
        self.generic_names.clear();

        // Validate that mutable (`[field]`) fields require a regional record.
        if default_cap != DefaultCap::Regional {
            let has_mut = match &fields {
                RecordFields::Named(fs) => fs.iter().any(|(_, _, m)| *m),
                RecordFields::Unnamed(fs) => fs.iter().any(|(_, m)| *m),
                RecordFields::Variants(_) => false,
            };
            if has_mut {
                self.error(
                    span,
                    "`[field]` mutable fields require a `[regional]` record",
                );
            }
        }

        if let Some(r) = self.records.get_mut(&def) {
            r.fields = Some(fields);
            r.regional_generics = regional_generics;
        }
    }

    /// Resolve and register an extern-trampoline root: look up the target
    /// function, evaluate its concrete type arguments, arity-check them against
    /// the target's generics, and record a [`TrampolineRoot`]. Exported
    /// trampolines are the entry points that seed monomorphization, so they must
    /// be collected rather than dropped.
    fn collect_trampoline(&mut self, tramp: &surface::ExternTrampoline, span: Option<Span>) {
        // The target's type arguments are concrete here (no generics in scope).
        let ty_args: Vec<Ty<'tcx>> = tramp.ty_args.iter().map(|t| self.eval_type(t)).collect();

        let Some(target) = self.defs.resolve_function(tramp.func.basename) else {
            self.error(
                span,
                format!(
                    "trampoline target function `{}` not found",
                    self.sym(tramp.func.basename)
                ),
            );
            return;
        };

        // `resolve_function` only yields a function `DefId`, so the prototype is
        // present; `get` rather than indexing keeps this total if that invariant
        // ever loosens (e.g. a name resolving to a non-function item).
        let Some(proto) = self.functions.get(&target) else {
            self.error(
                span,
                format!(
                    "trampoline target `{}` is not a function",
                    self.sym(tramp.func.basename)
                ),
            );
            return;
        };
        let expected = proto.generics.len();
        if ty_args.len() != expected {
            self.error(
                span,
                format!(
                    "trampoline type argument count mismatch: expected {expected}, got {}",
                    ty_args.len()
                ),
            );
            return;
        }

        self.trampolines.push(TrampolineRoot {
            name: tramp.name.clone(),
            abi: tramp.abi.clone(),
            target,
            ty_args,
        });
    }
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// The stored type of a field. A mutable (`[field]`) field is a nullable
    /// regional link, so its type is wrapped in `Nullable`.
    fn field_ty(&mut self, ty: &surface::Type, mutable: bool) -> Ty<'tcx> {
        let t = self.eval_type(ty);
        if mutable { self.tcx.mk_nullable(t) } else { t }
    }

    /// Enforce that a `[field]` link's element is regional. The element (peeled
    /// from the `Nullable` link) must be a regional record: a concrete
    /// value/shared element is rejected here; a generic element records a
    /// requirement checked at the monomorphization call boundary.
    fn note_link_element(
        &mut self,
        field_ty: Ty<'tcx>,
        regional: &mut Vec<GenericId>,
        span: Option<Span>,
    ) {
        let elem = match field_ty.kind() {
            TyKind::Nullable(inner) => *inner,
            _ => field_ty,
        };
        match elem.kind() {
            TyKind::Generic(g) => {
                if !regional.contains(g) {
                    regional.push(*g);
                }
            }
            TyKind::Record {
                flex: Capability::Irrelevant,
                ..
            } => self.error(span, "a `[field]` link element must be a regional record"),
            // Regional records are fine; non-record elements are already rejected
            // by the `Nullable` pointer-like check.
            _ => {}
        }
    }
}

fn span_of(stmt: &surface::Stmt) -> Option<Span> {
    Some(stmt.span())
}
