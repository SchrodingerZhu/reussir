//! The elaboration context: collected items, global + per-function state, and
//! the scan/collect driver.

use reussir_syntax::kind::{Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::semi::infer::InferCtxt;
use crate::semi::traits::builtins::Builtins;
use crate::semi::traits::{TraitDb, TraitId};
use crate::semi::ty::{GenericId, Ty, TyCtxt};
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
    pub name: String,
    pub bounds: Vec<TraitId>,
}

/// A record's fields, with concrete field types resolved.
#[derive(Clone, Debug)]
pub enum RecordFields<'tcx> {
    /// `(name, type, is_mutable)`.
    Named(Vec<(String, Ty<'tcx>, bool)>),
    /// `(type, is_mutable)`.
    Unnamed(Vec<(Ty<'tcx>, bool)>),
    Variants(Vec<Variant<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct Variant<'tcx> {
    pub name: String,
    pub fields: Vec<Ty<'tcx>>,
}

/// A collected record (struct or enum). Fields are populated in a second pass.
#[derive(Clone, Debug)]
pub struct Record<'tcx> {
    pub name: String,
    pub ty_params: Vec<(String, GenericId)>,
    pub kind: surface::RecordKind,
    pub default_cap: DefaultCap,
    pub fields: Option<RecordFields<'tcx>>,
    pub span: Option<Span>,
}

/// A collected function prototype (signature only).
#[derive(Clone, Debug)]
pub struct FuncProto<'tcx> {
    pub name: String,
    pub generics: Vec<(String, GenericId)>,
    /// `(name, colored type)`.
    pub params: Vec<(String, Ty<'tcx>)>,
    pub return_ty: Ty<'tcx>,
    pub is_regional: bool,
    pub span: Option<Span>,
}

/// A local variable binding.
#[derive(Clone, Debug)]
pub struct VarDef<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
    pub span: Option<Span>,
}

/// A lexically-scoped variable environment. `VarId`s are stable for a function;
/// only visibility is scoped.
#[derive(Default)]
pub struct VarEnv<'tcx> {
    defs: Vec<VarDef<'tcx>>,
    scope: Vec<(String, VarId)>,
}

impl<'tcx> VarEnv<'tcx> {
    pub fn fresh(&mut self, name: &str, ty: Ty<'tcx>, span: Option<Span>) -> VarId {
        let id = VarId(self.defs.len() as u32);
        self.defs.push(VarDef {
            name: name.to_owned(),
            ty,
            span,
        });
        self.scope.push((name.to_owned(), id));
        id
    }

    pub fn lookup(&self, name: &str) -> Option<(VarId, Ty<'tcx>)> {
        self.scope
            .iter()
            .rev()
            .find(|(n, _)| n == name)
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
    pub records: FxHashMap<String, Record<'tcx>>,
    pub functions: FxHashMap<String, FuncProto<'tcx>>,
    pub generics: Vec<GenericInfo>,
    pub strings: StringUniqifier,
    pub elaborated: Vec<Function<'tcx>>,
    pub reports: Vec<Report>,

    // ----- per-function working state (reset by `enter_function`) -----
    pub infer: InferCtxt<'a, 'tcx>,
    pub vars: VarEnv<'tcx>,
    pub generic_names: FxHashMap<String, GenericId>,
    pub inside_region: bool,
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
            traits,
            builtins,
            trait_names,
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
    pub fn fresh_generic(&mut self, name: &str, bounds: Vec<TraitId>) -> GenericId {
        let id = GenericId(self.generics.len() as u32);
        self.generics.push(GenericInfo {
            name: name.to_owned(),
            bounds,
        });
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
    pub(super) fn enter_function(&mut self, generics: &[(String, GenericId)]) {
        self.infer = InferCtxt::new(self.tcx);
        self.vars.reset();
        self.generic_names.clear();
        self.inside_region = false;
        self.fulfill = FulfillCtxt::default();
        for (name, id) in generics {
            self.generic_names.insert(name.clone(), *id);
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
        for stmt in program {
            let span = span_of(stmt);
            match stmt.kind() {
                surface::StmtKind::Record(rec) => records.push((rec, span)),
                surface::StmtKind::Function(func) => functions.push((func, span)),
                surface::StmtKind::Mod(..) | surface::StmtKind::ExternTrampoline(..) => {}
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
        let name = self.sym(rec.name).to_owned();
        let record = Record {
            name: name.clone(),
            ty_params,
            kind: rec.kind,
            default_cap,
            fields: None,
            span,
        };
        if self.records.insert(name.clone(), record).is_some() {
            self.error(span, format!("record `{name}` is defined more than once"));
        }
    }

    fn scan_function(&mut self, func: &surface::Function, span: Option<Span>) {
        let generics = self.collect_generics(&func.generics, span);
        self.generic_names = generics.iter().map(|(n, id)| (n.clone(), *id)).collect();

        let params = func
            .params
            .iter()
            .map(|(name, ty, flex)| (self.sym(*name).to_owned(), self.eval_type_flex(ty, *flex)))
            .collect();
        let return_ty = match &func.return_type {
            Some((ty, flex)) => self.eval_type_flex(ty, *flex),
            None => self.tcx.mk_unit(),
        };
        self.generic_names.clear();

        let name = self.sym(func.name).to_owned();
        let proto = FuncProto {
            name: name.clone(),
            generics,
            params,
            return_ty,
            is_regional: func.is_regional,
            span,
        };
        if self.functions.insert(name.clone(), proto).is_some() {
            self.error(span, format!("function `{name}` is defined more than once"));
        }
    }

    /// Allocate generics for a list of `(name, bounds)` declarations.
    fn collect_generics(
        &mut self,
        decls: &[(TokenKey, Vec<surface::Path>)],
        span: Option<Span>,
    ) -> Vec<(String, GenericId)> {
        decls
            .iter()
            .map(|(name, bounds)| {
                let name = self.sym(*name).to_owned();
                let bounds = self.resolve_bounds(bounds, span);
                let id = self.fresh_generic(&name, bounds);
                (name, id)
            })
            .collect()
    }

    fn populate_record(&mut self, rec: &surface::Record) {
        let name = self.sym(rec.name).to_owned();
        let Some(record) = self.records.get(&name) else {
            return;
        };
        let ty_params = record.ty_params.clone();
        let span = record.span;
        let default_cap = record.default_cap;
        self.generic_names = ty_params.iter().map(|(n, id)| (n.clone(), *id)).collect();

        let fields = match &rec.fields {
            surface::RecordFields::Named(fs) => RecordFields::Named(
                fs.iter()
                    .map(|f| {
                        let (name, ty, mutable) = &f.value;
                        (
                            self.sym(*name).to_owned(),
                            self.field_ty(ty, *mutable),
                            *mutable,
                        )
                    })
                    .collect(),
            ),
            surface::RecordFields::Unnamed(fs) => RecordFields::Unnamed(
                fs.iter()
                    .map(|f| {
                        let (ty, mutable) = &f.value;
                        (self.field_ty(ty, *mutable), *mutable)
                    })
                    .collect(),
            ),
            surface::RecordFields::Variants(vs) => RecordFields::Variants(
                vs.iter()
                    .map(|v| {
                        let (name, tys) = &v.value;
                        Variant {
                            name: self.sym(*name).to_owned(),
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

        if let Some(r) = self.records.get_mut(&name) {
            r.fields = Some(fields);
        }
    }
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// The stored type of a field. A mutable (`[field]`) field is a nullable
    /// regional link, so its type is wrapped in `Nullable`.
    fn field_ty(&mut self, ty: &surface::Type, mutable: bool) -> Ty<'tcx> {
        let t = self.eval_type(ty);
        if mutable { self.tcx.mk_nullable(t) } else { t }
    }
}

fn span_of(stmt: &surface::Stmt) -> Option<Span> {
    Some(stmt.span())
}
