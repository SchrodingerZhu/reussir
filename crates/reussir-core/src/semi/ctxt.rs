//! The elaboration context: collected items, global + per-function state, and
//! the scan/collect driver.

use reussir_syntax::kind::{Resolver, TokenKey};
use reussir_syntax::source::{FileId, SourceCache};
use rustc_hash::FxHashMap;

use crate::semi::infer::InferCtxt;
use crate::semi::resolve::DefTable;
use crate::semi::traits::builtins::Builtins;
use crate::semi::traits::{TraitDb, TraitId};
use crate::semi::ty::{DefId, Flexivity, GenericId, HoleId, Ty, TyCtxt, TyKind};
use crate::surface::{self, Span};
use crate::utils::frecency::Frecency;
use crate::utils::fuzzy::FuzzyIndex;
use crate::utils::string::StringUniqifier;

use super::fulfill::FulfillCtxt;
use super::hir::{ExprId, Function, VarId};

/// The capability a record declares by default. (Per-use [`crate::semi::ty::Flexivity`]
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

/// A diagnostic. The `span` indexes `file`'s text in the compilation's
/// [`SourceCache`]; items never straddle files, so one file id per report is
/// exact.
#[derive(Clone, Debug)]
pub struct Report {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    pub file: FileId,
}

/// A module-level transform script collected from source.
#[derive(Clone, Debug)]
pub struct TransformScript {
    pub body: String,
    pub span: Option<Span>,
    pub file: FileId,
}

/// Render `reports` to stderr with source-caret context and return whether any
/// was an error (warnings alone do not fail a compile). A report the middle-end
/// could not trace back to a span — an internal/whole-program error — prints as
/// a plain line. The frontend driver (`rrc`) funnels through here so parse and
/// elaboration diagnostics render identically.
pub fn render_reports(cache: &SourceCache, reports: &[Report]) -> bool {
    use std::io::IsTerminal;

    let color = std::io::stderr().is_terminal();
    render_reports_to(cache, reports, color, std::io::stderr().lock())
}

/// Writer-taking variant of [`render_reports`], for frontends that own their
/// display (e.g. the REPL TUI renders into a buffer and styles the lines
/// itself — stderr would be invisible in the alternate screen).
pub fn render_reports_to(
    cache: &SourceCache,
    reports: &[Report],
    color: bool,
    out: impl std::io::Write,
) -> bool {
    use reussir_syntax::diagnostics::{self, Diagnostic, Severity as RenderSeverity};

    let had_error = reports
        .iter()
        .any(|r| matches!(r.severity, Severity::Error));
    if reports.is_empty() {
        return had_error;
    }
    let diags: Vec<Diagnostic> = reports
        .iter()
        .map(|r| Diagnostic {
            file: r.file,
            span: r.span.map(|s| (s.start, s.end)),
            severity: match r.severity {
                Severity::Error => RenderSeverity::Error,
                Severity::Warning => RenderSeverity::Warning,
            },
            message: &r.message,
        })
        .collect();
    let _ = diagnostics::render(cache, &diags, color, out);
    had_error
}

/// One file of a package, ready to elaborate: its id in the compilation's
/// source cache, its module path (`[pkg]` for `lib.rr`, `[pkg, sub, …]` for
/// submodules), and its parsed program. All files of a package must share one
/// token interner — cross-file resolution compares interned keys.
pub struct PackageFile<'p> {
    pub file: FileId,
    pub module: Vec<TokenKey>,
    pub program: &'p surface::Program,
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
    /// `#[repr(fixed)]`: uniform max-arm box sizing for this enum instead of the
    /// default per-constructor sizing. Only ever set for enums.
    pub repr_fixed: bool,
    pub fields: Option<RecordFields<'tcx>>,
    /// Generics that must be instantiated regional because they appear as the
    /// element of a `[field]` link (e.g. `inner: [field] T`). Checked at the
    /// monomorphization call boundary, like a function's `regional_generics`.
    pub regional_generics: Vec<GenericId>,
    pub span: Option<Span>,
    /// The file the record is declared in (spans index it).
    pub file: FileId,
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
    /// The file the function is declared in (spans index it).
    pub file: FileId,
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

    /// Names currently visible in scope (innermost shadowing outermost is not
    /// deduplicated). Used to build "did you mean" hints for an unknown name.
    pub fn names(&self) -> impl Iterator<Item = TokenKey> + '_ {
        self.scope.iter().map(|(name, _)| *name)
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
    /// Frequency-and-recency weight per name, accumulated as names resolve
    /// successfully during checking; used to rank "did you mean" suggestions.
    pub frecency: Frecency<TokenKey>,
    pub records: FxHashMap<DefId, Record<'tcx>>,
    pub functions: FxHashMap<DefId, FuncProto<'tcx>>,
    /// Resolved extern-trampoline roots (mono seeds). See [`TrampolineRoot`].
    pub trampolines: Vec<TrampolineRoot<'tcx>>,
    /// Functions explicitly marked with `#[transform_anchor]`, in source order.
    pub transform_anchors: Vec<DefId>,
    /// Inline transform scripts, in source order.
    pub transform_scripts: Vec<TransformScript>,
    pub generics: Vec<GenericInfo>,
    pub strings: StringUniqifier,
    pub elaborated: Vec<Function<'tcx>>,
    pub reports: Vec<Report>,
    /// The file whose items are currently being processed; stamped onto every
    /// report and declared item. The driver sets it per input file
    /// ([`Elaborator::set_current_file`]); the check passes restore it per item
    /// from the item's own declaration file.
    pub current_file: FileId,
    /// Whether record fields are populated for the current [`run_files`]
    /// batch. `Arc` well-formedness needs member types (the structural `Sync`
    /// check), so annotation-site checks fire only once this is set; types
    /// evaluated earlier (signatures, record members) are re-checked by the
    /// post-populate sweep ([`Elaborator::sweep_arc_wf`]).
    ///
    /// [`run_files`]: Elaborator::run_files
    pub(super) records_complete: bool,

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
    /// Holes already reported as ambiguous for the current function, so each
    /// yields exactly one diagnostic — from `resolve_obligations` when a bound
    /// on it can't be resolved, else from zonking (see `zonk_ty`).
    pub(super) reported_holes: rustc_hash::FxHashSet<HoleId>,
    expr_counter: u32,
}

/// A checkpoint of the elaborator's accumulated state; see
/// [`Elaborator::checkpoint`].
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    pub(super) defs: usize,
    pub(super) generics: usize,
    pub(super) trampolines: usize,
    pub(super) transform_anchors: usize,
    pub(super) transform_scripts: usize,
    pub(super) elaborated: usize,
    pub(super) reports: usize,
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
            ("Sync", builtins.sync),
        ]
        .into_iter()
        .collect();
        Elaborator {
            tcx,
            resolver,
            defs: DefTable::new(),
            frecency: Frecency::new(),
            traits,
            builtins,
            trait_names,
            trampolines: Vec::new(),
            transform_anchors: Vec::new(),
            transform_scripts: Vec::new(),
            records: FxHashMap::default(),
            functions: FxHashMap::default(),
            generics: Vec::new(),
            strings: StringUniqifier::new(),
            elaborated: Vec::new(),
            reports: Vec::new(),
            current_file: FileId::ROOT,
            records_complete: false,
            infer: InferCtxt::new(tcx),
            vars: VarEnv::default(),
            generic_names: FxHashMap::default(),
            inside_region: false,
            regional_generics: Vec::new(),
            fulfill: FulfillCtxt::default(),
            reported_holes: rustc_hash::FxHashSet::default(),
            expr_counter: 0,
        }
    }

    // ----- diagnostics -----

    /// Switch the file whose items are being processed. Reports and item
    /// declarations from here on are attributed to `file`.
    pub fn set_current_file(&mut self, file: FileId) {
        self.current_file = file;
    }

    pub fn error(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Error,
            message: message.into(),
            span,
            file: self.current_file,
        });
    }

    pub fn warn(&mut self, span: Option<Span>, message: impl Into<String>) {
        self.reports.push(Report {
            severity: Severity::Warning,
            message: message.into(),
            span,
            file: self.current_file,
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

    /// Render `ty` the way the surface spells it, for diagnostics: `i32`,
    /// `bool`, `Cell<u64>`, `[rigid] Node`, `Nullable<Data>`,
    /// `(i32, i32) -> i32`. A generic prints its declared name; an unsolved
    /// inference hole prints `_` — pass a *deeply resolved* type
    /// (`InferCtxt::resolve`) so solved holes show their solutions. The
    /// `Debug` form (`Int(Signed(32))`) is for compiler logs only and must
    /// not reach user-facing reports.
    pub fn ty_display(&self, ty: Ty<'tcx>) -> String {
        let mut out = String::new();
        self.push_ty_display(&mut out, ty);
        out
    }

    fn push_ty_display(&self, out: &mut String, ty: Ty<'tcx>) {
        use crate::semi::ty::{FpTy, IntTy};
        use std::fmt::Write as _;
        match *ty.kind() {
            TyKind::Int(IntTy::Signed(w)) => {
                let _ = write!(out, "i{w}");
            }
            TyKind::Int(IntTy::Unsigned(w)) => {
                let _ = write!(out, "u{w}");
            }
            TyKind::Fp(FpTy::Ieee(w)) => {
                let _ = write!(out, "f{w}");
            }
            TyKind::Fp(FpTy::BFloat16) => out.push_str("bfloat16"),
            TyKind::Fp(FpTy::Float8) => out.push_str("float8"),
            TyKind::Bool => out.push_str("bool"),
            TyKind::Str => out.push_str("str"),
            TyKind::Char => out.push_str("char"),
            TyKind::Unit => out.push_str("unit"),
            TyKind::Bottom => out.push('!'),
            TyKind::Generic(g) => out.push_str(
                self.generics
                    .get(g.0 as usize)
                    .map_or("_", |info| self.resolver.resolve(info.name)),
            ),
            TyKind::Hole(_) => out.push('_'),
            TyKind::Nullable(inner) => {
                out.push_str("Nullable<");
                self.push_ty_display(out, inner);
                out.push('>');
            }
            TyKind::Cell { elem, exclusive } => {
                out.push_str(if exclusive { "RefCell<" } else { "Cell<" });
                self.push_ty_display(out, elem);
                out.push('>');
            }
            TyKind::Arc(inner) => {
                out.push_str("Arc<");
                self.push_ty_display(out, inner);
                out.push('>');
            }
            TyKind::Array { elem, dims } => {
                out.push('[');
                self.push_ty_display(out, elem);
                out.push(';');
                for (i, extent) in dims.iter().enumerate() {
                    let sep = if i > 0 { "," } else { "" };
                    let _ = write!(out, "{sep} {extent}");
                }
                out.push(']');
            }
            TyKind::Record { def, args, flex } => {
                match flex {
                    Flexivity::Flex => out.push_str("[flex] "),
                    Flexivity::Rigid => out.push_str("[rigid] "),
                    Flexivity::Regional => out.push_str("[regional] "),
                    Flexivity::Irrelevant => {}
                }
                out.push_str(&self.defs.path(def).display(self.resolver));
                if !args.is_empty() {
                    out.push('<');
                    for (i, &arg) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        self.push_ty_display(out, arg);
                    }
                    out.push('>');
                }
            }
            TyKind::Closure { params, ret } => {
                if let [single] = params {
                    self.push_ty_display(out, *single);
                } else {
                    out.push('(');
                    for (i, &p) in params.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        self.push_ty_display(out, p);
                    }
                    out.push(')');
                }
                out.push_str(" -> ");
                self.push_ty_display(out, ret);
            }
        }
    }

    /// Record one successful resolution of `name`, feeding the frecency model
    /// that ranks future "did you mean" suggestions.
    pub(super) fn record_use(&mut self, name: TokenKey) {
        self.frecency.record(name);
    }

    // ----- module-aware reference resolution -----

    /// Classify a reference path's qualifier segments: the `root`/`super`
    /// module keywords (by their interned text) or ordinary names.
    pub(super) fn classify_segs(&self, path: &surface::Path) -> Vec<crate::semi::resolve::PathSeg> {
        use crate::semi::resolve::PathSeg;
        path.segments
            .iter()
            .map(|&k| match self.sym(k) {
                "root" => PathSeg::Root,
                "super" => PathSeg::Super,
                _ => PathSeg::Name(k),
            })
            .collect()
    }

    /// Render a reference path as the user wrote it (`a::b::name`).
    pub(super) fn path_display(&self, path: &surface::Path) -> String {
        let mut out = String::new();
        for &seg in &path.segments {
            out.push_str(self.sym(seg));
            out.push_str("::");
        }
        out.push_str(self.sym(path.basename));
        out
    }

    /// Resolve a type reference: bare names in the current module (falling
    /// back to the crate root), qualified paths per the module-relative rules
    /// (see [`DefTable::resolve_record_path`]).
    pub(super) fn resolve_record_ref(&self, path: &surface::Path) -> Option<DefId> {
        if path.segments.is_empty() {
            self.defs.resolve_record(path.basename)
        } else {
            self.defs
                .resolve_record_path(&self.classify_segs(path), path.basename)
        }
    }

    /// Resolve a value (function) reference; see [`Self::resolve_record_ref`].
    pub(super) fn resolve_function_ref(&self, path: &surface::Path) -> Option<DefId> {
        if path.segments.is_empty() {
            self.defs.resolve_function(path.basename)
        } else {
            self.defs
                .resolve_function_path(&self.classify_segs(path), path.basename)
        }
    }

    /// Resolve a constructor path's *qualifier* as a record: for
    /// `m::Enum::Variant` the qualifier is `m::Enum` (the basename names the
    /// variant). `None` when the path has no qualifier.
    pub(super) fn resolve_ctor_qualifier(&self, path: &surface::Path) -> Option<DefId> {
        let (&enum_name, mods) = path.segments.split_last()?;
        if mods.is_empty() {
            self.defs.resolve_record(enum_name)
        } else {
            let mods_path = surface::Path {
                basename: enum_name,
                segments: mods.iter().copied().collect(),
            };
            self.defs
                .resolve_record_path(&self.classify_segs(&mods_path), enum_name)
        }
    }

    /// Enter `def`'s declaration scope: reports attribute to `file`, and
    /// bare/relative references resolve against the item's own module (its
    /// qualified path minus the item name).
    pub(super) fn enter_item_scope(&mut self, def: DefId, file: FileId) {
        self.set_current_file(file);
        let path = &self.defs.path(def).0;
        let module = path[..path.len() - 1].to_vec();
        self.defs.set_module(module);
    }

    // ----- "did you mean" suggestions -----
    //
    // Each helper fuzzy-searches the relevant namespace for the closest spelling
    // to an unresolved name and returns a hint suffix (e.g. "; did you mean
    // `List`?") ready to append to the diagnostic, or an empty string when no
    // candidate is close enough. Candidates are weighted by their frecency, so
    // among equally-plausible spellings the name the program leans on wins.
    // Appending the empty string is a no-op, so call sites stay uniform whether
    // or not a suggestion exists.

    /// Hint for an unresolved record/type/enum name.
    pub(super) fn record_suggestion(&self, name: TokenKey) -> String {
        self.fuzzy_hint(
            self.sym(name),
            self.defs
                .record_names()
                .map(|k| (self.sym(k), self.frecency.score(k))),
        )
    }

    /// Hint for an unresolved function name.
    pub(super) fn function_suggestion(&self, name: TokenKey) -> String {
        self.fuzzy_hint(
            self.sym(name),
            self.defs
                .function_names()
                .map(|k| (self.sym(k), self.frecency.score(k))),
        )
    }

    /// Hint for an unknown bare variable, drawn from the names in scope.
    pub(super) fn variable_suggestion(&self, name: TokenKey) -> String {
        self.fuzzy_hint(
            self.sym(name),
            self.vars
                .names()
                .map(|k| (self.sym(k), self.frecency.score(k))),
        )
    }

    /// Hint for an unknown trait bound, drawn from the built-in trait names.
    /// Built-ins carry no usage history, so every candidate has zero frecency.
    pub(super) fn trait_suggestion(&self, name: &str) -> String {
        self.fuzzy_hint(name, self.trait_names.keys().map(|&k| (k, 0.0)))
    }

    /// Search `candidates` (each a `(name, frecency)` pair) for the best
    /// correction of `query` and render it as a `"; did you mean `X`?"` suffix,
    /// or `""` if nothing is close enough.
    fn fuzzy_hint<'b>(
        &self,
        query: &str,
        candidates: impl Iterator<Item = (&'b str, f32)>,
    ) -> String {
        let mut index = FuzzyIndex::new();
        for (name, frecency) in candidates {
            index.insert(name, frecency);
        }
        match index.suggest(query) {
            Some(hint) => format!("; did you mean `{hint}`?"),
            None => String::new(),
        }
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
                let hint = self.trait_suggestion(name);
                self.error(span, format!("unknown trait bound `{name}`{hint}"));
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
        self.reported_holes.clear();
        for (name, id) in generics {
            self.generic_names.insert(*name, *id);
        }
    }

    // ----- driver -----

    /// A snapshot of the elaborator's accumulated (append-only) state, taken
    /// with [`Elaborator::checkpoint`] and restored with
    /// [`Elaborator::rollback`]. Everything a [`run`](Elaborator::run) call
    /// appends is covered; state that is content-addressed or purely
    /// monotonic (`strings`, `frecency`, `expr_counter`, the type arena) is
    /// deliberately left to grow — stale entries are unreachable once the
    /// defs that referenced them are retracted.
    ///
    /// Any *future* accumulated state must be added here: notably, once
    /// user-defined traits/impls land, `traits` stops being fixed at
    /// construction and a rejected batch's impls must roll back too.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            defs: self.defs.len(),
            generics: self.generics.len(),
            trampolines: self.trampolines.len(),
            transform_anchors: self.transform_anchors.len(),
            transform_scripts: self.transform_scripts.len(),
            elaborated: self.elaborated.len(),
            reports: self.reports.len(),
        }
    }

    /// Restore the accumulated state to `cp`. Per-function working state is
    /// reset by `enter_function` on the next check, so it needs no restoring.
    pub fn rollback(&mut self, cp: &Checkpoint) {
        self.defs.truncate(cp.defs);
        // `records`/`functions` are keyed by the dense DefId space, and the
        // passes only ever insert under DefIds declared in the same run, so
        // retracting every id past the checkpoint restores both maps.
        let live = cp.defs as u32;
        self.records.retain(|def, _| def.0 < live);
        self.functions.retain(|def, _| def.0 < live);
        self.generics.truncate(cp.generics);
        self.trampolines.truncate(cp.trampolines);
        self.transform_anchors.truncate(cp.transform_anchors);
        self.transform_scripts.truncate(cp.transform_scripts);
        self.elaborated.truncate(cp.elaborated);
        self.reports.truncate(cp.reports);
    }

    /// Incrementally elaborate one batch of items against the accumulated
    /// program, atomically: on any error the elaborator is rolled back to its
    /// state before the call and the batch's reports are returned as `Err`;
    /// on success the batch's warnings (if any) are returned as `Ok` and the
    /// items stay. Duplicate definitions are rejected (the existing "defined
    /// more than once" path), never replaced — a REPL session's items are
    /// immutable once accepted.
    ///
    /// Forward references *within* the batch work exactly as in batch mode;
    /// references to previously accepted items resolve through the persistent
    /// [`DefTable`].
    pub fn try_extend(&mut self, program: &surface::Program) -> Result<Vec<Report>, Vec<Report>> {
        let cp = self.checkpoint();
        self.run(program);
        // Split the batch's reports off first: the caller renders them either
        // way, and `has_errors` should keep reflecting only accepted state.
        let new = self.reports.split_off(cp.reports);
        if new.iter().any(|r| r.severity == Severity::Error) {
            self.rollback(&cp);
            Err(new)
        } else {
            Ok(new)
        }
    }

    /// Run the collection passes over one program in the current module (the
    /// single-file / REPL entry): scan record stubs and function prototypes,
    /// populate record fields, then check function bodies.
    pub fn run(&mut self, program: &surface::Program) {
        let file = self.current_file;
        let module = self.defs.module().to_vec();
        self.run_files(&[(file, module, program)]);
    }

    /// Run the collection passes over a whole package — one translation unit
    /// spanning many files, each with its own module path.
    pub fn run_package(&mut self, files: &[PackageFile<'_>]) {
        let files: Vec<(FileId, Vec<TokenKey>, &surface::Program)> = files
            .iter()
            .map(|f| (f.file, f.module.clone(), f.program))
            .collect();
        self.run_files(&files);
    }

    /// The shared driver: scan **all** files' record stubs and function
    /// prototypes first (cross-file forward references and mutual recursion),
    /// then populate record fields, check function bodies (each pass restores
    /// the owning item's file/module scope), and collect trampoline roots per
    /// file.
    fn run_files(&mut self, files: &[(FileId, Vec<TokenKey>, &surface::Program)]) {
        // This batch's record fields are not populated yet, so `Arc` wf
        // checks (which need member types for the structural `Sync` half)
        // stay silent from the very first scan — a previous batch may have
        // left the flag set — until the post-populate sweep re-checks.
        self.records_complete = false;
        // Project each item once — `kind()` re-walks the node and is not
        // memoized — then run the passes over the typed collections, each item
        // tagged with its file/module scope.
        #[derive(Clone, Copy)]
        struct Scope<'p> {
            file: FileId,
            module: &'p [TokenKey],
        }
        let mut records = Vec::new();
        let mut functions = Vec::new();
        let mut trampolines = Vec::new();
        for (file, module, program) in files {
            for stmt in *program {
                let scope = Scope {
                    file: *file,
                    module,
                };
                let span = span_of(stmt);
                let attrs = stmt.attributes();
                self.set_current_file(*file);
                match stmt.kind() {
                    surface::StmtKind::Record(rec) => {
                        self.validate_transform_anchor(&attrs, None);
                        records.push((rec, span, scope));
                    }
                    surface::StmtKind::Function(func) => {
                        let anchor =
                            self.validate_transform_anchor(&attrs, Some(func.body.is_some()));
                        functions.push((func, span, scope, anchor));
                    }
                    surface::StmtKind::ExternTrampoline(t) => {
                        self.validate_transform_anchor(&attrs, None);
                        trampolines.push((t, span, scope));
                    }
                    // `mod` declarations drive package *discovery* (the driver
                    // maps them to files before elaboration); here they carry
                    // no items.
                    surface::StmtKind::Mod(..) => {
                        self.validate_transform_anchor(&attrs, None);
                    }
                    surface::StmtKind::Transform(script) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.transform_scripts.push(TransformScript {
                            body: script.body,
                            span: Some(script.body_span),
                            file: *file,
                        });
                    }
                }
            }
        }
        // The later passes are keyed by the DefId each scan returned — never by
        // re-resolving the name — so an item whose declaration failed (e.g. a
        // duplicate) is skipped instead of clobbering the previously declared
        // item's fields or checking a body against the wrong prototype.
        let record_defs: Vec<Option<DefId>> = records
            .iter()
            .map(|(rec, span, scope)| {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                self.scan_record(rec, *span)
            })
            .collect();
        let function_defs: Vec<Option<DefId>> = functions
            .iter()
            .map(|(func, span, scope, _)| {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                self.scan_function(func, *span)
            })
            .collect();
        functions
            .iter()
            .zip(&function_defs)
            .filter_map(|((_, _, _, anchor), def)| if *anchor { *def } else { None })
            .for_each(|def| self.transform_anchors.push(def));
        let record_defs: Vec<DefId> = records
            .iter()
            .zip(record_defs)
            .filter_map(|((rec, _, _), def)| Some((rec, def?)))
            .map(|(rec, def)| {
                self.populate_record(rec, def);
                def
            })
            .collect();
        // Now that this batch's fields exist, reject inline-recursive record
        // cycles: a `[value]` record stored inside a `[value]` record, around
        // to itself, never crosses a box and has no finite layout. Forward
        // references only resolve within a batch, so a new cycle is always
        // rooted in this batch's defs.
        self.reject_infinite_value_recursion(&record_defs);
        // Post-populate sweep: signatures and record members were evaluated
        // before fields existed, so their `Arc` nodes wf-check only now.
        // Restricted to this batch's items so a REPL re-run does not re-report
        // earlier ones. Body annotations check inline during `check_function`.
        for def in record_defs {
            let Some(rec) = self.records.get(&def) else {
                continue;
            };
            let (span, file) = (rec.span, rec.file);
            let member_tys: Vec<Ty<'tcx>> = match &rec.fields {
                Some(RecordFields::Named(fs)) => fs.iter().map(|(_, t, _)| *t).collect(),
                Some(RecordFields::Unnamed(fs)) => fs.iter().map(|(t, _)| *t).collect(),
                Some(RecordFields::Variants(vs)) => {
                    vs.iter().flat_map(|v| v.fields.iter().copied()).collect()
                }
                None => Vec::new(),
            };
            self.set_current_file(file);
            for t in member_tys {
                self.sweep_arc_wf(t, span);
            }
        }
        for def in function_defs.iter().flatten() {
            let Some(proto) = self.functions.get(def) else {
                continue;
            };
            let (span, file) = (proto.span, proto.file);
            let tys: Vec<Ty<'tcx>> = proto
                .params
                .iter()
                .map(|(_, t)| *t)
                .chain([proto.return_ty])
                .collect();
            self.set_current_file(file);
            for t in tys {
                self.sweep_arc_wf(t, span);
            }
        }
        self.records_complete = true;
        functions
            .iter()
            .zip(function_defs)
            .filter_map(|((func, span, _, _), def)| Some((func, *span, def?)))
            .for_each(|(func, span, def)| {
                self.check_function(func, def, span);
            });
        trampolines.iter().for_each(|(tramp, span, scope)| {
            self.set_current_file(scope.file);
            self.defs.set_module(scope.module.to_vec());
            self.collect_trampoline(tramp, *span);
        });
    }

    fn validate_transform_anchor(
        &mut self,
        attrs: &[surface::Attribute],
        function_has_body: Option<bool>,
    ) -> bool {
        let mut seen = false;
        let mut valid = true;
        let mut span = None;
        for attr in attrs {
            if self.sym(attr.name) != "transform_anchor" {
                continue;
            }
            span = Some(attr.span);
            if function_has_body.is_none() {
                self.error(
                    span,
                    "`#[transform_anchor]` may only be attached to a function",
                );
                valid = false;
                continue;
            }
            if seen {
                self.error(
                    span,
                    "`#[transform_anchor]` may only be specified once per function",
                );
                valid = false;
            }
            seen = true;
            if !attr.args.is_empty() {
                self.error(span, "`#[transform_anchor]` does not accept arguments");
                valid = false;
            }
        }
        if seen && function_has_body == Some(false) {
            self.error(span, "`#[transform_anchor]` requires a function body");
            valid = false;
        }
        seen && valid
    }

    fn scan_record(&mut self, rec: &surface::Record, span: Option<Span>) -> Option<DefId> {
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
        // `#[repr(fixed)]` pins uniform max-arm box sizing — a property only a
        // *managed* (shared/regional, fused-header) enum's box has: a struct
        // has no arms and a `[value]` enum is never boxed. Reject it elsewhere
        // (rather than silently dropping it) and zero the flag so the "only
        // ever set for managed enums" invariant holds from here on.
        let repr_fixed = rec.repr_fixed
            && rec.kind == surface::RecordKind::EnumKind
            && default_cap != DefaultCap::Value;
        if rec.repr_fixed && !repr_fixed {
            self.error(
                span,
                "`#[repr(fixed)]` is only meaningful for managed (`shared`/`regional`) enums"
                    .to_string(),
            );
        }
        let name = rec.name;
        if matches!(self.sym(name), "Cell" | "RefCell" | "Arc") {
            self.error(
                span,
                format!(
                    "record name `{}` is reserved for the builtin type",
                    self.sym(name)
                ),
            );
            return None;
        }
        let Some(def) = self.defs.declare_record(name) else {
            self.error(
                span,
                format!("record `{}` is defined more than once", self.sym(name)),
            );
            return None;
        };
        let record = Record {
            def,
            name,
            ty_params,
            kind: rec.kind,
            default_cap,
            repr_fixed,
            fields: None,
            regional_generics: Vec::new(),
            span,
            file: self.current_file,
        };
        self.records.insert(def, record);
        Some(def)
    }

    fn scan_function(&mut self, func: &surface::Function, span: Option<Span>) -> Option<DefId> {
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
            return None;
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
            file: self.current_file,
        };
        self.functions.insert(def, proto);
        Some(def)
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

    fn populate_record(&mut self, rec: &surface::Record, def: DefId) {
        let Some(record) = self.records.get(&def) else {
            return;
        };
        let ty_params = record.ty_params.clone();
        let span = record.span;
        let default_cap = record.default_cap;
        let file = record.file;
        // Field types resolve (and reports attribute) in the record's own
        // declaration scope.
        self.enter_item_scope(def, file);
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
                        self.reject_arc_member(fty, span);
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
                        self.reject_arc_member(fty, span);
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
                            fields: tys
                                .iter()
                                .map(|t| {
                                    let fty = self.eval_type(t);
                                    self.reject_arc_member(fty, span);
                                    fty
                                })
                                .collect(),
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

        let Some(target) = self.resolve_function_ref(&tramp.func) else {
            let hint = if tramp.func.segments.is_empty() {
                self.function_suggestion(tramp.func.basename)
            } else {
                String::new()
            };
            self.error(
                span,
                format!(
                    "trampoline target function `{}` not found{hint}",
                    self.path_display(&tramp.func)
                ),
            );
            return;
        };
        self.record_use(tramp.func.basename);

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

    /// The defs stored *inline* in `def`'s layout: direct member (or variant
    /// arm field) heads that are `[value]`-capability records. Every other
    /// member kind — shared/regional records, `Arc`, `Nullable`, cells,
    /// arrays, closures — is a pointer and breaks an inline chain. Heads
    /// only: a `[value]` head's *type arguments* can also be stored inline,
    /// but which of them actually are is only known once the generic
    /// grounds, so that half is checked at monomorphization.
    fn inline_value_heads(&self, def: DefId) -> Vec<DefId> {
        let Some(rec) = self.records.get(&def) else {
            return Vec::new();
        };
        let member_tys: Vec<Ty<'tcx>> = match &rec.fields {
            Some(RecordFields::Named(fs)) => fs.iter().map(|(_, t, _)| *t).collect(),
            Some(RecordFields::Unnamed(fs)) => fs.iter().map(|(t, _)| *t).collect(),
            Some(RecordFields::Variants(vs)) => {
                vs.iter().flat_map(|v| v.fields.iter().copied()).collect()
            }
            None => Vec::new(),
        };
        member_tys
            .into_iter()
            .filter_map(|t| match *t.kind() {
                TyKind::Record { def, .. }
                    if self
                        .records
                        .get(&def)
                        .is_some_and(|r| r.default_cap == DefaultCap::Value) =>
                {
                    Some(def)
                }
                _ => None,
            })
            .collect()
    }

    /// Reject inline-recursive record cycles — a `[value]` record reachable
    /// from itself purely through inline storage has no finite layout (the
    /// LLVM conversion would recurse forever computing it). Only
    /// `[value]`-capability records can participate: every edge *into* a
    /// shared/regional record is a pointer, so a cycle is a cycle among
    /// value defs. Reported once per back-edge, naming the cycle.
    pub(super) fn reject_infinite_value_recursion(&mut self, batch: &[DefId]) {
        // 1 = on the current DFS path, 2 = finished.
        let mut color: FxHashMap<DefId, u8> = FxHashMap::default();
        for &root in batch {
            if color.contains_key(&root)
                || !self
                    .records
                    .get(&root)
                    .is_some_and(|r| r.default_cap == DefaultCap::Value)
            {
                continue;
            }
            let mut stack: Vec<(DefId, Vec<DefId>)> =
                vec![(root, self.inline_value_heads(root))];
            color.insert(root, 1);
            while let Some((_, succs)) = stack.last_mut() {
                let Some(next) = succs.pop() else {
                    let (done, _) = stack.pop().expect("non-empty stack");
                    color.insert(done, 2);
                    continue;
                };
                match color.get(&next) {
                    Some(1) => {
                        // Back-edge: the cycle is the path suffix from `next`.
                        let cycle: Vec<String> = stack
                            .iter()
                            .map(|(d, _)| d)
                            .skip_while(|&&d| d != next)
                            .map(|d| self.sym(self.records[d].name).to_owned())
                            .collect();
                        let (span, file) =
                            (self.records[&next].span, self.records[&next].file);
                        self.set_current_file(file);
                        self.error(
                            span,
                            format!(
                                "recursive `[value]` record has infinite size: \
                                 `{}` is stored inline within itself (through `{}`); \
                                 break the cycle with a boxed link (e.g. a \
                                 `[shared]` record or `Arc`)",
                                cycle.first().cloned().unwrap_or_default(),
                                cycle.join("` → `"),
                            ),
                        );
                    }
                    Some(2) => {}
                    None => {
                        color.insert(next, 1);
                        let succs = self.inline_value_heads(next);
                        stack.push((next, succs));
                    }
                    Some(_) => {}
                }
            }
        }
    }

    /// Enforce that a `[field]` link's element is regional. The element (peeled
    /// from the `Nullable` link) must be a regional record: a concrete
    /// Reject a record member whose slot is directly `Arc<…>`: the MLIR record
    /// encoding names an rc-managed member by its capability and has no
    /// per-member atomic axis yet, so an arc cannot sit inline in a record.
    fn reject_arc_member(&mut self, fty: Ty<'tcx>, span: Option<Span>) {
        if matches!(fty.kind(), TyKind::Arc(_)) {
            self.error(span, "an `Arc` record member is not supported yet");
        }
    }

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
                flex: Flexivity::Irrelevant,
                ..
            } => self.error(span, "a `[field]` link element must be a regional record"),
            // Arrays, cells, and arcs are pointer-like (one shared rc box), so
            // the `Nullable` check lets them through — but a `[field]` link
            // stores a regional record, never a shared box.
            TyKind::Array { .. } | TyKind::Cell { .. } | TyKind::Arc(_) => {
                self.error(span, "a `[field]` link element must be a regional record")
            }
            // Regional records are fine; non-record elements are already rejected
            // by the `Nullable` pointer-like check.
            _ => {}
        }
    }
}

fn span_of(stmt: &surface::Stmt) -> Option<Span> {
    Some(stmt.span())
}
