//! The elaboration context: collected items, global + per-function state, and
//! the scan/collect driver.

use reussir_syntax::SharedInterner;
use reussir_syntax::kind::{Resolver, TokenKey};
use reussir_syntax::source::{FileId, SourceCache};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::semi::infer::InferCtxt;
use crate::semi::resolve::{DefKind, DefTable};
use crate::semi::traits::builtins::Builtins;
use crate::semi::traits::{TraitDb, TraitId};
use crate::semi::ty::{DefId, Flexivity, GenericId, HoleId, Ty, TyCtxt, TyKind};
use crate::surface::{self, Span};
use crate::utils::frecency::Frecency;
use crate::utils::fuzzy::FuzzyIndex;
use crate::utils::string::StringUniqifier;

use super::fulfill::FulfillCtxt;
use super::hir;
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

/// A foreign source block (`extern "rust" [{ ... }];`): opaque Rust items
/// spliced verbatim ahead of every generated FFI wrapper of its file (`use`
/// declarations, helper functions, ...). `body` has the `[{`/`}]` stripped
/// down to the braces; the braces themselves are dropped when splicing.
#[derive(Clone, Debug)]
pub struct FfiPrelude {
    pub body: String,
    pub span: Option<Span>,
    pub file: FileId,
}

/// The foreign body of an `#[ffi(import)]` function: an opaque Rust block
/// (braces included) evaluated with the function's parameters in scope,
/// spliced into a generated boundary wrapper per monomorphized instance.
#[derive(Clone, Debug)]
pub struct FfiImport {
    pub body: String,
    pub span: Option<Span>,
    pub file: FileId,
}

/// A validated `#[ffi(...)]` attribute.
enum FfiSpec {
    /// `#[ffi(import)]` — a function implemented by its foreign body.
    Import,
    /// `#[ffi(rust = "path")]` — an opaque record aliasing the Rust type
    /// at `path` (a `#[repr(transparent)]` wrapper over `reussir_rt`'s `Rc`).
    Opaque(String),
}

/// One `import` binding: within `file`, reference paths headed by `name`
/// expand through `path` (see [`Elaborator::expand_path`] and
/// `docs/design/import.md`).
#[derive(Clone, Debug)]
pub struct ImportBinding {
    pub file: FileId,
    pub name: TokenKey,
    pub path: surface::Path,
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
    /// `(name, type, is_mutable, visibility)`.
    Named(Vec<(TokenKey, Ty<'tcx>, bool, surface::Visibility)>),
    /// `(type, is_mutable, visibility)`.
    Unnamed(Vec<(Ty<'tcx>, bool, surface::Visibility)>),
    Variants(Vec<Variant<'tcx>>),
    /// An opaque `#[ffi]` record: no fields, no constructors — the value is
    /// a foreign rc box whose payload only the foreign side interprets.
    Opaque,
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
    /// Whether consumer source outside the defining package may name this
    /// record. Inside a package nothing is hidden (the whole package is one
    /// unit); the marker governs the package interface (`--emit rri`) and the
    /// private-in-public discipline on `pub` surfaces.
    pub visibility: surface::Visibility,
    pub ty_params: Vec<(TokenKey, GenericId)>,
    pub kind: surface::RecordKind,
    pub default_cap: DefaultCap,
    /// `#[repr(fixed)]`: uniform max-arm box sizing for this enum instead of the
    /// default per-constructor sizing. Only ever set for enums.
    pub repr_fixed: bool,
    /// `#[ffi(rust = "path")]`: the Rust type path this opaque record
    /// aliases. Set exactly when `fields` is [`RecordFields::Opaque`].
    pub ffi: Option<String>,
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
    /// For an impl member, the module of the block that declared it — the
    /// scope its body checks under. `None` for plain functions (their scope
    /// derives from the declared path). Rust semantics: an impl may appear
    /// anywhere in the defining package, and its members see exactly what
    /// that module sees — private fields only when the block sits in (or
    /// under) the type's defining module.
    pub impl_module: Option<Vec<TokenKey>>,
    /// For an impl member, the target type in its unrefined coloring — the
    /// `Self` alias the body's type annotations resolve to. `None` for plain
    /// functions, extern prototypes, and reparsed dumps (printed HIR never
    /// contains `Self`; it is resolved away at elaboration).
    pub self_ty: Option<Ty<'tcx>>,
    pub is_regional: bool,
    pub span: Option<Span>,
    /// The file the function is declared in (spans index it).
    pub file: FileId,
}

/// Collects the private records `ty`'s structure mentions — nominal heads and
/// their arguments, through every structural former — for the
/// private-in-public check. Deduplicated so each offender reports once per
/// surface.
fn collect_private_records<'tcx>(
    records: &FxHashMap<DefId, Record<'tcx>>,
    ty: Ty<'tcx>,
    out: &mut Vec<DefId>,
) {
    match *ty.kind() {
        TyKind::Record { def, args, .. } => {
            if records
                .get(&def)
                .is_some_and(|r| r.visibility == surface::Visibility::Private)
                && !out.contains(&def)
            {
                out.push(def);
            }
            for &a in args {
                collect_private_records(records, a, out);
            }
        }
        TyKind::Closure { params, ret } => {
            for &p in params {
                collect_private_records(records, p, out);
            }
            collect_private_records(records, ret, out);
        }
        TyKind::Array { elem, .. } | TyKind::Cell { elem, .. } => {
            collect_private_records(records, elem, out);
        }
        TyKind::Nullable(inner) | TyKind::Arc(inner) => {
            collect_private_records(records, inner, out);
        }
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Char
        | TyKind::Unit
        | TyKind::Generic(_)
        | TyKind::Hole(_)
        | TyKind::Bottom => {}
    }
}

/// Records the generic at the head of a `[flex]`-annotated type (peeling
/// `Nullable`): a `[flex]` use of a bare generic means that generic must be
/// instantiated with a regional record.
/// The impl-block context a member is scanned under: the resolved target
/// record and the block-level generics its signature shares.
pub(super) struct ImplCtx<'x, 'tcx> {
    pub target: DefId,
    pub generics: &'x [(TokenKey, GenericId)],
    /// The applied target type — what `Self` (and a receiver annotation)
    /// means inside the block, in the record's unrefined coloring.
    pub self_ty: Ty<'tcx>,
    /// For a *trait* impl member: the implemented trait (declaration goes
    /// under `[impl-module.., trait-path.., head-name, method]`, visibility
    /// is the trait's, and the clash diagnostic names the trait).
    pub trait_of: Option<TraitId>,
}

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
    pub lang: crate::semi::lang::LangItems,
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
    /// Foreign source blocks (`extern "rust" [{ ... }];`), in source order.
    pub ffi_preludes: Vec<FfiPrelude>,
    /// Foreign bodies of `#[ffi(import)]` functions, keyed by their def.
    pub ffi_imports: FxHashMap<DefId, FfiImport>,
    /// File-scoped `import` path bindings, in declaration order (see
    /// `docs/design/import.md`). Append-only so checkpoint/rollback
    /// restores it by truncation; lookups scan backwards filtered by the
    /// current file — files hold a handful of bindings at most.
    pub imports: Vec<ImportBinding>,
    pub generics: Vec<GenericInfo>,
    pub strings: StringUniqifier,
    /// Heads of loaded extern packages ([`Elaborator::declare_extern_package`]):
    /// a reference path led by one resolves in that package's table only —
    /// never module-relatively — so an extern head and a local module cannot
    /// shadow one another.
    pub extern_heads: FxHashSet<TokenKey>,
    /// Extern-owned defs and the package head that owns each. The `pub` gate
    /// keys on this, and the HIR dump excludes these
    /// ([`Elaborator::local_records`]) — the dump describes the consumer
    /// package only.
    pub extern_defs: FxHashMap<DefId, TokenKey>,
    /// Extern functions remapped into the consumer's spaces — generic bodies
    /// to instantiate, ground prototypes to declare against — consumed by
    /// cross-package monomorphization ([`MonoInput::externs`]). Never part
    /// of the printed program or the export surface.
    ///
    /// [`MonoInput::externs`]: crate::full::mono::MonoInput::externs
    pub extern_functions: Vec<Function<'tcx>>,
    /// String literals referenced by extern bodies. Tokens are
    /// content-addressed (BLAKE3), so entries carry over without remapping.
    pub extern_strings: Vec<(crate::utils::string::StringToken, String)>,
    /// Foreign bodies of extern `#[ffi(import)]` functions, keyed by their
    /// consumer defs, with the preludes of their files.
    pub extern_ffi_imports: FxHashMap<DefId, FfiImport>,
    pub extern_ffi_preludes: Vec<FfiPrelude>,
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
    /// The type `Self` resolves to while an impl member's signature or body
    /// is being evaluated. Per-item working state like `generic_names`, NOT
    /// accumulated — reset by `enter_function` and after signature eval, so
    /// no Checkpoint entry.
    pub(super) self_alias: Option<Ty<'tcx>>,
    /// The interned spelling of `Self` — the display name given to each user
    /// trait's implicit `Self` generic (interned once at construction; the
    /// scan passes have no interner).
    pub(super) self_name: TokenKey,
    /// The prelude's name → lang-item table (`PartialEq` … `Ordering`),
    /// interned at construction like `self_name`. Bare-name resolution
    /// falls back through it ([`Elaborator::prelude_lookup`]), so items
    /// `core` declares under its own modules still resolve unqualified;
    /// the fallback registration ([`Elaborator::ensure_cmp_fallback`])
    /// draws its declared names from the same table.
    pub(super) prelude_names: [(TokenKey, crate::semi::lang::LangItem); 10],
    /// The operator-method names (`eq`, `lt`, `le`, `gt`, `ge`, `cmp`),
    /// interned at construction: operator desugaring and the builtin-impl
    /// rewrites resolve the active trait's method *by name*, so method
    /// order in `core`'s declarations is not a compiler contract.
    pub(super) cmp_method_names: [TokenKey; 6],
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
    pub(super) traits: usize,
    pub(super) impls: usize,
    pub(super) trampolines: usize,
    pub(super) transform_anchors: usize,
    pub(super) transform_scripts: usize,
    pub(super) ffi_preludes: usize,
    pub(super) imports: usize,
    pub(super) elaborated: usize,
    pub(super) reports: usize,
    pub(super) lang: usize,
    /// `Num`'s super-trait count — the one mutation a batch makes to a
    /// *pre-existing* trait def (the `Num ⊴ PartialOrd` edge,
    /// [`Elaborator::ensure_num_partial_ord_edge`]); truncating restores it.
    pub(super) num_supers: usize,
    /// The wired tower fields ([`crate::semi::lang::LangItems::fields`]) —
    /// a batch that binds `#[lang("num")]` repoints them, so rollback must
    /// restore the builtins.
    pub(super) lang_fields: [TraitId; 5],
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'tcx>, interner: &'a reussir_syntax::SessionInterner) -> Self {
        // Builtin trait defs occupy the first DefIds, declared before any
        // user or extern item and before the first REPL checkpoint — so
        // rollback can never retract them. `Builtins::register` is generic
        // over `&mut impl Interner` (it also serves owned tables, which
        // genuinely mutate); the session table satisfies that spelling by
        // a handle copy — an `Arc` clone, not a table copy.
        let mut defs = DefTable::new();
        let mut traits = TraitDb::new();
        let mut handle = interner.clone();
        let fallback = Builtins::register(&mut traits, &mut defs, &mut handle, tcx);
        let lang = crate::semi::lang::LangItems::new(fallback);
        let self_name = interner.intern("Self");
        let prelude_names = {
            use crate::semi::lang::LangItem;
            [
                ("PartialEq", LangItem::PartialEq),
                ("Eq", LangItem::Eq),
                ("PartialOrd", LangItem::PartialOrd),
                ("Ord", LangItem::Ord),
                ("Ordering", LangItem::Ordering),
                ("Num", LangItem::Num),
                ("Integral", LangItem::Integral),
                ("FloatingPoint", LangItem::FloatingPoint),
                ("PtrLike", LangItem::PtrLike),
                ("Sync", LangItem::Sync),
            ]
            .map(|(name, item)| (interner.intern(name), item))
        };
        let cmp_method_names =
            ["eq", "lt", "le", "gt", "ge", "cmp"].map(|name| interner.intern(name));
        Elaborator {
            tcx,
            resolver: &**interner,
            defs,
            frecency: Frecency::new(),
            traits,
            lang,
            trampolines: Vec::new(),
            transform_anchors: Vec::new(),
            transform_scripts: Vec::new(),
            ffi_preludes: Vec::new(),
            ffi_imports: FxHashMap::default(),
            imports: Vec::new(),
            records: FxHashMap::default(),
            functions: FxHashMap::default(),
            generics: Vec::new(),
            strings: StringUniqifier::new(),
            extern_heads: FxHashSet::default(),
            extern_defs: FxHashMap::default(),
            extern_functions: Vec::new(),
            extern_strings: Vec::new(),
            extern_ffi_imports: FxHashMap::default(),
            extern_ffi_preludes: Vec::new(),
            elaborated: Vec::new(),
            reports: Vec::new(),
            current_file: FileId::ROOT,
            records_complete: false,
            infer: InferCtxt::new(tcx),
            vars: VarEnv::default(),
            generic_names: FxHashMap::default(),
            self_alias: None,
            self_name,
            prelude_names,
            cmp_method_names,
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

    /// The private-in-public discipline, scoped to records: the declared
    /// surface of a `pub` item — a `pub` function's signature, a `pub`
    /// record's fields — may only mention `pub` records. Within a package
    /// nothing is hidden, but the package interface (`--emit rri`) ships
    /// exactly the `pub` surface as nameable, so a private record there would
    /// be unusable at every foreign point of use.
    fn check_private_in_public(&mut self, tys: &[Ty<'tcx>], item: &str, span: Option<Span>) {
        let mut private = Vec::new();
        for &t in tys {
            collect_private_records(&self.records, t, &mut private);
        }
        for def in private {
            let path = self.defs.path(def).display(self.resolver);
            self.error(
                span,
                format!(
                    "private record `{path}` in the public interface of {item}; \
                     mark the record `pub` or make the item private"
                ),
            );
        }
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
            TyKind::Cell { elem, kind } => {
                out.push_str(kind.surface_name());
                out.push('<');
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

    /// Declare one `import` binding for `file`. Rejects `pub`
    /// (bindings are file-private), the reserved path heads (`core`, `root`,
    /// `super`), and a duplicate name within the same file.
    pub(super) fn declare_import(
        &mut self,
        decl: &surface::ImportDecl,
        span: Option<Span>,
        file: FileId,
    ) {
        if decl.visibility == surface::Visibility::Public {
            self.error(
                span,
                "imports are file-scoped and cannot be `pub`; re-exporting is not supported",
            );
        }
        let name = decl.name.value;
        let shown = self.sym(name);
        if matches!(shown, "core" | "root" | "super") {
            self.error(
                Some(decl.name.span()),
                format!("`{shown}` is a reserved path head and cannot be bound by an import"),
            );
            return;
        }
        // Extern package heads are reserved the same way: a binding shadowing
        // one would silently reroute every `dep::…` reference in the file.
        if self.extern_heads.contains(&name) {
            self.error(
                Some(decl.name.span()),
                format!("`{shown}` names a loaded extern package and cannot be bound by an import"),
            );
            return;
        }
        if self
            .imports
            .iter()
            .any(|b| b.file == file && b.name == name)
        {
            self.error(
                Some(decl.name.span()),
                format!("`{shown}` is already bound by an earlier import in this file"),
            );
            return;
        }
        self.imports.push(ImportBinding {
            file,
            name,
            path: decl.path.clone(),
        });
    }

    /// Expand a reference path through the current file's `import` bindings:
    /// a bound *first segment* is replaced by the binding's target path; a
    /// bound *bare name* is replaced by the target outright. Expansion
    /// iterates so a binding may target another binding; a self-referential chain
    /// abandons expansion entirely (returns `None`), so the reference fails
    /// resolution with the ordinary diagnostic on the path *as written*.
    /// Returns `None` when nothing applies, so callers keep borrowing the
    /// original path in the common case.
    pub(super) fn expand_path(&self, path: &surface::Path) -> Option<surface::Path> {
        let binding = |name: TokenKey| {
            self.imports
                .iter()
                .rev()
                .find(|b| b.file == self.current_file && b.name == name)
                .map(|b| &b.path)
        };
        let mut expanded: Option<surface::Path> = None;
        let mut visited: smallvec::SmallVec<[TokenKey; 4]> = smallvec::SmallVec::new();
        loop {
            let p = expanded.as_ref().unwrap_or(path);
            let head = p.segments.first().copied().unwrap_or(p.basename);
            let Some(target) = binding(head) else { break };
            if visited.contains(&head) {
                return None;
            }
            visited.push(head);
            expanded = Some(if p.segments.is_empty() {
                target.clone()
            } else {
                let mut segments = target.segments.clone();
                segments.push(target.basename);
                segments.extend(p.segments[1..].iter().copied());
                surface::Path {
                    basename: p.basename,
                    segments,
                }
            });
        }
        expanded
    }

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

    /// The prelude: a bare name's lang-item def, kind-filtered (`Trait`
    /// for bound positions, `Record` for type and constructor positions).
    /// This is the resolution fallback that makes `core`'s declarations —
    /// which live under core's own modules — usable unqualified, exactly
    /// like the crate-root fallback tower they replace.
    pub(super) fn prelude_lookup(
        &self,
        name: TokenKey,
        kind: crate::semi::lang::LangItemKind,
    ) -> Option<DefId> {
        self.prelude_names
            .iter()
            .find(|&&(key, item)| key == name && item.kind() == kind)
            .and_then(|&(_, item)| self.lang.get(item))
    }

    /// Resolve a type reference: bare names in the current module (falling
    /// back to the crate root), qualified paths per the module-relative rules
    /// (see [`DefTable::resolve_record_path`]). The path is first expanded
    /// through the file's `import` bindings ([`Self::expand_path`]); one led
    /// by an extern package head resolves in that package's table only
    /// ([`Self::resolve_extern`]).
    pub(super) fn resolve_record_ref(&self, path: &surface::Path) -> Option<DefId> {
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        if self.extern_head(path).is_some() {
            return self.resolve_extern(path, DefKind::Record);
        }
        if path.segments.is_empty() {
            self.defs.resolve_record(path.basename).or_else(|| {
                self.prelude_lookup(path.basename, crate::semi::lang::LangItemKind::Record)
            })
        } else {
            self.defs
                .resolve_record_path(&self.classify_segs(path), path.basename)
        }
    }

    /// Resolve a value (function) reference; see [`Self::resolve_record_ref`].
    pub(super) fn resolve_function_ref(&self, path: &surface::Path) -> Option<DefId> {
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        if self.extern_head(path).is_some() {
            return self.resolve_extern(path, DefKind::Function);
        }
        if path.segments.is_empty() {
            self.defs.resolve_function(path.basename)
        } else {
            self.defs
                .resolve_function_path(&self.classify_segs(path), path.basename)
        }
    }

    /// Resolve a constructor path's *qualifier* as a record: for
    /// `m::Enum::Variant` the qualifier is `m::Enum` (the basename names the
    /// variant). `None` when the path has no qualifier. Imports expand first,
    /// so `L::Cons` with `import pkg::List as L` names `pkg::List`'s variant.
    pub(super) fn resolve_ctor_qualifier(&self, path: &surface::Path) -> Option<DefId> {
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        let (&enum_name, mods) = path.segments.split_last()?;
        if mods.is_empty() {
            self.defs
                .resolve_record(enum_name)
                .or_else(|| self.prelude_lookup(enum_name, crate::semi::lang::LangItemKind::Record))
        } else {
            let mods_path = surface::Path {
                basename: enum_name,
                segments: mods.iter().copied().collect(),
            };
            if self.extern_head(&mods_path).is_some() {
                return self.resolve_extern(&mods_path, DefKind::Record);
            }
            self.defs
                .resolve_record_path(&self.classify_segs(&mods_path), enum_name)
        }
    }

    // ----- extern-package resolution -----

    /// The path's first segment when it names a loaded extern package.
    fn extern_head(&self, path: &surface::Path) -> Option<TokenKey> {
        path.segments
            .first()
            .copied()
            .filter(|k| self.extern_heads.contains(k))
    }

    /// Resolve a reference into a loaded extern package: the path is looked
    /// up **absolute** in that package's namespace — the module-relative and
    /// root-relative fallbacks never apply. Only `pub` items resolve; a
    /// private hit yields `None` here and the targeted diagnostic through
    /// [`Self::extern_private_msg`] on the caller's failure path.
    fn resolve_extern(&self, path: &surface::Path, kind: DefKind) -> Option<DefId> {
        let def = self.lookup_extern(path, kind)?;
        (self.item_visibility(def)? == surface::Visibility::Public).then_some(def)
    }

    /// Exact-path lookup for an extern-headed reference (no access check).
    fn lookup_extern(&self, path: &surface::Path, kind: DefKind) -> Option<DefId> {
        let key: Vec<TokenKey> = path
            .segments
            .iter()
            .copied()
            .chain([path.basename])
            .collect();
        match kind {
            DefKind::Record => self.defs.lookup_record(&key),
            DefKind::Function => self.defs.lookup_function(&key),
            DefKind::Trait => self.defs.lookup_trait(&key),
        }
    }

    /// A declared item's visibility, from whichever table holds it. `None`
    /// for a def with no item entry (a path-only handle).
    fn item_visibility(&self, def: DefId) -> Option<surface::Visibility> {
        match self.defs.info(def).kind {
            DefKind::Record => self.records.get(&def).map(|r| r.visibility),
            DefKind::Function => self.functions.get(&def).map(|f| f.visibility),
            DefKind::Trait => self
                .traits
                .trait_by_def(def)
                .map(|id| self.traits.trait_def(id).visibility),
        }
    }

    /// The access-control diagnostic for a failed reference that names a
    /// *private* item of a loaded extern package; `None` for every other
    /// failure, so the caller keeps its ordinary not-found report and a
    /// private hit stays distinguishable from a missing one.
    pub(super) fn extern_private_msg(&self, path: &surface::Path, kind: DefKind) -> Option<String> {
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        let head = self.extern_head(path)?;
        let def = self.lookup_extern(path, kind)?;
        if self.item_visibility(def)? == surface::Visibility::Public {
            return None;
        }
        let what = match kind {
            DefKind::Record => "record",
            DefKind::Function => "function",
            DefKind::Trait => "trait",
        };
        Some(format!(
            "{what} `{}` in package `{}` is private",
            self.sym(path.basename),
            self.sym(head)
        ))
    }

    /// [`Self::extern_private_msg`] over a constructor path's *qualifier*
    /// (`dep::Enum::Variant` names the record `dep::Enum`).
    pub(super) fn extern_private_ctor_msg(&self, path: &surface::Path) -> Option<String> {
        let (&enum_name, mods) = path.segments.split_last()?;
        let qualifier = surface::Path {
            basename: enum_name,
            segments: mods.iter().copied().collect(),
        };
        self.extern_private_msg(&qualifier, DefKind::Record)
    }

    /// The consumer package's own records — extern-imported ones excluded —
    /// for printing: the HIR/rri dumps describe this package only.
    pub fn local_records(&self) -> FxHashMap<DefId, Record<'tcx>> {
        self.records
            .iter()
            .filter(|(def, _)| !self.extern_defs.contains_key(def))
            .map(|(def, rec)| (*def, rec.clone()))
            .collect()
    }

    /// Whether `prefix` spells the canonical `core::intrinsic::<family>`
    /// module of a prelude builtin. The package name `core` is reserved, so
    /// the qualified spelling can never collide with user code.
    pub(super) fn is_core_intrinsic_prefix(&self, prefix: &[TokenKey], family: &str) -> bool {
        prefix.len() == 3
            && self.sym(prefix[0]) == "core"
            && self.sym(prefix[1]) == "intrinsic"
            && self.sym(prefix[2]) == family
    }

    /// Whether a constructor path whose *qualifier* names a prelude builtin
    /// (`Nullable::…`, `Arc::Variant`) targets that builtin: the canonical
    /// `core::intrinsic::<family>::<Name>::…` spelling, or the prelude's
    /// bare `<Name>::…` when no user record shadows the name (a shadowing
    /// definition or import dispatches through ordinary enum resolution
    /// instead).
    pub(super) fn ctor_qualifier_prelude_applies(
        &self,
        path: &surface::Path,
        family: &str,
    ) -> bool {
        let Some((_, prefix)) = path.segments.split_last() else {
            return false;
        };
        if prefix.is_empty() {
            self.resolve_ctor_qualifier(path).is_none()
        } else {
            self.is_core_intrinsic_prefix(prefix, family)
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
        self.fuzzy_hint(
            name,
            self.defs
                .trait_names()
                .map(|k| (self.sym(k), self.frecency.score(k))),
        )
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

    /// Every bounded generic's bounds, for the HIR printer's binder
    /// serialization (`$0 (T): Num`). Derived data — no new elaborator
    /// state. Segments come from the trait's qualified path: builtin traits
    /// live at the crate root, so their serialization stays one segment.
    pub fn bound_names(&self) -> FxHashMap<GenericId, Vec<hir::Bound>> {
        self.generics
            .iter()
            .enumerate()
            .filter(|(_, info)| !info.bounds.is_empty())
            .map(|(i, info)| {
                let bounds = info
                    .bounds
                    .iter()
                    .map(|&t| {
                        let def = self.traits.trait_def(t).def;
                        let segments = self
                            .defs
                            .path(def)
                            .0
                            .iter()
                            .map(|&seg| self.sym(seg).to_owned())
                            .collect();
                        hir::Bound::Trait(hir::BoundPath { segments })
                    })
                    .collect();
                (GenericId(i as u32), bounds)
            })
            .collect()
    }

    /// A trait's display name — its qualified path.
    pub(super) fn trait_display(&self, id: TraitId) -> String {
        self.defs
            .path(self.traits.trait_def(id).def)
            .display(self.resolver)
    }

    /// Resolve a bound path to a trait, through the trait namespace: extern
    /// heads gate on `pub`, bare names see the current module then the crate
    /// root (where the builtins live), qualified paths resolve like any
    /// other reference.
    /// Resolve a (possibly qualified) trait reference without reporting —
    /// shared by [`Self::resolve_bound`] (which diagnoses on `None`) and
    /// trait-path method calls (which fall back to other interpretations).
    /// The path must already be import-expanded.
    pub(super) fn resolve_trait_ref_quiet(&mut self, path: &surface::Path) -> Option<DefId> {
        if self.extern_head(path).is_some() {
            self.resolve_extern(path, DefKind::Trait)
        } else if path.segments.is_empty() {
            self.defs
                .resolve_trait(path.basename)
                .map(|def| self.lang.override_builtin(def, &self.traits))
                .or_else(|| {
                    self.prelude_lookup(path.basename, crate::semi::lang::LangItemKind::Trait)
                })
        } else {
            self.defs
                .resolve_trait_path(&self.classify_segs(path), path.basename)
        }
    }

    pub fn resolve_bound(&mut self, path: &surface::Path, span: Option<Span>) -> Option<TraitId> {
        let expanded = self.expand_path(path);
        let path = expanded.as_ref().unwrap_or(path);
        let def = self.resolve_trait_ref_quiet(path);
        let Some(def) = def else {
            if let Some(msg) = self.extern_private_msg(path, DefKind::Trait) {
                self.error(span, msg);
                return None;
            }
            let hint = if path.segments.is_empty() {
                self.trait_suggestion(self.sym(path.basename))
            } else {
                String::new()
            };
            let shown = self.path_display(path);
            self.error(span, format!("unknown trait bound `{shown}`{hint}"));
            return None;
        };
        self.record_use(path.basename);
        Some(
            self.traits
                .trait_by_def(def)
                .expect("every trait-namespace def has a TraitDb entry"),
        )
    }

    /// Resolve a serialized bound's path segments (interned) — the `.rri`
    /// reload side. Absolute lookup: a single segment is a crate-root name,
    /// exactly where the builtins live.
    pub(super) fn resolve_bound_segments(
        &mut self,
        segments: &[TokenKey],
        span: Option<Span>,
    ) -> Option<TraitId> {
        let def = self.defs.lookup_trait(segments).map(|def| match segments {
            [_] => self.lang.override_builtin(def, &self.traits),
            _ => def,
        });
        let Some(def) = def else {
            let shown = segments
                .iter()
                .map(|&s| self.sym(s))
                .collect::<Vec<_>>()
                .join("::");
            let hint = if segments.len() == 1 {
                self.trait_suggestion(self.sym(segments[0]))
            } else {
                String::new()
            };
            self.error(span, format!("unknown trait bound `{shown}`{hint}"));
            return None;
        };
        Some(
            self.traits
                .trait_by_def(def)
                .expect("every trait-namespace def has a TraitDb entry"),
        )
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
        self.self_alias = None;
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
    /// Any *future* accumulated state must be added here. The trait registry
    /// is covered: `traits`/`impls` counters truncate the [`TraitDb`] tails
    /// (dense ids), so a rejected batch's trait declarations and impls
    /// retract with their defs.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            defs: self.defs.len(),
            generics: self.generics.len(),
            traits: self.traits.traits_len(),
            impls: self.traits.impls_len(),
            trampolines: self.trampolines.len(),
            transform_anchors: self.transform_anchors.len(),
            transform_scripts: self.transform_scripts.len(),
            ffi_preludes: self.ffi_preludes.len(),
            imports: self.imports.len(),
            elaborated: self.elaborated.len(),
            lang: self.lang.len(),
            reports: self.reports.len(),
            num_supers: self.traits.trait_def(self.lang.num).supertraits.len(),
            lang_fields: self.lang.fields(),
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
        self.ffi_imports.retain(|def, _| def.0 < live);
        self.generics.truncate(cp.generics);
        self.traits.truncate(cp.traits, cp.impls);
        self.trampolines.truncate(cp.trampolines);
        self.transform_anchors.truncate(cp.transform_anchors);
        self.transform_scripts.truncate(cp.transform_scripts);
        self.ffi_preludes.truncate(cp.ffi_preludes);
        self.imports.truncate(cp.imports);
        self.elaborated.truncate(cp.elaborated);
        self.lang.truncate(cp.lang);
        self.reports.truncate(cp.reports);
        self.lang.restore_fields(cp.lang_fields);
        self.traits
            .trait_def_mut(self.lang.num)
            .supertraits
            .truncate(cp.num_supers);
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
    /// The `#[lang("…")]` marker on an item, validated: one string
    /// argument naming a compiler-known lang item.
    fn lang_attr(
        &mut self,
        attrs: &[surface::Attribute],
        span: Option<Span>,
    ) -> Option<crate::semi::lang::LangItem> {
        let attr = attrs.iter().find(|a| self.sym(a.name) == "lang")?;
        let [name] = attr.literals.as_slice() else {
            self.error(
                span,
                "`#[lang]` takes exactly one string argument, e.g. `#[lang(\"partial_eq\")]`",
            );
            return None;
        };
        let Some(item) = crate::semi::lang::LangItem::from_name(name) else {
            self.error(span, format!("unknown lang item `{name}`"));
            return None;
        };
        Some(item)
    }

    /// A built-in type former's anchor (`#[lang("arc")] pub struct
    /// Arc<T> { }`): a declaration *site* for the wired former — one type
    /// parameter, no fields. The former itself keeps resolving by
    /// spelling; the anchor is never a nominal type.
    fn validate_former_anchor(
        &mut self,
        rec: &surface::Record,
        former: crate::semi::lang::FormerItem,
        span: Option<Span>,
    ) {
        let empty = matches!(
            &rec.fields,
            surface::RecordFields::Named(fields) if fields.is_empty()
        );
        if rec.kind != surface::RecordKind::StructKind || !empty || rec.ty_params.len() != 1 {
            self.error(
                span,
                format!(
                    "a builtin-type anchor must be an empty struct with one type parameter: `#[lang(\"{}\")]` binds the former's declaration site",
                    former.name()
                ),
            );
        }
    }

    /// The `#[sealed]` marker on a trait: no user impls (the compiler
    /// provides them, as with the numeric tower). Bare — any argument is a
    /// diagnostic.
    fn sealed_attr(&mut self, attrs: &[surface::Attribute], span: Option<Span>) -> bool {
        let Some(attr) = attrs.iter().find(|a| self.sym(a.name) == "sealed") else {
            return false;
        };
        if !attr.literals.is_empty() {
            self.error(span, "`#[sealed]` takes no arguments");
        }
        true
    }

    /// Reject `#[sealed]` on anything but a trait declaration.
    fn reject_sealed_attr(&mut self, attrs: &[surface::Attribute], span: Option<Span>) {
        if attrs.iter().any(|a| self.sym(a.name) == "sealed") {
            self.error(span, "only a trait can be `#[sealed]`");
        }
    }

    /// Bind a scanned declaration to its lang item: kind check, then a
    /// duplicate check against the registry (first declaration wins).
    pub(super) fn declare_lang(
        &mut self,
        item: crate::semi::lang::LangItem,
        def: DefId,
        actual: crate::semi::lang::LangItemKind,
        span: Option<Span>,
    ) {
        use crate::semi::lang::LangItemKind;
        if item.kind() != actual {
            let want = match item.kind() {
                LangItemKind::Trait => "a trait",
                LangItemKind::Record => "a struct or enum",
                LangItemKind::Function => "a function prototype",
            };
            self.error(span, format!("lang item `{}` must be {want}", item.name()));
            return;
        }
        if let Err(prior) = self.lang.declare(item, def) {
            let shown = self.defs.path(prior).display(self.resolver);
            self.error(
                span,
                format!(
                    "lang item `{}` is already declared by `{shown}`",
                    item.name()
                ),
            );
            return;
        }
        // A declared numeric/marker-tower trait takes over its wired field.
        if crate::semi::lang::LangItem::TOWER.contains(&item)
            && let Some(id) = self.traits.trait_by_def(def)
        {
            self.lang.repoint(item, id);
        }
    }

    /// The declaration site of a lang item, for diagnostics, debug info,
    /// and the LSP: the file and span of whichever declaration the
    /// registry holds. `None` for fallback-registered items — they have no
    /// source until their declarations live in `core`.
    pub fn lang_item_site(
        &self,
        item: crate::semi::lang::LangItem,
    ) -> Option<(FileId, Option<Span>)> {
        let def = self.lang.get(item)?;
        if let Some(tid) = self.traits.trait_by_def(def) {
            let t = self.traits.trait_def(tid);
            return Some((t.file, t.span));
        }
        if let Some(rec) = self.records.get(&def) {
            return Some((rec.file, rec.span));
        }
        self.functions.get(&def).map(|f| (f.file, f.span))
    }

    /// Ensure the comparison tower's *traits* exist: unsealed, method-less
    /// `PartialEq`/`Eq`/`PartialOrd`/`Ord` at the crate root. Sessions
    /// without `core` — unit tests, bare `rrc file.rr`, the REPL today —
    /// keep every comparison operator working: the checker constrains
    /// against these traits, and monomorphization lowers selections of
    /// their method-less impls ([`Elaborator::ensure_cmp_scalar_impls`])
    /// straight to the intrinsic comparison. Implementing them for user
    /// types needs `core`'s real, method-carrying declarations.
    ///
    /// Idempotent per item, so a tower an interface declared fully is left
    /// alone and a partially bound one is completed. Runs *before* the
    /// batch's trait scan: a root-name collision is then always the user
    /// declaration's error, exactly as with the numeric tower.
    pub(super) fn ensure_cmp_fallback(&mut self) {
        use crate::semi::lang::LangItem;

        // Resolve-or-declare the tower's traits: a declared item (a loaded
        // `core` interface) wins; the fallback is a method-less stand-in
        // at the crate root.
        let module = self.defs.module().to_vec();
        self.defs.set_module(Vec::new());
        let [pe_key, eq_key, po_key, ord_key, ..] = self.prelude_names.map(|(key, _)| key);
        // `None` from the chain means a prior batch squatted a tower name
        // without binding the lang item; the remaining traits stay absent
        // and comparisons degrade to the numeric constraint.
        let _ = (|| {
            let pe = self.fallback_cmp_trait(LangItem::PartialEq, pe_key, &[])?;
            let eq = self.fallback_cmp_trait(LangItem::Eq, eq_key, &[pe])?;
            let po = self.fallback_cmp_trait(LangItem::PartialOrd, po_key, &[pe])?;
            self.fallback_cmp_trait(LangItem::Ord, ord_key, &[eq, po])
        })();
        self.defs.set_module(module);
    }

    /// Complete the tower's method-less scalar impls, for *whichever*
    /// tower is active — `core` declares no scalar impls (an `impl` target
    /// must be a named type), so every session completes its own here and
    /// monomorphization lowers their selections intrinsically.
    /// `PartialEq`/`PartialOrd` cover every scalar *directly* — super-trait
    /// projection would satisfy the bounds, but method dispatch needs a
    /// direct impl — while `Ord` covers only the totally ordered ones (NaN
    /// keeps the floats out of `Eq`/`Ord`; the super-trait closure answers
    /// `Eq`, which is method-less everywhere and needs no impls). Runs
    /// after the trait-stub scan so a tower the batch itself declares —
    /// `core` compiling itself, a hand-declared REPL tower — completes the
    /// same way as the fallback's or a loaded interface's.
    pub(super) fn ensure_cmp_scalar_impls(&mut self) {
        use crate::semi::lang::LangItem;
        use crate::semi::ty::{FpTy, IntTy};
        let resolve = |el: &Self, item| {
            el.lang
                .get(item)
                .and_then(|def| el.traits.trait_by_def(def))
        };
        let (Some(pe), Some(po), Some(ord)) = (
            resolve(self, LangItem::PartialEq),
            resolve(self, LangItem::PartialOrd),
            resolve(self, LangItem::Ord),
        ) else {
            return;
        };
        let ordered: Vec<Ty<'tcx>> = [8u16, 16, 32, 64]
            .into_iter()
            .flat_map(|width| [IntTy::Signed(width), IntTy::Unsigned(width)])
            .map(|int| self.tcx.mk_int(int))
            .chain([self.tcx.mk_bool(), self.tcx.mk_char()])
            .collect();
        let floats: Vec<Ty<'tcx>> = [
            FpTy::Ieee(16),
            FpTy::Ieee(32),
            FpTy::Ieee(64),
            FpTy::BFloat16,
            FpTy::Float8,
        ]
        .into_iter()
        .map(|fp| self.tcx.mk_fp(fp))
        .collect();
        let all = || ordered.iter().chain(&floats).copied();
        // The numeric tower needs the same completion: the construction
        // impls target the *builtin* ids, so once `core`'s declarations
        // repoint the wired fields, `T: Integral` obligations select
        // against the declared traits — which get their own method-less
        // compiler impls here (idempotent; a no-op in plain sessions).
        let ints = || ordered.iter().take(8).copied();
        all()
            .map(|ty| (pe, ty))
            .chain(all().map(|ty| (po, ty)))
            .chain(ordered.iter().copied().map(|ty| (ord, ty)))
            .chain(ints().map(|ty| (self.lang.integral, ty)))
            .chain(
                floats
                    .iter()
                    .copied()
                    .map(|ty| (self.lang.floating_point, ty)),
            )
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(trait_id, self_ty)| self.ensure_scalar_impl(trait_id, self_ty));
    }

    /// Resolve one comparison-tower item to its trait id, declaring the
    /// method-less fallback trait when nothing has bound the item yet.
    /// `None` when the root name is taken by a declaration that is not the
    /// lang item.
    fn fallback_cmp_trait(
        &mut self,
        item: crate::semi::lang::LangItem,
        name: TokenKey,
        supers: &[TraitId],
    ) -> Option<TraitId> {
        use crate::semi::traits::def::TraitDef;
        if let Some(def) = self.lang.get(item) {
            let id = self
                .traits
                .trait_by_def(def)
                .expect("a trait-kinded lang item");
            return Some(id);
        }
        let def = self.defs.declare_trait(name)?;
        let self_g = GenericId(0);
        let id = TraitId(self.traits.traits_len() as u32);
        self.traits.add_trait(TraitDef {
            id,
            def,
            visibility: surface::Visibility::Public,
            sealed: false,
            self_param: self_g,
            params: Vec::new(),
            supertraits: supers
                .iter()
                .map(|&trait_id| crate::semi::traits::TraitRef {
                    trait_id,
                    args: vec![self.tcx.mk_generic(self_g)],
                })
                .collect(),
            methods: Vec::new(),
            assoc_tys: Vec::new(),
            span: None,
            file: FileId::ROOT,
        });
        self.lang
            .declare(item, def)
            .expect("guarded by the `get` above");
        Some(id)
    }

    /// Register the method-less compiler impl `self_ty : trait_id` unless
    /// one with that head already exists (idempotence across batches; the
    /// probe is exact under REPL rollback, unlike a flag would be).
    fn ensure_scalar_impl(&mut self, trait_id: TraitId, self_ty: Ty<'tcx>) {
        use crate::semi::traits::def::ImplDef;
        let head = crate::semi::traits::coherence::head_key(self_ty).expect("scalars have heads");
        if !self.traits.impls_for(trait_id, head).is_empty() {
            return;
        }
        self.traits.add_impl(ImplDef {
            id: crate::semi::traits::ImplId(self.traits.impls_len() as u32),
            generics: Vec::new(),
            trait_ref: crate::semi::traits::TraitRef {
                trait_id,
                args: vec![self_ty],
            },
            self_ty,
            where_clauses: Vec::new(),
            methods: Vec::new(),
            span: None,
            file: FileId::ROOT,
        });
    }

    /// Push the `Num ⊴ PartialOrd` super-trait edge once `PartialOrd`
    /// exists — every machine numeric is at least partially ordered, which
    /// is what lets `T: Num` code keep comparing. Runs after the trait-stub
    /// scan so the edge targets whichever `PartialOrd` is active: the
    /// fallback's, a loaded interface's, or this very batch's (`core`
    /// compiling itself). Checkpoint-covered by `num_supers`.
    pub(super) fn ensure_num_partial_ord_edge(&mut self) {
        use crate::semi::lang::LangItem;
        let Some(po) = self
            .lang
            .get(LangItem::PartialOrd)
            .and_then(|def| self.traits.trait_by_def(def))
        else {
            return;
        };
        let num = self.lang.num;
        if self
            .traits
            .trait_def(num)
            .supertraits
            .iter()
            .any(|sup| sup.trait_id == po)
        {
            return;
        }
        let self_param = self.traits.trait_def(num).self_param;
        let arg = self.tcx.mk_generic(self_param);
        self.traits
            .trait_def_mut(num)
            .supertraits
            .push(crate::semi::traits::TraitRef {
                trait_id: po,
                args: vec![arg],
            });
    }

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
        let mut impls = Vec::new();
        let mut trait_decls = Vec::new();
        for (file, module, program) in files {
            for stmt in *program {
                let scope = Scope {
                    file: *file,
                    module,
                };
                let span = span_of(stmt);
                let attrs = stmt.attributes();
                self.set_current_file(*file);
                let ffi = self.ffi_attr(&attrs);
                match stmt.kind() {
                    surface::StmtKind::Record(rec) => {
                        self.validate_transform_anchor(&attrs, None);
                        let ffi = self.validate_ffi_record(&rec, ffi, span);
                        let lang = self.lang_attr(&attrs, span);
                        self.reject_sealed_attr(&attrs, span);
                        if let Some(crate::semi::lang::LangItem::Former(f)) = lang {
                            self.validate_former_anchor(&rec, f, span);
                        }
                        records.push((rec, span, scope, ffi, lang));
                    }
                    surface::StmtKind::Function(func) => {
                        let anchor =
                            self.validate_transform_anchor(&attrs, Some(func.body.is_some()));
                        let is_import = self.validate_ffi_function(&func, ffi, span);
                        let is_main = self.validate_main_attr(&attrs, &func, span);
                        let lang = self.lang_attr(&attrs, span);
                        self.reject_sealed_attr(&attrs, span);
                        if let Some(item) = lang
                            && item.kind() == crate::semi::lang::LangItemKind::Function
                            && func.body.is_some()
                        {
                            self.error(
                                span,
                                format!(
                                    "an intrinsic prototype must not have a body: \
                                     `{}` is checked and lowered by the compiler",
                                    item.name()
                                ),
                            );
                        }
                        functions.push((func, span, scope, anchor, is_import, is_main, lang));
                    }
                    surface::StmtKind::ExternTrampoline(t) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        trampolines.push((t, span, scope));
                    }
                    // `mod` declarations drive package *discovery* (the driver
                    // maps them to files before elaboration); here they carry
                    // no items.
                    surface::StmtKind::Mod(..) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                    }
                    // Bindings register during this scan — before any
                    // resolution — so they apply file-wide regardless of
                    // declaration order.
                    surface::StmtKind::Import(decl) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        self.declare_import(&decl, span, *file);
                    }
                    surface::StmtKind::Transform(script) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        self.transform_scripts.push(TransformScript {
                            body: script.body,
                            span: Some(script.body_span),
                            file: *file,
                        });
                    }
                    surface::StmtKind::Impl(ib) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        if ib.visibility == surface::Visibility::Public {
                            self.error(
                                span,
                                "`pub` is not allowed on an `impl` block; \
                                 mark individual members `pub` instead",
                            );
                        }
                        impls.push((ib, span, scope));
                    }
                    surface::StmtKind::Trait(td) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        let lang = self.lang_attr(&attrs, span);
                        let sealed = self.sealed_attr(&attrs, span);
                        trait_decls.push((td, span, scope, lang, sealed));
                    }
                    // Foreign preludes register during this scan (like
                    // bindings) so they apply file-wide regardless of order.
                    surface::StmtKind::ExternSource(src) => {
                        self.validate_transform_anchor(&attrs, None);
                        self.reject_ffi_attr(ffi, span);
                        if src.abi != "rust" {
                            self.error(
                                span,
                                format!(
                                    "unsupported foreign source ABI `{}`; only `rust` is supported",
                                    src.abi
                                ),
                            );
                        }
                        self.ffi_preludes.push(FfiPrelude {
                            body: src.block.body,
                            span: Some(src.block.body_span),
                            file: *file,
                        });
                    }
                }
            }
        }
        // The comparison tower must exist before any bound resolves. A batch
        // that declares one of its traits — `core` compiling itself —
        // suppresses the fallback wholesale: its own `#[lang]` declarations
        // are about to bind.
        if !trait_decls
            .iter()
            .any(|(_, _, _, lang, _)| lang.is_some_and(|item| item.is_cmp_trait()))
        {
            self.ensure_cmp_fallback();
        }
        // The later passes are keyed by the DefId each scan returned — never by
        // re-resolving the name — so an item whose declaration failed (e.g. a
        // duplicate) is skipped instead of clobbering the previously declared
        // item's fields or checking a body against the wrong prototype.
        // Phase A: trait stubs first — the record/function stub scans
        // resolve generic bounds immediately, so every trait must already
        // have a def and a TraitDb entry.
        let trait_ids: Vec<Option<TraitId>> = trait_decls
            .iter()
            .map(|(td, span, scope, lang, sealed)| {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                let id = self.scan_trait_stub(td, *sealed, *span);
                if let (Some(item), Some(id)) = (lang, id) {
                    let def = self.traits.trait_def(id).def;
                    self.declare_lang(*item, def, crate::semi::lang::LangItemKind::Trait, *span);
                }
                id
            })
            .collect();
        // Every provenance of the tower is in place now — fallback, loaded
        // interface, or this very batch's stubs.
        self.ensure_num_partial_ord_edge();
        let record_defs: Vec<Option<DefId>> = records
            .iter()
            .map(|(rec, span, scope, ffi, lang)| {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                let def = self.scan_record(rec, ffi.clone(), *span);
                if let (Some(item), Some(def)) = (lang, def) {
                    self.declare_lang(*item, def, crate::semi::lang::LangItemKind::Record, *span);
                }
                def
            })
            .collect();
        // Phase C, split: every trait's own binder first (a supertrait's
        // arity gate reads the super's params, which a same-batch
        // later-declared super would otherwise still be missing), then
        // supertraits and member signatures (which need the record stubs).
        for ((td, span, scope, ..), id) in trait_decls.iter().zip(&trait_ids) {
            if let Some(id) = id {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                self.populate_trait_params(td, *id, *span);
            }
        }
        for ((td, span, scope, ..), id) in trait_decls.iter().zip(&trait_ids) {
            if let Some(id) = id {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                self.populate_trait_members(td, *id, *span);
            }
        }
        // Phase D: reject (and sever) super-trait cycles — `reaches`
        // recurses unguarded, so a surviving cycle would overflow.
        self.reject_supertrait_cycles(trait_ids.iter().flatten().copied());
        let function_defs: Vec<Option<DefId>> = functions
            .iter()
            .map(|(func, span, scope, _, is_import, _, lang)| {
                self.set_current_file(scope.file);
                self.defs.set_module(scope.module.to_vec());
                let def = self.scan_function(func, *is_import, *span);
                if let (Some(item), Some(def)) = (lang, def) {
                    self.declare_lang(*item, def, crate::semi::lang::LangItemKind::Function, *span);
                }
                def
            })
            .collect();
        // Impl members: declared after the record scan so targets resolve,
        // under `[module.., Type, name]` in the function namespace. Their
        // protos join the same wf-sweep and body-check flows as plain
        // functions below. Members are ordinary defs/generics/functions
        // entries, all covered by the REPL Checkpoint's truncate-and-retain
        // rollback — no new state to track.
        let mut methods: Vec<(&surface::Function, Option<Span>, Option<DefId>)> = Vec::new();
        for (ib, span, scope) in &impls {
            self.set_current_file(scope.file);
            self.defs.set_module(scope.module.to_vec());
            if ib.trait_ref.is_some() {
                let declared = self.scan_trait_impl(ib, *span);
                for (m, def) in ib.members.iter().zip(declared) {
                    let mspan = Some(surface::Span {
                        start: m.start,
                        end: m.end,
                    });
                    methods.push((&m.value, mspan, def));
                }
                continue;
            }
            let target = self.scan_impl_target(ib, false, *span);
            let impl_generics = target.map(|_| self.collect_generics(&ib.generics, *span));
            // The applied target type: `Self` inside the block. Evaluated
            // once, under the block-level generics.
            let self_ty = match (target, &impl_generics) {
                (Some(target), Some(generics)) => {
                    self.generic_names = generics.iter().map(|(n, id)| (*n, *id)).collect();
                    let args: Vec<Ty<'tcx>> =
                        ib.target_args.iter().map(|a| self.eval_type(a)).collect();
                    self.generic_names.clear();
                    let flex = match self.records[&target].default_cap {
                        DefaultCap::Value | DefaultCap::Shared => Flexivity::Irrelevant,
                        DefaultCap::Regional => Flexivity::Regional,
                    };
                    Some(self.tcx.mk_record(target, &args, flex))
                }
                _ => None,
            };
            for m in &ib.members {
                let mspan = Some(surface::Span {
                    start: m.start,
                    end: m.end,
                });
                let def = match (target, &impl_generics, self_ty) {
                    (Some(target), Some(generics), Some(self_ty)) => {
                        self.defs.set_module(scope.module.to_vec());
                        let ctx = ImplCtx {
                            target,
                            generics,
                            self_ty,
                            trait_of: None,
                        };
                        self.scan_method(&m.value, false, Some(&ctx), mspan)
                    }
                    _ => None,
                };
                methods.push((&m.value, mspan, def));
            }
        }
        // Scalar-impl completion runs after the impl scan, so a batch that
        // declares the builtin impls itself — `core`, whose source carries
        // them — is left alone; the completion only fills what no source
        // or interface provided (the no-`core` fallback).
        self.ensure_cmp_scalar_impls();
        functions
            .iter()
            .zip(&function_defs)
            .filter_map(|((_, _, _, anchor, _, _, _), def)| if *anchor { *def } else { None })
            .for_each(|def| self.transform_anchors.push(def));
        // `#[main]`: export the entry point under the symbol the runtime's
        // `__reussir_start` is handed. Registered here, once the scan has
        // assigned DefIds, and through the same trampoline machinery an
        // explicit `extern "C" trampoline` uses — so the entry point is a
        // C-ABI export like any other, and monomorphization roots it.
        let mains: Vec<(DefId, Option<Span>)> = functions
            .iter()
            .zip(&function_defs)
            .filter_map(|((_, span, _, _, _, is_main, _), def)| {
                (*is_main).then(|| Some(((*def)?, *span))).flatten()
            })
            .collect();
        self.register_main_entry(&mains);
        let record_defs: Vec<DefId> = records
            .iter()
            .zip(record_defs)
            .filter_map(|((rec, _, _, _, _), def)| Some((rec, def?)))
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
            let (span, file, vis, name) = (rec.span, rec.file, rec.visibility, rec.name);
            let member_tys: Vec<Ty<'tcx>> = match &rec.fields {
                Some(RecordFields::Named(fs)) => fs.iter().map(|(_, t, _, _)| *t).collect(),
                Some(RecordFields::Unnamed(fs)) => fs.iter().map(|(t, _, _)| *t).collect(),
                Some(RecordFields::Variants(vs)) => {
                    vs.iter().flat_map(|v| v.fields.iter().copied()).collect()
                }
                Some(RecordFields::Opaque) | None => Vec::new(),
            };
            self.set_current_file(file);
            for &t in &member_tys {
                self.sweep_arc_wf(t, span);
            }
            if vis == surface::Visibility::Public {
                let item = format!("`pub` record `{}`", self.sym(name));
                self.check_private_in_public(&member_tys, &item, span);
            }
        }
        let method_defs: Vec<Option<DefId>> = methods.iter().map(|(_, _, def)| *def).collect();
        for def in function_defs.iter().chain(method_defs.iter()).flatten() {
            let Some(proto) = self.functions.get(def) else {
                continue;
            };
            let (span, file, vis, name) = (proto.span, proto.file, proto.visibility, proto.name);
            let tys: Vec<Ty<'tcx>> = proto
                .params
                .iter()
                .map(|(_, t)| *t)
                .chain([proto.return_ty])
                .collect();
            self.set_current_file(file);
            for &t in &tys {
                self.sweep_arc_wf(t, span);
            }
            if vis == surface::Visibility::Public {
                let item = format!("`pub fn {}`", self.sym(name));
                self.check_private_in_public(&tys, &item, span);
            }
        }
        self.records_complete = true;
        functions
            .iter()
            .zip(function_defs)
            .filter_map(|((func, span, _, _, _, _, _), def)| Some((func, *span, def?)))
            .for_each(|(func, span, def)| {
                self.check_function(func, def, span);
            });
        methods
            .iter()
            .filter_map(|(func, span, def)| Some((*func, *span, (*def)?)))
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

    /// Recognize a `#[ffi(...)]` attribute among `attrs`: `#[ffi(import)]`
    /// or `#[ffi(rust = "path")]`. Malformed or duplicated uses report and
    /// yield `None` (the item elaborates as if unannotated).
    fn ffi_attr(&mut self, attrs: &[surface::Attribute]) -> Option<FfiSpec> {
        let mut spec = None;
        for attr in attrs {
            if self.sym(attr.name) != "ffi" {
                continue;
            }
            let span = Some(attr.span);
            if spec.is_some() {
                self.error(span, "`#[ffi(...)]` may only be specified once per item");
                continue;
            }
            match (attr.args.as_slice(), attr.values.as_slice()) {
                ([arg], []) if self.sym(*arg) == "import" => spec = Some(FfiSpec::Import),
                ([], [(k, path)]) if self.sym(*k) == "rust" => {
                    spec = Some(FfiSpec::Opaque(path.clone()));
                }
                _ => self.error(
                    span,
                    "malformed `#[ffi(...)]`: expected `#[ffi(import)]` or \
                     `#[ffi(rust = \"path\")]`",
                ),
            }
        }
        spec
    }

    /// The symbol a `#[main]` function is exported under — what the runtime's
    /// `__reussir_start` is handed to run the program.
    pub const MAIN_SYMBOL: &'static str = "__reussir_main";

    /// Recognize `#[main]` on a function and check it can be an entry point:
    /// it must have a body, take no parameters, and be non-generic, since the
    /// runtime calls it as a bare `extern "C" fn()`. Reports and yields false
    /// when it cannot.
    fn validate_main_attr(
        &mut self,
        attrs: &[surface::Attribute],
        func: &surface::Function,
        span: Option<Span>,
    ) -> bool {
        let mut seen = false;
        for attr in attrs {
            if self.sym(attr.name) != "main" {
                continue;
            }
            let at = Some(attr.span);
            if seen {
                self.error(at, "`#[main]` may only be specified once per item");
                continue;
            }
            seen = true;
            if !attr.args.is_empty() || !attr.values.is_empty() {
                self.error(at, "malformed `#[main]`: it takes no arguments");
            }
        }
        if !seen {
            return false;
        }

        let mut valid = true;
        if func.body.is_none() {
            self.error(
                span,
                "`#[main]` needs a function body: it is the program's entry point",
            );
            valid = false;
        }
        if !func.generics.is_empty() {
            self.error(
                span,
                "`#[main]` cannot be generic: the entry point is called with no type information",
            );
            valid = false;
        }
        if !func.params.is_empty() {
            self.error(
                span,
                format!(
                    "`#[main]` takes no parameters, but this one takes {}; \
                     the runtime calls the entry point with no arguments",
                    func.params.len()
                ),
            );
            valid = false;
        }
        valid
    }

    /// Register the program's entry point: at most one `#[main]`, exported as
    /// [`Self::MAIN_SYMBOL`] through the ordinary trampoline path.
    fn register_main_entry(&mut self, mains: &[(DefId, Option<Span>)]) {
        let Some(((target, _), rest)) = mains.split_first() else {
            return;
        };
        for (_, span) in rest {
            self.error(
                *span,
                "a program can have only one `#[main]` function; another is already declared",
            );
        }
        // The trampoline is the *only* record of the entry point, and that is
        // deliberate: it is what the HIR and MIR dumps carry, so a build that
        // re-enters from one of them (`rrc prog.hir -o prog.o`) sees the same
        // entry point a build from source does. A side-channel field here
        // would be lost across that boundary — silently, and only for
        // resumed builds.
        self.trampolines.push(TrampolineRoot {
            name: Self::MAIN_SYMBOL.to_owned(),
            abi: "C".to_owned(),
            target: *target,
            ty_args: Vec::new(),
        });
    }

    /// Reject an `#[ffi(...)]` attribute on an item kind that takes none.
    fn reject_ffi_attr(&mut self, ffi: Option<FfiSpec>, span: Option<Span>) {
        if ffi.is_some() {
            self.error(
                span,
                "`#[ffi(...)]` may only be attached to a struct or a function",
            );
        }
    }

    /// Check the `#[ffi(rust = "path")]` rules on a record and return the
    /// Rust path when the record is a valid opaque FFI record.
    fn validate_ffi_record(
        &mut self,
        rec: &surface::Record,
        ffi: Option<FfiSpec>,
        span: Option<Span>,
    ) -> Option<String> {
        let opaque = matches!(rec.fields, surface::RecordFields::Opaque);
        let Some(spec) = ffi else {
            if opaque {
                self.error(
                    span,
                    "a field-less record is an opaque FFI record and requires \
                     `#[ffi(rust = \"path\")]`",
                );
            }
            return None;
        };
        let FfiSpec::Opaque(path) = spec else {
            self.error(span, "`#[ffi(import)]` may only be attached to a function");
            return None;
        };
        if !opaque {
            self.error(
                span,
                "an `#[ffi(rust = ...)]` record cannot declare fields or variants",
            );
            return None;
        }
        // The foreign side is a `#[repr(transparent)]` wrapper over
        // `reussir_rt`'s `Rc`, i.e. a shared box: the capability is fixed.
        if rec.default_cap != surface::Capability::Shared {
            self.error(span, "an `#[ffi(rust = ...)]` record is always `[shared]`");
            return None;
        }
        Some(path)
    }

    /// Check the `#[ffi(import)]` rules on a function; returns whether the
    /// function is a valid FFI import (its foreign body registers under its
    /// def in [`Self::scan_function`]).
    fn validate_ffi_function(
        &mut self,
        func: &surface::Function,
        ffi: Option<FfiSpec>,
        span: Option<Span>,
    ) -> bool {
        let Some(spec) = ffi else {
            if func.foreign_body.is_some() {
                self.error(
                    span,
                    "a foreign function body (`[{ ... }];`) requires `#[ffi(import)]`",
                );
            }
            return false;
        };
        if !matches!(spec, FfiSpec::Import) {
            self.error(
                span,
                "`#[ffi(rust = ...)]` may only be attached to a field-less struct",
            );
            return false;
        }
        if func.foreign_body.is_none() {
            self.error(
                span,
                "`#[ffi(import)]` requires a foreign body: `fn f(...) -> T [{ ... }];`",
            );
            return false;
        }
        if func.is_regional {
            self.error(span, "an `#[ffi(import)]` function cannot be `regional`");
            return false;
        }
        true
    }

    fn scan_record(
        &mut self,
        rec: &surface::Record,
        ffi: Option<String>,
        span: Option<Span>,
    ) -> Option<DefId> {
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
            visibility: rec.visibility,
            ty_params,
            kind: rec.kind,
            default_cap,
            repr_fixed,
            ffi,
            fields: None,
            regional_generics: Vec::new(),
            span,
            file: self.current_file,
        };
        self.records.insert(def, record);
        Some(def)
    }

    fn scan_function(
        &mut self,
        func: &surface::Function,
        is_ffi_import: bool,
        span: Option<Span>,
    ) -> Option<DefId> {
        self.scan_method(func, is_ffi_import, None, span)
    }

    /// Scan a trait implementation `impl<G..> Trait<args> for Type<..>`:
    /// resolve and gate the trait (sealed builtins rejected), validate the
    /// target and trait-argument arity, enforce E0207 constrainedness,
    /// declare every member under the trait-impl path, check conformance
    /// against the trait's method signatures (exact set, positional, Self
    /// substituted), and register the `ImplDef`. Returns each member's
    /// declared def in member order so bodies join the common check flow.
    /// The scalar type a primitive impl-target names: a bare primitive
    /// spelling with no arguments, unshadowed by any record in scope.
    fn prim_impl_target(
        &mut self,
        target: &surface::Path,
        target_args: &[surface::Type],
    ) -> Option<Ty<'tcx>> {
        use crate::semi::ty::{FpTy, IntTy};
        if !target.segments.is_empty()
            || !target_args.is_empty()
            || self.resolve_record_ref(target).is_some()
        {
            return None;
        }
        Some(match self.sym(target.basename) {
            "i8" => self.tcx.mk_int(IntTy::Signed(8)),
            "i16" => self.tcx.mk_int(IntTy::Signed(16)),
            "i32" => self.tcx.mk_int(IntTy::Signed(32)),
            "i64" => self.tcx.mk_int(IntTy::Signed(64)),
            "u8" => self.tcx.mk_int(IntTy::Unsigned(8)),
            "u16" => self.tcx.mk_int(IntTy::Unsigned(16)),
            "u32" => self.tcx.mk_int(IntTy::Unsigned(32)),
            "u64" => self.tcx.mk_int(IntTy::Unsigned(64)),
            "f16" => self.tcx.mk_fp(FpTy::Ieee(16)),
            "f32" => self.tcx.mk_fp(FpTy::Ieee(32)),
            "f64" => self.tcx.mk_fp(FpTy::Ieee(64)),
            "bfloat16" => self.tcx.mk_fp(FpTy::BFloat16),
            "float8" => self.tcx.mk_fp(FpTy::Float8),
            "bool" => self.tcx.mk_bool(),
            "char" => self.tcx.mk_char(),
            "str" => self.tcx.mk_str(),
            "unit" => self.tcx.mk_unit(),
            _ => return None,
        })
    }

    /// The builtin-impl special form: `impl Trait for <primitive> { }`,
    /// how `core` declares the compiler-provided impls in source — with a
    /// real declaration site, serialized into the interface. Reserved to
    /// lang-bound traits declared by the current package; the members are
    /// compiler-provided (selection of the method-less impl lowers
    /// intrinsically), so the body must be empty.
    fn scan_builtin_impl(
        &mut self,
        ib: &surface::ImplBlock,
        tid: TraitId,
        targs_surface: &[surface::Type],
        self_ty: Ty<'tcx>,
        span: Option<Span>,
    ) -> Vec<Option<DefId>> {
        let nones = vec![None; ib.members.len()];
        let tdef = self.traits.trait_def(tid);
        let lang_bound = self.lang.item_of(tdef.def).is_some();
        let local = !self.extern_defs.contains_key(&tdef.def);
        if !lang_bound || !local {
            self.error(
                span,
                format!(
                    "cannot implement `{}` for a builtin type: builtin impls \
                     are reserved to the package declaring the lang trait",
                    self.trait_display(tid)
                ),
            );
            return nones;
        }
        if !ib.members.is_empty() {
            self.error(
                span,
                "the members of a builtin impl are compiler-provided; the \
                 impl body must be empty",
            );
            return nones;
        }
        if !ib.generics.is_empty() {
            self.error(span, "a builtin impl takes no generic parameters");
            return nones;
        }
        let trait_args: Vec<Ty<'tcx>> = targs_surface.iter().map(|a| self.eval_type(a)).collect();
        let expected = self.traits.trait_def(tid).params.len();
        if trait_args.len() != expected {
            self.error(
                span,
                format!(
                    "trait `{}` expects {expected} type argument(s), got {}",
                    self.trait_display(tid),
                    trait_args.len()
                ),
            );
            return nones;
        }
        let head = crate::semi::traits::coherence::head_key(self_ty).expect("scalars have heads");
        if !self.traits.impls_for(tid, head).is_empty() {
            self.error(
                span,
                format!(
                    "conflicting implementations of trait `{}` for `{}`",
                    self.trait_display(tid),
                    self.ty_display(self_ty)
                ),
            );
            return nones;
        }
        let id = crate::semi::traits::ImplId(self.traits.impls_len() as u32);
        let mut args = Vec::with_capacity(1 + trait_args.len());
        args.push(self_ty);
        args.extend(trait_args.iter().copied());
        self.traits.add_impl(crate::semi::traits::def::ImplDef {
            id,
            generics: Vec::new(),
            trait_ref: crate::semi::traits::TraitRef {
                trait_id: tid,
                args,
            },
            self_ty,
            where_clauses: Vec::new(),
            methods: Vec::new(),
            span,
            file: self.current_file,
        });
        nones
    }

    fn scan_trait_impl(
        &mut self,
        ib: &surface::ImplBlock,
        span: Option<Span>,
    ) -> Vec<Option<DefId>> {
        let nones = vec![None; ib.members.len()];
        let tref = ib.trait_ref.as_ref().expect("trait impl");
        let (tpath, targs_surface) = (&tref.path, &tref.args);

        // Resolve the trait through the trait namespace.
        let expanded = self.expand_path(tpath);
        let rpath = expanded.as_ref().unwrap_or(tpath);
        let tdef = if self.extern_head(rpath).is_some() {
            self.resolve_extern(rpath, DefKind::Trait)
        } else if rpath.segments.is_empty() {
            self.defs
                .resolve_trait(rpath.basename)
                .map(|def| self.lang.override_builtin(def, &self.traits))
                .or_else(|| {
                    self.prelude_lookup(rpath.basename, crate::semi::lang::LangItemKind::Trait)
                })
        } else {
            self.defs
                .resolve_trait_path(&self.classify_segs(rpath), rpath.basename)
        };
        let Some(tid) = tdef.and_then(|d| self.traits.trait_by_def(d)) else {
            if let Some(msg) = self.extern_private_msg(rpath, DefKind::Trait) {
                self.error(span, msg);
                return nones;
            }
            let hint = if rpath.segments.is_empty() {
                self.trait_suggestion(self.sym(rpath.basename))
            } else {
                String::new()
            };
            let shown = self.path_display(rpath);
            self.error(
                span,
                format!("cannot implement unknown trait `{shown}`{hint}"),
            );
            return nones;
        };
        // The builtin-impl special form: `impl Trait for <primitive> { }`.
        // Detected before the sealed guard — `core` declares the sealed
        // tower's own impls this way; every other package is rejected by
        // the form's locality rule instead.
        if let Some(scalar) = self.prim_impl_target(&ib.target, &ib.target_args) {
            return self.scan_builtin_impl(ib, tid, targs_surface, scalar, span);
        }
        if self.traits.trait_def(tid).sealed {
            self.error(
                span,
                format!(
                    "`{}` is a built-in trait and cannot be implemented; its \
                     impls are provided by the language",
                    self.trait_display(tid)
                ),
            );
            return nones;
        }
        // The operator-dispatch traits are implementable only through
        // `core`'s method-carrying declarations. An impl of the compiler's
        // method-less stand-in has nothing to conform to — and accepting one
        // would let a non-scalar satisfy a comparison bound and reach the
        // intrinsic lowering. (`Eq` stays free: it is a marker in `core`
        // too, and no operator dispatches through it.)
        {
            use crate::semi::lang::LangItem;
            let t = self.traits.trait_def(tid);
            if t.methods.is_empty()
                && matches!(
                    self.lang.item_of(t.def),
                    Some(LangItem::PartialEq | LangItem::PartialOrd | LangItem::Ord)
                )
            {
                self.error(
                    span,
                    format!(
                        "cannot implement `{}`: without `core` it is a \
                         method-less compiler fallback; the operator traits \
                         are implementable through `core`'s declarations",
                        self.trait_display(tid)
                    ),
                );
                return nones;
            }
        }

        let Some(target) = self.scan_impl_target(ib, true, span) else {
            return nones;
        };
        let generics = self.collect_generics(&ib.generics, span);

        // Trait arguments and the self type evaluate under the block
        // generics.
        self.generic_names = generics.iter().map(|(n, id)| (*n, *id)).collect();
        let trait_args: Vec<Ty<'tcx>> = targs_surface.iter().map(|a| self.eval_type(a)).collect();
        let self_args: Vec<Ty<'tcx>> = ib.target_args.iter().map(|a| self.eval_type(a)).collect();
        self.generic_names.clear();
        let flex = match self.records[&target].default_cap {
            DefaultCap::Value | DefaultCap::Shared => Flexivity::Irrelevant,
            DefaultCap::Regional => Flexivity::Regional,
        };
        let self_ty = self.tcx.mk_record(target, &self_args, flex);

        let expected = self.traits.trait_def(tid).params.len();
        if trait_args.len() != expected {
            self.error(
                span,
                format!(
                    "trait `{}` expects {expected} type argument(s), got {}",
                    self.trait_display(tid),
                    trait_args.len()
                ),
            );
            return nones;
        }

        // E0207: every block generic must be constrained by the trait
        // reference or the self type, or monomorphization could never derive
        // its instantiation.
        let mut constrained = rustc_hash::FxHashSet::default();
        std::iter::once(self_ty)
            .chain(trait_args.iter().copied())
            .for_each(|t| crate::semi::traits::subst::collect_generics_of(t, &mut constrained));
        generics
            .iter()
            .filter(|(_, g)| !constrained.contains(g))
            .map(|&(name, _)| {
                format!(
                    "impl generic `{}` is not constrained by the trait \
                     reference or the self type",
                    self.sym(name)
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|msg| self.error(span, msg));

        // ORPHAN (package granularity): the trait or the self type's head
        // must be local — extern provenance is exactly the extern_defs map.
        let trait_local = !self
            .extern_defs
            .contains_key(&self.traits.trait_def(tid).def);
        let head_local = !self.extern_defs.contains_key(&target);
        if !trait_local && !head_local {
            self.error(
                span,
                format!(
                    "cannot implement extern trait `{}` for extern type `{}`: \
                     the trait or the self type's head must be local to this \
                     package",
                    self.trait_display(tid),
                    self.defs.path(target).display(self.resolver)
                ),
            );
            return nones;
        }

        // OVERLAP: within the (trait, head) bucket, two-sided unification —
        // both impls' generics unifiable — so crossing generic heads
        // (`Pair<T, i32>` vs `Pair<u8, U>`) conflict. Coherence is what
        // makes selection's committed choice unique.
        let head = crate::semi::traits::coherence::head_key(self_ty)
            .expect("impl targets resolve to record heads");
        let new_args: Vec<Ty<'tcx>> = std::iter::once(self_ty)
            .chain(trait_args.iter().copied())
            .collect();
        let conflict = self
            .traits
            .impls_for(tid, head)
            .iter()
            .find_map(|&existing| {
                let other = self.traits.impl_def(existing);
                let vars: rustc_hash::FxHashSet<GenericId> = other
                    .generics
                    .iter()
                    .chain(generics.iter().map(|(_, g)| g))
                    .copied()
                    .collect();
                let mut subst = FxHashMap::default();
                crate::semi::traits::coherence::unify_args(
                    &other.trait_ref.args,
                    &new_args,
                    &vars,
                    &mut subst,
                )
                .then_some(other.self_ty)
            });
        if let Some(other_self) = conflict {
            self.error(
                span,
                format!(
                    "conflicting implementations of trait `{}` for `{}`: \
                     `{0}` is already implemented for `{}`",
                    self.trait_display(tid),
                    self.ty_display(self_ty),
                    self.ty_display(other_self)
                ),
            );
            return nones;
        }

        // Members: declare + conformance, in member order; track coverage in
        // trait-method order for the missing-set diagnostic and the
        // ImplDef's method table.
        let sigs = self.traits.trait_def(tid).methods.clone();
        let mut by_sig: Vec<Option<DefId>> = vec![None; sigs.len()];
        let declared: Vec<Option<DefId>> = ib
            .members
            .iter()
            .map(|m| {
                let func = &m.value;
                let mspan = Some(Span {
                    start: m.start,
                    end: m.end,
                });
                let Some(sig_idx) = sigs.iter().position(|sig| sig.name == func.name) else {
                    self.error(
                        mspan,
                        format!(
                            "`{}` is not a member of trait `{}`",
                            self.sym(func.name),
                            self.trait_display(tid)
                        ),
                    );
                    return None;
                };
                if func.visibility == surface::Visibility::Public {
                    self.error(
                        mspan,
                        "visibility is not allowed on a trait impl method; the \
                         trait's visibility applies",
                    );
                }
                self.defs.set_module(self.defs.module().to_vec());
                let ctx = ImplCtx {
                    target,
                    generics: &generics,
                    self_ty,
                    trait_of: Some(tid),
                };
                let def = self.scan_method(func, false, Some(&ctx), mspan)?;
                if by_sig[sig_idx].replace(def).is_some() {
                    // Duplicate member: the declare clash already reported.
                    return Some(def);
                }
                self.check_conformance(
                    tid,
                    &sigs[sig_idx],
                    def,
                    self_ty,
                    &trait_args,
                    generics.len(),
                    mspan,
                );
                Some(def)
            })
            .collect();

        let missing: Vec<String> = sigs
            .iter()
            .zip(&by_sig)
            .filter(|(_, d)| d.is_none())
            .map(|(sig, _)| format!("`{}`", self.sym(sig.name)))
            .collect();
        if !missing.is_empty() {
            self.error(
                span,
                format!(
                    "impl of trait `{}` for `{}` is missing method(s) {}",
                    self.trait_display(tid),
                    self.ty_display(self_ty),
                    missing.join(", ")
                ),
            );
            return declared;
        }

        let id = crate::semi::traits::ImplId(self.traits.impls_len() as u32);
        let mut args = Vec::with_capacity(1 + trait_args.len());
        args.push(self_ty);
        args.extend(trait_args.iter().copied());
        let where_clauses = generics
            .iter()
            .flat_map(|(_, g)| {
                let gt = self.tcx.mk_generic(*g);
                self.generic_bounds(*g)
                    .iter()
                    .map(move |&b| {
                        crate::semi::traits::Obligation::Trait(crate::semi::traits::TraitRef {
                            trait_id: b,
                            args: vec![gt],
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        self.traits.add_impl(crate::semi::traits::def::ImplDef {
            id,
            generics: generics.iter().map(|(_, g)| *g).collect(),
            trait_ref: crate::semi::traits::TraitRef {
                trait_id: tid,
                args,
            },
            self_ty,
            where_clauses,
            methods: by_sig
                .into_iter()
                .map(|d| d.expect("missing checked"))
                .collect(),
            span,
            file: self.current_file,
        });
        declared
    }

    /// Conformance of one declared member against its trait signature: the
    /// trait's binder substituted (`Self` -> the impl self type, trait
    /// params -> the impl's trait arguments, the signature's own generics ->
    /// the member's own), then positional pointer-equality over the
    /// non-receiver parameters and the return type, plus receiver-form,
    /// regional, and generic-count parity.
    #[allow(clippy::too_many_arguments)]
    fn check_conformance(
        &mut self,
        tid: TraitId,
        sig: &crate::semi::traits::def::MethodSig<'tcx>,
        def: DefId,
        self_ty: Ty<'tcx>,
        trait_args: &[Ty<'tcx>],
        block_generics: usize,
        span: Option<Span>,
    ) {
        use crate::semi::traits::def::ReceiverForm;
        let proto = self.functions[&def].clone();
        let tdef = self.traits.trait_def(tid);
        let name = self.sym(proto.name).to_owned();
        let tr = self.trait_display(tid);

        // The member's own generics follow the block's in its binder.
        let own = &proto.generics[block_generics..];
        if own.len() != sig.generics.len() {
            self.error(
                span,
                format!(
                    "method `{name}` declares {} generic parameter(s); trait \
                     `{tr}` declares {}",
                    own.len(),
                    sig.generics.len()
                ),
            );
            return;
        }

        let map: FxHashMap<GenericId, Ty<'tcx>> = std::iter::once((tdef.self_param, self_ty))
            .chain(
                tdef.params
                    .iter()
                    .zip(trait_args)
                    .map(|(&(_, p), &a)| (p, a)),
            )
            .chain(
                sig.generics
                    .iter()
                    .zip(own)
                    .map(|(&(_, sg), &(_, ig))| (sg, self.tcx.mk_generic(ig))),
            )
            .collect();

        if sig.params.len() != proto.params.len() {
            self.error(
                span,
                format!(
                    "method `{name}` takes {} parameter(s); trait `{tr}` \
                     requires {}",
                    proto.params.len(),
                    sig.params.len()
                ),
            );
            return;
        }
        // Receiver: form parity, not type equality (flexivity refinement and
        // the arc peel make the spellings differ structurally). An associated
        // function (no receiver in the trait) has no slot to compare — every
        // parameter is positional, and a stray `self`-typed first parameter
        // would already fail the positional type check below.
        if let Some(sig_form) = sig.receiver {
            let impl_form = {
                let (_, pty) = &proto.params[0];
                match pty.kind() {
                    TyKind::Arc(_) => ReceiverForm::Arc,
                    _ if self.is_flex(*pty) => ReceiverForm::Flex,
                    _ => ReceiverForm::Value,
                }
            };
            if impl_form != sig_form {
                let want = match sig_form {
                    ReceiverForm::Value => "self: Self",
                    ReceiverForm::Arc => "self: Arc<Self>",
                    ReceiverForm::Flex => "self: [flex] Self",
                };
                self.error(
                    span,
                    format!("the receiver of `{name}` must be `{want}` to match trait `{tr}`"),
                );
            }
        }
        let skip = sig.receiver.is_some() as usize;
        for (i, ((_, ity), &sty)) in proto.params.iter().zip(&sig.params).enumerate().skip(skip) {
            let want = crate::semi::traits::subst::replace_generics(self.tcx, sty, &map);
            if *ity != want {
                self.error(
                    span,
                    format!(
                        "parameter {i} of method `{name}` has type `{}` but trait \
                         `{tr}` requires `{}`",
                        self.ty_display(*ity),
                        self.ty_display(want)
                    ),
                );
            }
        }
        let want_ret = crate::semi::traits::subst::replace_generics(self.tcx, sig.ret, &map);
        if proto.return_ty != want_ret {
            self.error(
                span,
                format!(
                    "method `{name}` returns `{}` but trait `{tr}` requires `{}`",
                    self.ty_display(proto.return_ty),
                    self.ty_display(want_ret)
                ),
            );
        }
        if sig.is_regional != proto.is_regional {
            let (a, b) = if sig.is_regional {
                ("is", "but not in this impl")
            } else {
                ("is not", "but is in this impl")
            };
            self.error(
                span,
                format!("method `{name}` {a} `regional` in trait `{tr}` {b}"),
            );
        }
    }

    /// Validate an `impl` block's target: it must resolve to a record this
    /// package defines, in the module the block appears in, with a fully
    /// applied generic head.
    fn scan_impl_target(
        &mut self,
        ib: &surface::ImplBlock,
        extern_target_ok: bool,
        span: Option<Span>,
    ) -> Option<DefId> {
        let Some(def) = self.resolve_record_ref(&ib.target) else {
            // A private record of a loaded extern package reports access,
            // not absence — though methods for it are rejected either way.
            if let Some(msg) = self.extern_private_msg(&ib.target, DefKind::Record) {
                self.error(span, msg);
            } else {
                let hint = if ib.target.segments.is_empty() {
                    self.record_suggestion(ib.target.basename)
                } else {
                    String::new()
                };
                let shown = self.path_display(&ib.target);
                self.error(
                    span,
                    format!("cannot define methods for unknown type `{shown}`{hint}"),
                );
            }
            return None;
        };
        if !extern_target_ok && let Some(&head) = self.extern_defs.get(&def) {
            self.error(
                span,
                format!(
                    "cannot define methods for `{}` from extern package `{}`",
                    self.path_display(&ib.target),
                    self.sym(head)
                ),
            );
            return None;
        }
        let expected = self.records[&def].ty_params.len();
        if ib.target_args.len() != expected {
            self.error(
                span,
                format!(
                    "`impl` target `{}` expects {} type argument(s), got {}",
                    self.sym(ib.target.basename),
                    expected,
                    ib.target_args.len()
                ),
            );
            return None;
        }
        Some(def)
    }

    fn scan_method(
        &mut self,
        func: &surface::Function,
        is_ffi_import: bool,
        impl_ctx: Option<&ImplCtx<'_, 'tcx>>,
        span: Option<Span>,
    ) -> Option<DefId> {
        // The block's own module, captured before declaration temporarily
        // switches to the type's path: the member's body scope.
        let impl_module = impl_ctx.map(|_| self.defs.module().to_vec());
        let own = self.collect_generics(&func.generics, span);
        let generics = match impl_ctx {
            Some(ctx) => {
                for (n, _) in &own {
                    if ctx.generics.iter().any(|(g, _)| g == n) {
                        self.error(
                            span,
                            format!(
                                "generic `{0}` on the method shadows the impl block's \
                                 generic `{0}`",
                                self.sym(*n)
                            ),
                        );
                    }
                }
                ctx.generics.iter().copied().chain(own).collect()
            }
            None => own,
        };
        self.generic_names = generics.iter().map(|(n, id)| (*n, *id)).collect();
        self.self_alias = impl_ctx.map(|ctx| ctx.self_ty);

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
        self.self_alias = None;

        // Receiver validation: a first parameter named `self` must be the
        // impl target, optionally behind one `Arc` and/or a flexivity
        // refinement; `self` anywhere else is rejected.
        if let Some(ctx) = impl_ctx {
            for (i, (pname, pty)) in params.iter().enumerate() {
                if self.sym(*pname) != "self" {
                    continue;
                }
                if i != 0 {
                    self.error(
                        span,
                        "a parameter named `self` must be the method's first parameter",
                    );
                    continue;
                }
                let peeled = match pty.kind() {
                    TyKind::Arc(inner) => *inner,
                    _ => *pty,
                };
                let TyKind::Record {
                    def: target,
                    args: target_args,
                    ..
                } = ctx.self_ty.kind()
                else {
                    unreachable!("an impl target is always a record type");
                };
                let matches_target = matches!(
                    peeled.kind(),
                    TyKind::Record { def, args, .. } if def == target && args == target_args
                );
                if !matches_target {
                    self.error(
                        span,
                        "the receiver of a method must be `Self`, `Arc<Self>`, or                          `[flex] Self` (or the equivalent spelling of the impl target)",
                    );
                }
            }
        }

        let name = func.name;
        let declared = match impl_ctx {
            // A trait-impl member declares under
            // `[impl-module.., trait-path.., head-name, method]` — unique by
            // coherence, never spelled at call sites (dispatch goes through
            // the trait), mangled by the ordinary path machinery.
            Some(ctx) if ctx.trait_of.is_some() => {
                let tid = ctx.trait_of.expect("checked");
                let module = self.defs.module().to_vec();
                let mut decl = module.clone();
                decl.extend(self.defs.path(self.traits.trait_def(tid).def).0.iter());
                decl.push(self.defs.path(ctx.target).name());
                self.defs.set_module(decl);
                let def = self.defs.declare_function(name);
                self.defs.set_module(module);
                def
            }
            // An inherent member declares under the type's own qualified
            // path, so `Type::method` resolves like any path.
            Some(ctx) => {
                let module = self.defs.module().to_vec();
                let type_path = self.defs.path(ctx.target).0.clone();
                self.defs.set_module(type_path);
                let def = self.defs.declare_function(name);
                self.defs.set_module(module);
                def
            }
            None => self.defs.declare_function(name),
        };
        let Some(def) = declared else {
            match impl_ctx {
                Some(ctx) if ctx.trait_of.is_some() => {
                    let tid = ctx.trait_of.expect("checked");
                    let ty = self.sym(self.defs.path(ctx.target).name()).to_owned();
                    let tr = self.trait_display(tid);
                    self.error(
                        span,
                        format!(
                            "a method `{0}` for `{ty}` under trait `{tr}` is already \
                             declared by another impl; merge the impls of `{tr}` for \
                             `{ty}` into one parameterized impl",
                            self.sym(name)
                        ),
                    );
                }
                Some(ctx) => {
                    let ty = self.sym(self.defs.path(ctx.target).name());
                    self.error(
                        span,
                        format!(
                            "method `{0}` conflicts with an existing function `{ty}::{0}` \
                             (a function in a module named `{ty}` or another impl of `{ty}`)",
                            self.sym(name)
                        ),
                    );
                }
                None => self.error(
                    span,
                    format!("function `{}` is defined more than once", self.sym(name)),
                ),
            }
            return None;
        };
        let visibility = match impl_ctx {
            // Trait members take the trait's visibility.
            Some(ctx) if ctx.trait_of.is_some() => {
                self.traits
                    .trait_def(ctx.trait_of.expect("checked"))
                    .visibility
            }
            _ => func.visibility,
        };
        let proto = FuncProto {
            def,
            name,
            visibility,
            generics,
            regional_generics,
            params,
            return_ty,
            impl_module,
            self_ty: impl_ctx.map(|ctx| ctx.self_ty),
            is_regional: func.is_regional,
            span,
            file: self.current_file,
        };
        self.functions.insert(def, proto);
        if is_ffi_import {
            let block = func.foreign_body.as_ref().expect("validated ffi import");
            self.ffi_imports.insert(
                def,
                FfiImport {
                    body: block.body.clone(),
                    span: Some(block.body_span),
                    file: self.current_file,
                },
            );
        }
        Some(def)
    }

    /// Phase A of the trait scan: declare the trait's def and register a
    /// stub `TraitDef` so bounds resolve against it from the very first
    /// record/function stub scan. Params, supertraits, and members fill in
    /// during the populate phases.
    fn scan_trait_stub(
        &mut self,
        td: &surface::TraitDecl,
        sealed: bool,
        span: Option<Span>,
    ) -> Option<TraitId> {
        let Some(def) = self.defs.declare_trait(td.name) else {
            // The name is taken in this module's trait namespace: a sealed
            // builtin (only reachable at the crate root) gets its own
            // wording, everything else is an ordinary redeclaration.
            let mut path = self.defs.module().to_vec();
            path.push(td.name);
            let existing = self
                .defs
                .lookup_trait(&path)
                .and_then(|d| self.traits.trait_by_def(d));
            let msg = match existing {
                Some(id) if self.traits.trait_def(id).sealed => format!(
                    "`{}` is a built-in trait and cannot be redeclared",
                    self.sym(td.name)
                ),
                _ => format!("trait `{}` is defined more than once", self.sym(td.name)),
            };
            self.error(span, msg);
            return None;
        };
        let id = TraitId(self.traits.traits_len() as u32);
        self.traits.add_trait(crate::semi::traits::def::TraitDef {
            id,
            def,
            visibility: td.visibility,
            sealed,
            // Placeholder until `populate_trait_params`; nothing reads a
            // stub's binder before that phase runs.
            self_param: GenericId(0),
            params: Vec::new(),
            supertraits: Vec::new(),
            methods: Vec::new(),
            assoc_tys: Vec::new(),
            span,
            file: self.current_file,
        });
        Some(id)
    }

    /// Phase C1: allocate the trait's own binder — the implicit `Self` and
    /// the declared parameters — for every stub before any supertrait or
    /// member resolves against it.
    fn populate_trait_params(&mut self, td: &surface::TraitDecl, id: TraitId, span: Option<Span>) {
        let self_param = self.fresh_generic(self.self_name, Vec::new());
        if td.generics.iter().any(|&(name, _)| name == self.self_name) {
            self.error(span, "`Self` is implicit in a trait; do not declare it");
        }
        let self_name = self.self_name;
        let params: Vec<(TokenKey, GenericId)> = td
            .generics
            .iter()
            .filter(|(name, _)| *name != self_name)
            .map(|(name, bounds)| {
                let bounds = self.resolve_bounds(bounds, span);
                (*name, self.fresh_generic(*name, bounds))
            })
            .collect();
        let def = self.traits.trait_def_mut(id);
        def.self_param = self_param;
        def.params = params;
    }

    /// Phase C2: resolve supertraits and member signatures (record stubs
    /// exist by now, so signature types evaluate).
    fn populate_trait_members(&mut self, td: &surface::TraitDecl, id: TraitId, span: Option<Span>) {
        let self_param = self.traits.trait_def(id).self_param;
        let trait_params = self.traits.trait_def(id).params.clone();
        let self_ty = self.tcx.mk_generic(self_param);

        let supertraits: Vec<crate::semi::traits::TraitRef<'tcx>> = td
            .supertraits
            .iter()
            .filter_map(|sup_ref| {
                // Trait references are not parameterized yet: the surface
                // shape carries arguments (and the core `TraitRef`,
                // serialization, and reload all plumb them), but admitting
                // them here needs their evaluation under the trait's binder
                // — deferred, so a spelled argument list is a clean
                // rejection rather than a silent drop.
                if !sup_ref.args.is_empty() {
                    self.error(
                        span,
                        "parameterized supertrait references are not supported yet",
                    );
                    return None;
                }
                let sup = self.resolve_bound(&sup_ref.path, span)?;
                let arity = self.traits.trait_def(sup).params.len();
                if arity != 0 {
                    self.error(
                        span,
                        format!(
                            "trait `{}` takes {arity} type argument(s); parameterized \
                             bounds are not supported here",
                            self.trait_display(sup),
                        ),
                    );
                    return None;
                }
                Some(crate::semi::traits::TraitRef {
                    trait_id: sup,
                    args: vec![self_ty],
                })
            })
            .collect();

        let mut methods: Vec<crate::semi::traits::def::MethodSig<'tcx>> = Vec::new();
        for member in &td.members {
            let mspan = Some(Span {
                start: member.start,
                end: member.end,
            });
            if methods.iter().any(|m| m.name == member.value.name) {
                self.error(
                    mspan,
                    format!(
                        "trait `{}` declares method `{}` more than once",
                        self.sym(td.name),
                        self.sym(member.value.name)
                    ),
                );
                continue;
            }
            methods.extend(self.trait_method_sig(&member.value, mspan, &trait_params, self_ty));
        }
        let def = self.traits.trait_def_mut(id);
        def.supertraits = supertraits;
        def.methods = methods;
    }

    /// One member of a `trait` body as a [`MethodSig`]: allocate its own
    /// binder after the trait's, evaluate the positional parameter types,
    /// and derive the receiver form from the first parameter. `None` (with
    /// a report) if no `self` receiver leads.
    ///
    /// [`MethodSig`]: crate::semi::traits::def::MethodSig
    fn trait_method_sig(
        &mut self,
        func: &surface::Function,
        mspan: Option<Span>,
        trait_params: &[(TokenKey, GenericId)],
        self_ty: Ty<'tcx>,
    ) -> Option<crate::semi::traits::def::MethodSig<'tcx>> {
        use crate::semi::traits::def::ReceiverForm;

        // The member's binder: the trait's params, then its own.
        self.generic_names = trait_params.iter().copied().collect();
        let own: Vec<(TokenKey, GenericId)> = func
            .generics
            .iter()
            .map(|(name, bounds)| {
                if trait_params.iter().any(|(n, _)| n == name) {
                    self.error(
                        mspan,
                        format!(
                            "generic `{0}` on the method shadows the trait's generic `{0}`",
                            self.sym(*name)
                        ),
                    );
                }
                let bounds = self.resolve_bounds(bounds, mspan);
                let g = self.fresh_generic(*name, bounds);
                self.generic_names.insert(*name, g);
                (*name, g)
            })
            .collect();
        self.self_alias = Some(self_ty);

        let params: Vec<Ty<'tcx>> = func
            .params
            .iter()
            .map(|(_, ty, flex)| self.eval_type_flex(ty, *flex))
            .collect();
        // The receiver is the leading parameter named `self`; its evaluated
        // type (or the `[flex]` marker) picks the form.
        let receiver = func
            .params
            .first()
            .filter(|(pname, _, _)| self.sym(*pname) == "self")
            .map(|&(_, _, flex)| {
                if flex {
                    ReceiverForm::Flex
                } else {
                    match params[0].kind() {
                        TyKind::Arc(inner) if *inner == self_ty => ReceiverForm::Arc,
                        _ if params[0] == self_ty => ReceiverForm::Value,
                        _ => {
                            self.error(
                                mspan,
                                "the receiver of a trait method must be `Self`, \
                                 `Arc<Self>`, or `[flex] Self`",
                            );
                            ReceiverForm::Value
                        }
                    }
                }
            });
        let ret = match &func.return_type {
            Some((ty, flex)) => self.eval_type_flex(ty, *flex),
            None => self.tcx.mk_unit(),
        };
        self.generic_names.clear();
        self.self_alias = None;

        // No leading `self` parameter is an *associated function*: every
        // parameter is ordinary, and the call forms are `Trait::f(…)`,
        // `Type::f(…)`, and `G::f(…)` rather than dot dispatch.
        if receiver == Some(ReceiverForm::Flex) && !func.is_regional {
            self.error(
                mspan,
                "a `[flex] Self` receiver requires a `regional fn`: a flex \
                 value cannot escape its region",
            );
        }
        Some(crate::semi::traits::def::MethodSig {
            name: func.name,
            generics: own,
            receiver,
            params,
            ret,
            is_regional: func.is_regional,
            span: mspan,
        })
    }

    /// Phase D: a tricolor DFS over the batch's supertrait edges. A back
    /// edge is reported and SEVERED — `TraitDb::reaches` recurses without a
    /// guard, so a surviving cycle would overflow the checker later.
    fn reject_supertrait_cycles(&mut self, batch: impl Iterator<Item = TraitId>) {
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Gray,
            Black,
        }
        let n = self.traits.traits_len();
        let mut color = vec![Color::White; n];
        for root in batch {
            // Iterative DFS with an explicit stack of (node, next-edge).
            let mut stack: Vec<(TraitId, usize)> = vec![(root, 0)];
            if color[root.0 as usize] != Color::White {
                continue;
            }
            color[root.0 as usize] = Color::Gray;
            while let Some(&mut (node, ref mut edge)) = stack.last_mut() {
                let supers = &self.traits.trait_def(node).supertraits;
                if *edge >= supers.len() {
                    color[node.0 as usize] = Color::Black;
                    stack.pop();
                    continue;
                }
                let next = supers[*edge].trait_id;
                *edge += 1;
                match color[next.0 as usize] {
                    Color::White => {
                        color[next.0 as usize] = Color::Gray;
                        stack.push((next, 0));
                    }
                    Color::Gray => {
                        // Back edge: node -> next closes a cycle.
                        let span = self.traits.trait_def(node).span;
                        let chain = format!(
                            "trait `{0}` participates in a super-trait cycle: \
                             `{0}`: `{1}`",
                            self.trait_display(node),
                            self.trait_display(next),
                        );
                        self.error(span, chain);
                        let idx = *edge - 1;
                        self.traits.trait_def_mut(node).supertraits.remove(idx);
                        *stack.last_mut().map(|(_, e)| e).unwrap_or(&mut 0) -= 1;
                    }
                    Color::Black => {}
                }
            }
        }
    }

    /// Allocate generics for a list of `(name, bounds)` declarations.
    pub(super) fn collect_generics(
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
                        let (name, ty, mutable, vis) = &f.value;
                        let fty = self.field_ty(ty, *mutable);
                        self.reject_arc_member(default_cap, fty, span);
                        if *mutable {
                            self.note_link_element(fty, &mut regional_generics, span);
                        }
                        (*name, fty, *mutable, *vis)
                    })
                    .collect(),
            ),
            surface::RecordFields::Unnamed(fs) => RecordFields::Unnamed(
                fs.iter()
                    .map(|f| {
                        let (ty, mutable, vis) = &f.value;
                        let fty = self.field_ty(ty, *mutable);
                        self.reject_arc_member(default_cap, fty, span);
                        if *mutable {
                            self.note_link_element(fty, &mut regional_generics, span);
                        }
                        (fty, *mutable, *vis)
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
                                    self.reject_arc_member(default_cap, fty, span);
                                    fty
                                })
                                .collect(),
                        }
                    })
                    .collect(),
            ),
            surface::RecordFields::Opaque => RecordFields::Opaque,
        };
        self.generic_names.clear();

        // Validate that mutable (`[field]`) fields require a regional record.
        if default_cap != DefaultCap::Regional {
            let has_mut = match &fields {
                RecordFields::Named(fs) => fs.iter().any(|(_, _, m, _)| *m),
                RecordFields::Unnamed(fs) => fs.iter().any(|(_, m, _)| *m),
                RecordFields::Variants(_) | RecordFields::Opaque => false,
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
            Some(RecordFields::Named(fs)) => fs.iter().map(|(_, t, _, _)| *t).collect(),
            Some(RecordFields::Unnamed(fs)) => fs.iter().map(|(t, _, _)| *t).collect(),
            Some(RecordFields::Variants(vs)) => {
                vs.iter().flat_map(|v| v.fields.iter().copied()).collect()
            }
            Some(RecordFields::Opaque) | None => Vec::new(),
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
            let mut stack: Vec<(DefId, Vec<DefId>)> = vec![(root, self.inline_value_heads(root))];
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
                        let (span, file) = (self.records[&next].span, self.records[&next].file);
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
    /// Reject an `Arc` member of a `[regional]` record — **not implemented
    /// for now**: nothing in the current machinery is known to break (the
    /// region scanner only walks `[field]` links, and member release goes
    /// through the ordinary drop glue at the written member type), but the
    /// interaction with the deferred sync-regional design (§6 —
    /// freeze/rigid release, region teardown) is deliberately left
    /// undecided. Shared and `[value]` records may hold `Arc` members —
    /// the member type spells its own atomic link (§3.3).
    fn reject_arc_member(&mut self, record_cap: DefaultCap, fty: Ty<'tcx>, span: Option<Span>) {
        if record_cap == DefaultCap::Regional && matches!(fty.kind(), TyKind::Arc(_)) {
            self.error(
                span,
                "an `Arc` member of a `[regional]` record is not implemented yet",
            );
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
