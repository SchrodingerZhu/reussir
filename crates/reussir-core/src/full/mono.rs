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
//! Every non-generic function (emitted even if uncalled — whole-program DCE is
//! the backend's job) plus every trampoline target. Generic functions are
//! emitted once per distinct instantiation discovered from a reachable body.
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

use std::collections::VecDeque;

use lasso::Rodeo;
use rustc_hash::{FxHashMap, FxHashSet};

use reussir_syntax::kind::{Resolver, TokenKey};
use reussir_syntax::source::FileId;

use crate::full::mangle::Mangler;
use crate::full::mir;
use crate::full::subst::{Subst, subst_ty};
use crate::literal;
use crate::semi::ctxt::{
    DefaultCap, Elaborator, Record, RecordFields, Report, Severity, TrampolineRoot,
};
use crate::semi::hir::{self, DecisionTree, Expr, ExprKind, Function, SwitchCases};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{DefId, Flexivity, Ty, TyCtxt, TyKind};
use crate::semi::ty::{FpTy, IntTy};
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
    pub strings: Vec<(StringToken, String)>,
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
            strings: self.strings.entries(),
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

/// Monomorphize an elaborated program into its ground Full MIR, alongside any
/// diagnostics raised by the call-boundary regional check.
pub fn monomorphize<'a, 'tcx>(input: &MonoInput<'a, 'tcx>) -> (mir::Program<'tcx>, Vec<Report>) {
    let tcx: &'a TyCtxt<'tcx> = input.tcx;
    let by_def: FxHashMap<DefId, &Function<'tcx>> =
        input.elaborated.iter().map(|f| (f.def, f)).collect();

    let mut driver = Driver {
        tcx,
        mangler: Mangler::new(input.defs, input.resolver),
        symbols: Rodeo::default(),
        queue: VecDeque::new(),
        seen: FxHashSet::default(),
        records: FxHashSet::default(),
        reports: Vec::new(),
        cur_file: FileId::ROOT,
        ids: mir::ExprIdGen::default(),
    };

    // Seed roots: every non-generic function, then every trampoline target.
    for f in input.elaborated {
        if f.generics.is_empty() {
            driver.enqueue(f.def, tcx.intern_tys(&[]));
        }
    }
    for t in input.trampolines {
        driver.enqueue(t.target, tcx.intern_tys(&t.ty_args));
    }

    let mut functions = Vec::new();
    while let Some(inst) = driver.queue.pop_front() {
        let Some(func) = by_def.get(&inst.def) else {
            // No body/prototype recorded (e.g. an item that failed to elaborate).
            continue;
        };
        driver.cur_file = func.file;

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
                mir::Param {
                    name: *name,
                    var: *var,
                    ty,
                }
            })
            .collect();
        let return_ty = subst_ty(tcx, func.return_ty, &subst);
        driver.note_records(return_ty);
        let body = func.body.as_ref().map(|b| driver.lower_ref(b, &subst));

        let symbol = driver.symbol_of(inst.def, inst.ty_args);
        functions.push(mir::Function {
            symbol,
            visibility: func.visibility,
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
        records.push(mir::RecordInstance {
            symbol,
            ty,
            default_cap,
            layout,
        });
    }

    let trampolines: Vec<mir::Trampoline> = input
        .trampolines
        .iter()
        .map(|t| mir::Trampoline {
            export: mir::Symbol(driver.symbols.get_or_intern(&t.name)),
            abi: t.abi.clone(),
            target: driver.symbol_of(t.target, tcx.intern_tys(&t.ty_args)),
        })
        .collect();

    // Deterministic output: sort by mangled text.
    functions.sort_by_cached_key(|f| driver.symbols.resolve(&f.symbol.0));
    records.sort_by_cached_key(|r| driver.symbols.resolve(&r.symbol.0));

    let Driver {
        symbols, reports, ..
    } = driver;
    (
        mir::Program {
            functions,
            records,
            trampolines,
            string_literals: input.strings.clone(),
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
        // A record that failed field population elaborates with no fields; treat
        // it as an empty compound so lowering still has a well-formed layout.
        None => mir::RecordLayout::Compound(&[]),
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

/// Mutable worklist + lowering state.
struct Driver<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    mangler: Mangler<'a>,
    /// Interner for every mangled symbol; moved into the final `mir::Program`.
    symbols: Rodeo,
    queue: VecDeque<Instance<'tcx>>,
    /// Function instances already enqueued (so each is emitted once).
    seen: FxHashSet<Instance<'tcx>>,
    /// Ground record instances whose layout is needed (keyed flex-independently).
    records: FxHashSet<(DefId, &'tcx [Ty<'tcx>])>,
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

    /// Enqueue a function instance if it has not been seen.
    fn enqueue(&mut self, def: DefId, ty_args: &'tcx [Ty<'tcx>]) {
        let inst = Instance { def, ty_args };
        if self.seen.insert(inst) {
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
                None => Vec::new(),
            };
            for field_ty in field_tys {
                let ground = subst_ty(tcx, field_ty, &subst);
                self.discover_records(ground, &mut worklist);
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
        let kind = self.lower_kind(&e.kind, subst);
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

    fn lower_kind(&mut self, kind: &ExprKind<'tcx>, subst: &Subst<'tcx>) -> mir::ExprKind<'tcx> {
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
                self.enqueue(*target, ty_args);
                let callee = self.symbol_of(*target, ty_args);
                M::Call {
                    callee,
                    args: self.lower_slice(args, subst),
                    regional: *regional,
                }
            }
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

        let source = "struct Pair { a: i32, b: i32 } \
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
            let hir_text = HirPrinter::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            );
            let parsed = parse_program(tcx, &hir_text).expect("re-parse HIR");
            let input = MonoInput {
                tcx,
                defs: &parsed.defs,
                resolver: &parsed.names,
                elaborated: &parsed.funcs,
                records: &parsed.records,
                trampolines: &parsed.trampolines,
                strings: parsed.strings.clone(),
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
            struct Pair { a: i32, b: i32 }
            struct Point { x: i32, y: i32 }

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

            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }

            regional fn fresh<T>(x: T) -> [flex] Cell<T> { Cell { v: x, next: Nullable::Null } }

            regional fn loop_back(seed: i32) -> i32 {
                let c = Cell { v: seed, next: Nullable::Null };
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
            let hir_text = HirPrinter::new(&elab.defs, elab.resolver).program(
                &elab.elaborated,
                &strings,
                &elab.records,
                &elab.trampolines,
            );
            let hir = parse_hir(tcx, &hir_text).expect("re-parse HIR");
            let hir_text2 = HirPrinter::new(&hir.defs, &hir.names).program(
                &hir.funcs,
                &hir.strings,
                &hir.records,
                &hir.trampolines,
            );
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
                strings: hir.strings.clone(),
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
            _ => true,
        }
    }

    /// The immediate sub-expressions of a MIR node (decision-tree arms aside).
    fn children<'tcx>(e: &mir::Expr<'tcx>) -> Vec<&'tcx mir::Expr<'tcx>> {
        use mir::ExprKind::*;
        match e.kind {
            GlobalStr(_) | ConstChar(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_)
            | Poison => vec![],
            Negate(x) | Not(x) | Cast(x, _) | RegionRun(x) | Proj(x, _) => vec![x],
            Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => vec![l, r],
            If(c, t, f) => vec![c, t, f],
            Let { value, .. } => vec![value],
            Seq(es) => es.iter().collect(),
            Call { args, .. } | Ctor { args, .. } | Variant { args, .. } => args.iter().collect(),
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
            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }
            regional fn foo<T>(bar: [flex] T) -> i32 { 0 }
            regional fn use_ok(c: [flex] Cell<i32>) -> i32 { foo(c) }
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
            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }
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
    fn field_link_generic_accepts_regional() {
        // `Wrapper<T>` with `[field] T` instantiated at a regional record is fine.
        let src = r#"
            struct [regional] Cell<T> { v: T, next: [field] Cell<T> }
            struct [regional] Wrapper<T> { inner: [field] T }
            regional fn use_ok(w: [flex] Wrapper<Cell<i32>>) -> i32 { 0 }
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
