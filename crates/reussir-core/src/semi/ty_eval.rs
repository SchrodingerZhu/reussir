//! Evaluating surface types into interned [`Ty`]s, including the capability
//! ("flexivity") coloring of regional records.
//!
//! A `[regional]` record is a type that can be *created locally* inside a region.
//! Flexivity is the per-use color such a value carries; the four [`Flexivity`]
//! states mean:
//!
//! * **Flex** — a live, freshly created regional object. Its `[field]` links can
//!   be mutated while inside the region, and it can be passed to a `regional`
//!   function, but it *cannot be materialized*: it cannot escape the region —
//!   neither returned/frozen-out without first freezing, nor **captured by a
//!   closure** (a capture would let it outlive the region).
//! * **Rigid** — the frozen form: immutable, and *materializable* (it may escape
//!   the region). Freezing on region exit turns Flex into Rigid.
//! * **Regional** — not a value's birth state, but the "inherit my container's
//!   flexivity" color of a `[field]` mutable link: read through a Flex container
//!   it is mutable, through a Rigid one it is frozen. Because nested links
//!   *inherit* rather than store their own flexivity, freezing only the *head*
//!   re-interprets them all at once — so [`Elaborator::freeze_region`] is O(1),
//!   with no recursion.
//! * **Irrelevant** — a non-regional (value/shared) record; primitives carry no
//!   flexivity.
//!
//! Lifecycle: create ⟶ Flex ──(freeze on escape)──▶ Rigid (materializable).
//!
//! Representation note: a regional record's *type annotation* is evaluated by
//! `eval_type` to an unpinned [`Flexivity::Regional`], which
//! [`Elaborator::eval_type_flex`] then pins to Flex (`[flex]`) or Rigid. A
//! freshly *constructed* regional record is different: it is born Flex directly
//! (the checker's `record_ty`) — the live, region-local form, so it is
//! immediately assignable and freezes to Rigid on region exit
//! ([`Elaborator::freeze_region`] maps Flex/Regional → Rigid). Because `unify`
//! ignores flexivity, that Flex value still flows into a binding/return annotated
//! otherwise (e.g. `[rigid]`), where the annotation's coloring is what is read. A
//! `[field]` mutable link is wrapped in `Nullable` (nullable, and
//! reassignable in-region), so *assembling* (assigning) and *projecting*
//! (reading) those fields ride the `Nullable` machinery — and so
//! [`Elaborator::refine_flex`] looks through a `Nullable` head when pinning a
//! binding. Generics are left uncolored — their modality is resolved at
//! monomorphization.

use crate::semi::infer::Instantiation;
use crate::semi::traits::sync::{SyncEnv, SyncVerdict, wf_arc};
use crate::semi::ty::{DefId, Flexivity, FpTy, IntTy, Ty, TyKind};
use crate::surface::{self, FpType, IntegralType, TypeKind};

use super::ctxt::{DefaultCap, Elaborator, RecordFields};

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Evaluate a surface type, coloring concrete regional records as
    /// `Regional` (later refined by [`Elaborator::eval_type_flex`]).
    pub fn eval_type(&mut self, ty: &surface::Type) -> Ty<'tcx> {
        let kind = ty.kind();
        match &kind {
            TypeKind::TypeIntegral(it) => self.tcx.mk_int(match it {
                IntegralType::Signed(w) => IntTy::Signed(*w),
                IntegralType::Unsigned(w) => IntTy::Unsigned(*w),
            }),
            TypeKind::TypeFp(fp) => self.tcx.mk_fp(match fp {
                FpType::Ieee(w) => FpTy::Ieee(*w),
                FpType::BFloat16 => FpTy::BFloat16,
                FpType::Float8 => FpTy::Float8,
            }),
            TypeKind::TypeBool => self.tcx.mk_bool(),
            TypeKind::TypeStr => self.tcx.mk_str(),
            TypeKind::TypeChar => self.tcx.mk_char(),
            TypeKind::TypeUnit => self.tcx.mk_unit(),
            TypeKind::TypeArrow(args, ret) => {
                let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
                let ret = self.eval_type(ret);
                self.tcx.mk_closure(&args, ret)
            }
            TypeKind::TypeExpr(path, args) => self.eval_type_expr(path, args, ty.span()),
            TypeKind::TypeArray(elem, extents) => self.eval_array_type(elem, extents, ty.span()),
        }
    }

    fn eval_type_expr(
        &mut self,
        path: &surface::Path,
        args: &[surface::Type],
        span: surface::Span,
    ) -> Ty<'tcx> {
        let key = path.basename;

        // A bare generic parameter in scope.
        if path.segments.is_empty()
            && let Some(&generic) = self.generic_names.get(&key)
        {
            if !args.is_empty() {
                self.error(
                    Some(span),
                    format!("generic `{}` cannot take type arguments", self.sym(key)),
                );
            }
            return self.tcx.mk_generic(generic);
        }

        // Built-in type constructors are roots and never module-qualified.
        // `Cell<T>` is the plain get/set cell; `RefCell<T>` is the exclusive
        // flavor that additionally supports guarded read-modify-write.
        if path.segments.is_empty() && matches!(self.sym(key), "Cell" | "RefCell") {
            let exclusive = self.sym(key) == "RefCell";
            let name = if exclusive { "RefCell" } else { "Cell" };
            return match args {
                [inner] => {
                    let inner = self.eval_type(inner);
                    self.tcx.mk_cell(inner, exclusive)
                }
                _ => {
                    self.error(
                        Some(span),
                        format!("`{name}` takes exactly one type argument"),
                    );
                    self.tcx.mk(TyKind::Bottom)
                }
            };
        }

        // The built-in nullable type (a root builtin: never module-qualified).
        if path.segments.is_empty() && self.sym(key) == "Nullable" {
            return match args {
                [inner] => {
                    let inner = self.eval_type(inner);
                    // A nullable only accepts a non-null pointer-like inner
                    // (rc/record/closure/…): it encodes the null case in the
                    // pointer's own representation, so a scalar or a nested
                    // nullable has no place to store the discriminant. Reject a
                    // *concretely* non-pointer inner now; generics/holes are left
                    // for the later groundness check, and records are the
                    // intended case.
                    if is_concretely_non_pointer(inner) {
                        self.error(
                            Some(span),
                            format!(
                                "`Nullable` inner type `{}` is not a pointer-like type",
                                self.ty_display(inner)
                            ),
                        );
                    }
                    self.tcx.mk_nullable(inner)
                }
                _ => {
                    self.error(Some(span), "`Nullable` takes exactly one type argument");
                    self.tcx.mk(TyKind::Bottom)
                }
            };
        }

        // The built-in atomic-rc coloring (a root builtin: never
        // module-qualified). `Arc<X>` is the shared rc box `X` — a `[shared]`
        // record, an array, or a closure — whose refcount is adjusted
        // atomically, so the value may cross threads; everything but the rc
        // discipline is `X`'s own.
        if path.segments.is_empty() && self.sym(key) == "Arc" {
            return match args {
                [inner] => {
                    let inner = self.eval_type(inner);
                    // Only a shared rc box can be arc-colored: the box layout
                    // is the inner type's own, with atomic counting. A
                    // generic/hole/bottom inner is deferred to the
                    // instantiation check (it lands with `Arc` construction;
                    // until then a generic inner can only be grounded from an
                    // already-checked `Arc` annotation).
                    if let Some(kind) = self.arc_inner_rejection(inner) {
                        self.error(
                            Some(span),
                            format!(
                                "`Arc` inner type `{}` is not a `[shared]` record, array, or closure ({kind})",
                                self.ty_display(inner)
                            ),
                        );
                    } else if self.records_complete {
                        // Body-position annotations run after record fields
                        // populate, so the member (`Sync`) half of the wf
                        // check answers here. Signature and member positions
                        // evaluate earlier and blocked — the post-populate
                        // sweep re-checks them.
                        self.report_arc_wf(inner, Some(span));
                    }
                    self.tcx.mk_arc(inner)
                }
                _ => {
                    self.error(Some(span), "`Arc` takes exactly one type argument");
                    self.tcx.mk(TyKind::Bottom)
                }
            };
        }

        // A user record, resolved to its def — bare in the current module, or
        // module-qualified (`utils::math::Widget`, `root::…`, `super::…`).
        return self.eval_record_type_expr(path, args, span, key);
    }

    /// [`arc_inner_rejection`] against this elaborator's record table.
    pub(crate) fn arc_inner_rejection(&self, t: Ty<'tcx>) -> Option<&'static str> {
        arc_inner_rejection(|def| self.records.get(&def).map(|r| r.default_cap), t)
    }

    /// Evaluate a statically shaped array type (`[f64; 512]`,
    /// `[f64; 5, 16, 8]`) into a [`TyKind::Array`].
    pub(crate) fn eval_array_type(
        &mut self,
        elem: &surface::Type,
        extents: &[surface::Expr],
        span: surface::Span,
    ) -> Ty<'tcx> {
        if extents.is_empty() {
            // Only reachable through parser recovery (`[T; ]` is a syntax
            // error); avoid constructing a rank-0 array on top of it.
            return self.tcx.mk(TyKind::Bottom);
        }
        let elem = self.eval_type(elem);
        // Phase A restricts elements to plain scalars: views over the
        // payload must be free of reference-count traffic.
        if !is_plain_scalar_or_deferred(elem) {
            self.error(
                Some(span),
                format!(
                    "non-scalar array element types are not supported yet; found `{}`",
                    self.ty_display(elem)
                ),
            );
        }
        let mut dims = Vec::with_capacity(extents.len());
        let mut product: u128 = 1;
        for e in extents {
            let extent = self.eval_extent(e);
            if extent == 0 {
                self.error(Some(e.span()), "array extents must be positive");
            }
            product = product.saturating_mul(u128::from(extent.max(1)));
            dims.push(extent.max(1));
        }
        if product > (1u128 << 32) {
            self.error(Some(span), "array is too large (more than 2^32 elements)");
        }
        self.tcx.mk_array(elem, &dims)
    }

    /// Evaluate one array extent expression. The grammar admits any
    /// expression; for now only an integer literal evaluates — everything
    /// else reports and recovers with extent 1. An out-of-range literal
    /// clamps; the product check reports it as too large.
    fn eval_extent(&mut self, e: &surface::Expr) -> u64 {
        match e.kind() {
            surface::ExprKind::ConstExpr(crate::surface::Const::ConstInt(n)) => {
                u64::try_from(&n).unwrap_or(u64::MAX)
            }
            _ => {
                self.error(
                    Some(e.span()),
                    "this kind of array extent cannot be evaluated yet (only integer literals are supported)",
                );
                1
            }
        }
    }

    fn eval_record_type_expr(
        &mut self,
        path: &surface::Path,
        args: &[surface::Type],
        span: surface::Span,
        key: reussir_syntax::kind::TokenKey,
    ) -> Ty<'tcx> {
        let Some(def) = self.resolve_record_ref(path) else {
            let hint = if path.segments.is_empty() {
                self.record_suggestion(key)
            } else {
                String::new()
            };
            let shown = self.path_display(path);
            self.error(Some(span), format!("unknown type `{shown}`{hint}"));
            return self.tcx.mk(TyKind::Bottom);
        };
        self.record_use(key);
        let default_cap = self.records[&def].default_cap;
        let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
        let flex = match default_cap {
            DefaultCap::Value | DefaultCap::Shared => Flexivity::Irrelevant,
            DefaultCap::Regional => Flexivity::Regional,
        };
        self.tcx.mk_record(def, &args, flex)
    }

    /// Evaluate a type that carries a `[flex]` flag (a parameter, return type,
    /// or `let` binding). A regional record refines to `Flex` or `Rigid`.
    pub fn eval_type_flex(&mut self, ty: &surface::Type, is_flex: bool) -> Ty<'tcx> {
        let t = self.eval_type(ty);
        self.refine_flex(t, is_flex)
    }

    /// Refine a regional record's flexivity by a `[flex]` flag.
    pub fn refine_flex(&self, t: Ty<'tcx>, is_flex: bool) -> Ty<'tcx> {
        let refined = if is_flex {
            Flexivity::Flex
        } else {
            Flexivity::Rigid
        };
        match t.kind() {
            TyKind::Record {
                def,
                args,
                flex: Flexivity::Regional,
            } => self.tcx.mk_record(*def, args, refined),
            TyKind::Nullable(inner) => {
                if let TyKind::Record {
                    def,
                    args,
                    flex: Flexivity::Regional,
                } = inner.kind()
                {
                    let inner = self.tcx.mk_record(*def, args, refined);
                    self.tcx.mk_nullable(inner)
                } else {
                    t
                }
            }
            _ => t,
        }
    }

    /// Freeze a value escaping a region: peel any `Nullable` wrapper and turn a
    /// `Flex` (or still-unpinned `Regional`) record *head* into `Rigid`, the
    /// materializable frozen form.
    ///
    /// It re-colors only the head (through `Nullable` wrappers); it does *not*
    /// descend into a record's fields or type arguments. That is sound, not a
    /// shortcut: a `[field]` link is colored `Regional` (= inherit the
    /// container's flexivity), never independently `Flex`, so once the head is
    /// `Rigid` every link reads as frozen. The other field kinds — primitive,
    /// shared/value rc (`Irrelevant`), already-rigid rc — are flex-invariant.
    pub fn freeze_region(&self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.kind() {
            TyKind::Record {
                def,
                args,
                flex: Flexivity::Regional | Flexivity::Flex,
            } => self.tcx.mk_record(*def, args, Flexivity::Rigid),
            TyKind::Nullable(inner) => {
                let inner = self.freeze_region(*inner);
                self.tcx.mk_nullable(inner)
            }
            _ => t,
        }
    }
}

/// Is `t` *concretely* a non-pointer-like type — i.e. one that cannot be the
/// inner of a `Nullable` (which needs a pointer-like inner to encode the null
/// case)? Conservative: only the by-value scalars and a nested `Nullable` are
/// rejected; `Record`/`Closure`/`Array`/`Cell` are valid inners, and
/// `Generic`/`Hole`/`Bottom` are deferred (resolved/checked later) so this never
/// reports a false positive on a not-yet-ground type.
/// Whether `t` qualifies as a Phase-A array element: a plain scalar, or a
/// generic/hole deferred to the monomorphization-time groundness check.
pub(crate) fn is_plain_scalar_or_deferred(t: Ty<'_>) -> bool {
    matches!(
        t.kind(),
        TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Generic(_)
            | TyKind::Hole(_)
            | TyKind::Bottom
    )
}

/// Why `t` cannot be the inner of an `Arc`, or `None` when it can — a shared
/// rc box: a `[shared]` record, an array, or a closure — or when the answer
/// must wait (generic/hole/bottom). `cap_of` looks up a record's default
/// capability; an unknown record defers rather than rejecting. Shared between
/// `ty_eval` (a concrete annotation) and monomorphization (the deferred half:
/// a generic inner is first grounded by an instantiation, e.g. a trampoline's
/// explicit type arguments).
pub fn arc_inner_rejection<'tcx>(
    cap_of: impl FnOnce(DefId) -> Option<DefaultCap>,
    t: Ty<'tcx>,
) -> Option<&'static str> {
    match t.kind() {
        TyKind::Record { def, .. } => match cap_of(*def) {
            Some(DefaultCap::Shared) | None => None,
            Some(DefaultCap::Value) => Some("a `[value]` record"),
            Some(DefaultCap::Regional) => Some("a `[regional]` record"),
        },
        // An array or a closure is one shared rc box exactly like a
        // `[shared]` record, so it takes the atomic coloring the same way.
        TyKind::Array { .. } | TyKind::Closure { .. } => None,
        TyKind::Generic(_) | TyKind::Hole(_) | TyKind::Bottom => None,
        // A cell is a shared box too, but synchronization is the cell's own
        // axis (plain vs exclusive, and the atomic flavor at the MLIR level)
        // — `Arc` must not stack a second discipline on top.
        TyKind::Cell { .. } => Some("a cell"),
        TyKind::Arc(_) => Some("already an `Arc`"),
        TyKind::Nullable(_) => Some("a nullable"),
        // Scalars, `str`, unit: not rc-managed boxes at all.
        _ => Some("not an rc box"),
    }
}

fn is_concretely_non_pointer(t: Ty<'_>) -> bool {
    matches!(
        t.kind(),
        TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Str
            | TyKind::Char
            | TyKind::Unit
            | TyKind::Nullable(_)
    )
}

//===------------------------------------------------------------------===//
// Structural `Sync` over the elaborator's tables
//===------------------------------------------------------------------===//

/// [`SyncEnv`] over the elaborator: record capabilities from the collected
/// table, member types instantiated through the inference context.
pub(crate) struct ElabSyncEnv<'e, 'a, 'tcx> {
    pub el: &'e Elaborator<'a, 'tcx>,
}

impl<'tcx> SyncEnv<'tcx> for ElabSyncEnv<'_, '_, 'tcx> {
    fn default_cap(&self, def: DefId) -> Option<DefaultCap> {
        self.el.records.get(&def).map(|r| r.default_cap)
    }

    fn members(&self, def: DefId, args: &'tcx [Ty<'tcx>]) -> Option<Vec<(String, Ty<'tcx>)>> {
        let rec = self.el.records.get(&def)?;
        let fields = rec.fields.as_ref()?;
        let inst = Instantiation::from_pairs(
            rec.ty_params
                .iter()
                .map(|(_, g)| *g)
                .zip(args.iter().copied()),
        );
        let subst = |t: Ty<'tcx>| self.el.infer.instantiate_ty(t, &inst);
        Some(match fields {
            RecordFields::Named(fs) => fs
                .iter()
                .map(|(n, t, _)| (self.el.sym(*n).to_owned(), subst(*t)))
                .collect(),
            RecordFields::Unnamed(fs) => fs
                .iter()
                .enumerate()
                .map(|(i, (t, _))| (i.to_string(), subst(*t)))
                .collect(),
            RecordFields::Variants(vs) => vs
                .iter()
                .flat_map(|v| {
                    let vname = self.el.sym(v.name);
                    v.fields
                        .iter()
                        .enumerate()
                        .map(move |(i, t)| (format!("{vname}.{i}"), *t))
                })
                .map(|(label, t)| (label, subst(t)))
                .collect(),
        })
    }
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Report an ill-formed `Arc<inner>` whose kind is admissible but whose
    /// contents are not `Sync`. Call only where records are complete (the
    /// post-populate sweep and body checking); a blocked verdict defers to
    /// the monomorphization backstop.
    pub(crate) fn report_arc_wf(&mut self, inner: Ty<'tcx>, span: Option<surface::Span>) {
        if self.arc_inner_rejection(inner).is_some() {
            // The kind gate owns this error at its own site.
            return;
        }
        if let SyncVerdict::NotSync(w) = wf_arc(&ElabSyncEnv { el: self }, inner) {
            self.error(
                span,
                format!(
                    "`Arc<{}>` is ill-formed: {}; every member of an `Arc` inner must be `Sync`",
                    self.ty_display(inner),
                    w.describe()
                ),
            );
        }
    }

    /// Walk a signature or member type and wf-check every `Arc` node inside
    /// it. Function prototypes are scanned before record fields populate, so
    /// their annotation-site checks blocked; the post-populate sweep re-runs
    /// them through this walker (mono's `check_arc_inners` is the ground
    /// backstop for what still blocks here, e.g. generic inners).
    pub(crate) fn sweep_arc_wf(&mut self, ty: Ty<'tcx>, span: Option<surface::Span>) {
        match *ty.kind() {
            TyKind::Arc(inner) => {
                self.report_arc_wf(inner, span);
                self.sweep_arc_wf(inner, span);
            }
            TyKind::Record { args, .. } => {
                for &arg in args {
                    self.sweep_arc_wf(arg, span);
                }
            }
            TyKind::Closure { params, ret } => {
                for &p in params {
                    self.sweep_arc_wf(p, span);
                }
                self.sweep_arc_wf(ret, span);
            }
            TyKind::Nullable(inner) => self.sweep_arc_wf(inner, span),
            TyKind::Cell { elem, .. } => self.sweep_arc_wf(elem, span),
            TyKind::Array { elem, .. } => self.sweep_arc_wf(elem, span),
            _ => {}
        }
    }
}
