//! Evaluating surface types into interned [`Ty`]s, including the capability
//! ("flexivity") coloring of regional records.
//!
//! A `[regional]` record is a type that can be *created locally* inside a region.
//! Flexivity is the per-use color such a value carries; the four [`Capability`]
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
//! `eval_type` to an unpinned [`Capability::Regional`], which
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

use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyKind};
use crate::surface::{self, FpType, IntegralType, TypeKind};

use super::ctxt::{DefaultCap, Elaborator};

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
            TypeKind::TypeUnit => self.tcx.mk_unit(),
            TypeKind::TypeArrow(args, ret) => {
                let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
                let ret = self.eval_type(ret);
                self.tcx.mk_closure(&args, ret)
            }
            TypeKind::TypeExpr(path, args) => self.eval_type_expr(path, args, ty.span()),
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

        // The built-in nullable type.
        if self.sym(key) == "Nullable" {
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
                            format!("`Nullable` inner type `{inner:?}` is not a pointer-like type"),
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

        // A user record, resolved to its def.
        let Some(def) = self.defs.resolve_record(key) else {
            self.error(Some(span), format!("unknown type `{}`", self.sym(key)));
            return self.tcx.mk(TyKind::Bottom);
        };
        let default_cap = self.records[&def].default_cap;
        let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
        let flex = match default_cap {
            DefaultCap::Value | DefaultCap::Shared => Capability::Irrelevant,
            DefaultCap::Regional => Capability::Regional,
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
            Capability::Flex
        } else {
            Capability::Rigid
        };
        match t.kind() {
            TyKind::Record {
                def,
                args,
                flex: Capability::Regional,
            } => self.tcx.mk_record(*def, args, refined),
            TyKind::Nullable(inner) => {
                if let TyKind::Record {
                    def,
                    args,
                    flex: Capability::Regional,
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
                flex: Capability::Regional | Capability::Flex,
            } => self.tcx.mk_record(*def, args, Capability::Rigid),
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
/// rejected; `Record`/`Closure` are valid inners, and `Generic`/`Hole`/`Bottom`
/// are deferred (resolved/checked later) so this never reports a false positive
/// on a not-yet-ground type.
fn is_concretely_non_pointer(t: Ty<'_>) -> bool {
    matches!(
        t.kind(),
        TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Str
            | TyKind::Unit
            | TyKind::Nullable(_)
    )
}
