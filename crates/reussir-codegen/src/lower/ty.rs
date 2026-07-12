//! Type lowering: ground Reussir types → MLIR types.
//!
//! Scalars map directly to builtin MLIR types. A record has no `DefTable` here,
//! so it resolves through a layout table ([`TypeCtx`]) keyed by its `(def, args)`
//! identity — the ground layout `mono` baked into the MIR.
//!
//! Three record flavours lower, selected by the instance's declared capability:
//! * a **`[value]`** record is an inline aggregate — an identified
//!   `!reussir.record<…>` held directly by value;
//! * a **`[shared]`** record is heap-allocated and reference-counted — the same
//!   identified record type wrapped in a `!reussir.rc<…>` pointer. A shared
//!   record that appears as a field of another record is therefore stored as an
//!   `rc` link, which falls out of lowering its field type here.
//! * a **`[regional]`** record is also a boxed `rc` pointer, but region-allocated;
//!   the box capability is read from the value's flexivity coloring — `flex`
//!   while it is region-local and mutable, `rigid` once it has been frozen out of
//!   its region.
//!
//! Both struct (`compound`) and enum (`variant`) records lower this way. An enum
//! is an identified `!reussir.record<variant …>` whose members are the per-case
//! payload compounds (`{enum}::{case}`); the variant carries a tag plus a payload
//! area sized to the largest case. This module also holds the numeric
//! classification [`expr`](super::expr) uses to choose the right cast op.

use std::cell::RefCell;

use rustc_hash::{FxHashMap, FxHashSet};

use reussir_backend::dialect::ty::{
    ReussirAtomicKind, ReussirCapability, ReussirLifeScope, ReussirRecordKind, array, closure,
    nullable, rc, record_complete_in_place, record_incomplete, record_is_complete, r#ref, region,
    str as str_type,
};
use reussir_backend::melior::Context;
use reussir_backend::melior::ir::Type;
use reussir_backend::melior::ir::attribute::StringAttribute;
use reussir_backend::melior::ir::r#type::IntegerType;

use reussir_core::full::mir;
use reussir_core::semi::ctxt::DefaultCap;
use reussir_core::semi::ty::{DefId, Flexivity, FpTy, IntTy, Ty, TyKind};

use super::{LoweringError, Result, err};

/// The identity of a record instance — its definition and ground type arguments
/// (capability-canonicalized) — used to match a record-typed value to its layout.
type RecordInstanceKey<'tcx> = (DefId, &'tcx [Ty<'tcx>]);

/// Type-lowering state: the MLIR context to build types in, the program (whose
/// symbol table names record types), and the record instances indexed by their
/// `(def, args)` identity so a record-typed value can find its ground layout.
///
/// The MLIR context already uniques identified record types by name, so it is the
/// type cache; `building` is just the set of record symbols whose body is
/// currently being lowered — the construction stack. A record that reaches itself
/// (directly or through an `rc` field) re-enters lowering on the back-edge, finds
/// its symbol already on the stack, and hands back the same (still-incomplete)
/// identified type rather than recursing forever.
pub(super) struct TypeCtx<'c, 'p, 'tcx> {
    context: &'c Context,
    program: &'p mir::Program<'tcx>,
    records: FxHashMap<RecordInstanceKey<'tcx>, &'p mir::RecordInstance<'tcx>>,
    building: RefCell<FxHashSet<mir::Symbol>>,
}

impl<'c, 'p, 'tcx> TypeCtx<'c, 'p, 'tcx> {
    pub(super) fn new(context: &'c Context, program: &'p mir::Program<'tcx>) -> Self {
        let records = program
            .records
            .iter()
            .filter_map(|r| match *r.ty.kind() {
                TyKind::Record { def, args, .. } => Some(((def, args), r)),
                _ => None,
            })
            .collect();
        TypeCtx {
            context,
            program,
            records,
            building: RefCell::new(FxHashSet::default()),
        }
    }

    /// The record instance for a nominal record type, if known.
    pub(super) fn record_of(&self, ty: Ty<'tcx>) -> Option<&'p mir::RecordInstance<'tcx>> {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => self.records.get(&(def, args)).copied(),
            _ => None,
        }
    }

    /// Whether `ty` is a `[shared]` record — one whose value is represented as an
    /// `!reussir.rc<…>` pointer rather than held inline. (Regional records are
    /// also managed, but region-allocated and lowered separately.)
    pub(super) fn is_shared_record(&self, ty: Ty<'tcx>) -> bool {
        self.record_of(ty)
            .is_some_and(|rec| rec.default_cap == DefaultCap::Shared)
    }

    /// Whether `ty` is a `[regional]` record — a region-allocated boxed `rc`
    /// pointer whose `flex`/`rigid` capability comes from its flexivity coloring
    /// (see [`regional_capability`]). The box carries a larger three-word header
    /// than a shared one, so debug info reaches the payload at a different offset.
    pub(super) fn is_regional_record(&self, ty: Ty<'tcx>) -> bool {
        self.record_of(ty)
            .is_some_and(|rec| rec.default_cap == DefaultCap::Regional)
    }

    /// Whether a value of type `ty` is represented as a managed `!reussir.rc<…>`
    /// pointer whose refcount is adjusted directly with `rc.inc`/`rc.dec`: a
    /// `[shared]` record, a regional record (reference-counted once frozen to
    /// `rigid` — the only regional coloring that is a resource), or a closure.
    /// This is the complement, among reference-counted values, of an inline
    /// `[value]` record — which is acquired/dropped through a reference instead.
    pub(super) fn is_managed_rc(&self, ty: Ty<'tcx>) -> bool {
        self.is_shared_record(ty)
            || self.is_regional_record(ty)
            || is_closure(ty)
            || matches!(ty.kind(), TyKind::Array { .. })
    }

    /// Lower a ground type to MLIR: records resolve through the layout table, a
    /// nullable pointer wraps its (pointer-typed) element, and everything else is
    /// a scalar.
    pub(super) fn mlir_ty(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        match *ty.kind() {
            TyKind::Record { .. } => self.record_type(ty),
            // A `Nullable<T>` wraps its element's MLIR pointer type (an `rc` link
            // for the regional/shared records that can sit behind a nullable
            // `[field]` slot) so it may additionally hold null. The dialect's
            // `nullable` needs a non-null *pointer* element; the frontend's
            // pointer-like check is conservative (it passes a `[value]` record),
            // so reject a concrete non-pointer element here rather than build a
            // malformed `nullable<record<…>>` that later passes fall over.
            TyKind::Nullable(inner) => {
                if !(self.is_shared_record(inner) || self.is_regional_record(inner)) {
                    return err("`Nullable` element does not lower to a pointer: only a \
                         managed `[shared]`/`[regional]` record can sit behind a nullable");
                }
                Ok(nullable(self.mlir_ty(inner)?))
            }
            // A closure value is a shared `rc<closure<params -> ret>>`; captures
            // have already been supplied (see [`closure_type`](Self::closure_type)).
            TyKind::Closure { params, ret } => self.closure_type(params, ret),
            // A statically shaped array is one shared `rc` box holding the whole
            // payload (`!reussir.rc<!reussir.array<dims x elem>>`); see #344.
            TyKind::Array { .. } => Ok(self.rc_type(self.array_inner_of(ty)?)),
            _ => scalar_ty(self.context, ty),
        }
    }

    /// Lower a type as it appears *as a record member*. A managed member is named
    /// by its bare element type, never an `rc` pointer: the dialect boxes a member
    /// whose element capability is `shared`/`regional` to a pointer on its own (a
    /// `!reussir.rc<…>` member is rejected — "use capability instead"). This holds
    /// for a `shared`/`regional` record member (its inner record type), for a
    /// closure member (its bare `closure` type — a closure is a shared `rc` value
    /// just like a shared record), and for an array member (its bare `array`
    /// payload type — an array is one shared `rc` box too). A scalar or `[value]`
    /// record member is the same as [`mlir_ty`](Self::mlir_ty).
    fn member_ty(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        match *ty.kind() {
            TyKind::Record { .. } => self.record_inner_of(ty),
            TyKind::Closure { params, ret } => self.closure_inner(params, ret),
            TyKind::Array { .. } => self.array_inner_of(ty),
            _ => self.mlir_ty(ty),
        }
    }

    /// Lower a mutable `[field]` link member to its stored type. The member is
    /// typed `Nullable<Record>` in the MIR, but the record stores the bare
    /// element record type (under the `field` flag the dialect derives the
    /// `nullable<rc<…>>` slot itself), so this strips the nullable wrapper and
    /// names the link's element record.
    fn field_member_ty(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let element = field_link_element(ty);
        self.record_inner_of(element)
    }

    /// The payload compound type for variant `idx` of an enum-typed `ty` — the
    /// inline `{enum}::{case}` record holding that case's fields, as used by the
    /// `reussir.record.compound` / `reussir.record.variant` construction pair.
    pub(super) fn variant_payload_of(&self, ty: Ty<'tcx>, idx: usize) -> Result<Type<'c>> {
        let rec = self
            .record_of(ty)
            .ok_or_else(|| LoweringError("record instance has no resolved layout".into()))?;
        match rec.layout {
            mir::RecordLayout::Variant(variants) => {
                let v = variants
                    .get(idx)
                    .ok_or_else(|| LoweringError("variant index out of range".into()))?;
                self.variant_payload_type(v)
            }
            mir::RecordLayout::Compound(_) => err("variant payload requested for a struct record"),
        }
    }

    /// The MLIR type of a value of record type `ty`: the inline record type for a
    /// `[value]` record, or an `rc` pointer to it for a `[shared]` record.
    pub(super) fn record_type(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let rec = self
            .record_of(ty)
            .ok_or_else(|| LoweringError("record instance has no resolved layout".into()))?;
        let inner = self.record_inner_type(rec)?;
        match rec.default_cap {
            DefaultCap::Value => Ok(inner),
            DefaultCap::Shared => Ok(self.rc_type(inner)),
            // A regional record is also an `rc` box, but its capability is read
            // from the record's flexivity coloring (the type carries it): a `flex`
            // box is region-local and mutable, a `rigid` box is frozen (immutable,
            // free to escape its region) — still a regional `rc`, not a shared one.
            DefaultCap::Regional => Ok(rc(
                inner,
                regional_capability(ty)?,
                ReussirAtomicKind::Normal,
            )),
        }
    }

    /// The identified `!reussir.record<…>` payload type for a record-typed `ty`
    /// (the inline layout, before any `rc` wrapping). For a `[shared]` record this
    /// is the type stored inside its `rc` box, as produced by
    /// `reussir.record.compound` before `reussir.rc.create`.
    pub(super) fn record_inner_of(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let rec = self
            .record_of(ty)
            .ok_or_else(|| LoweringError("record instance has no resolved layout".into()))?;
        self.record_inner_type(rec)
    }

    /// Build the identified `!reussir.record<…>` payload type for a record
    /// instance (the inline layout, before any `rc` wrapping). Shared record-typed
    /// members lower to `rc` links through [`mlir_ty`](Self::mlir_ty).
    ///
    /// The identified type is obtained by its unique v0 symbol — which is how the
    /// MLIR context uniques it, so this returns the same handle however many times
    /// it is called. The body is filled in only when the record is not already
    /// complete and not already on the construction stack, so each record is
    /// completed exactly once and a self-referential record's back-edge resolves
    /// to the still-incomplete handle instead of recursing forever.
    fn record_inner_type(&self, rec: &mir::RecordInstance<'tcx>) -> Result<Type<'c>> {
        let kind = match rec.layout {
            mir::RecordLayout::Compound(_) => ReussirRecordKind::Compound,
            mir::RecordLayout::Variant(_) => ReussirRecordKind::Variant,
        };
        let name = StringAttribute::new(self.context, self.program.symbol(rec.symbol));
        let record = record_incomplete(self.context, name, kind);
        // Already built, or currently being built further up the stack (a
        // recursive reference): hand back the identified handle as-is.
        if record_is_complete(record) || !self.building.borrow_mut().insert(rec.symbol) {
            return Ok(record);
        }
        let result = self.complete_record(rec, record);
        self.building.borrow_mut().remove(&rec.symbol);
        result
    }

    /// Fill in `record`'s body from `rec`'s layout. The caller has marked
    /// `rec.symbol` as on the construction stack, so the recursion through
    /// [`mlir_ty`](Self::mlir_ty) below terminates at any back-edge.
    fn complete_record(
        &self,
        rec: &mir::RecordInstance<'tcx>,
        record: Type<'c>,
    ) -> Result<Type<'c>> {
        let (member_tys, member_is_field) = match rec.layout {
            mir::RecordLayout::Compound(members) => {
                let mut tys = Vec::with_capacity(members.len());
                let mut is_field = Vec::with_capacity(members.len());
                for m in members {
                    // A `[field]` link member stores the bare element record type;
                    // the dialect derives its `nullable<rc<…>>` slot from the flag.
                    let member_ty = if m.is_field {
                        self.field_member_ty(m.ty)?
                    } else {
                        self.member_ty(m.ty)?
                    };
                    tys.push(member_ty);
                    is_field.push(m.is_field);
                }
                (tys, is_field)
            }
            // An enum's members are its per-case payload compounds; the tag and
            // payload area the variant adds around them are the dialect's concern.
            mir::RecordLayout::Variant(variants) => {
                let mut tys = Vec::with_capacity(variants.len());
                for v in variants {
                    tys.push(self.variant_payload_type(v)?);
                }
                let is_field = vec![false; variants.len()];
                (tys, is_field)
            }
        };
        // `#[repr(fixed)]` pins uniform max-arm box sizing; only a *managed*
        // (shared/regional) variant carries it — the dialect verifier rejects
        // `fixed` on a compound or a [value] variant, and the elaborator
        // already diagnosed misuse (this also guards MIR ingested as text,
        // which bypasses that diagnostic).
        let fixed = matches!(rec.layout, mir::RecordLayout::Variant(_))
            && rec.default_cap != DefaultCap::Value
            && rec.repr_fixed;
        record_complete_in_place(
            record,
            &member_tys,
            &member_is_field,
            capability(rec.default_cap),
            fixed,
        );
        Ok(record)
    }

    /// The identified payload compound holding variant `v`'s fields — the payload
    /// member stored at `v`'s tag in the enum's variant record. It is named by the
    /// variant's mangled payload symbol (`Enum<Params>::Variant`, baked into the
    /// MIR by mono — see [`mir::VariantDef`]).
    ///
    /// The payload is always a `[value]` compound: it is the inline storage for a
    /// case, sitting in the variant's payload area, so `reussir.record.variant`
    /// takes it by value (the enum's own `shared` capability boxes the whole
    /// variant, not each case). Built the first time it is asked for and cached by
    /// the MLIR context under its unique name; a field that refers back to the enum
    /// resolves to the still-incomplete enum handle, because the enum's symbol is
    /// on the construction stack while its payloads are built.
    fn variant_payload_type(&self, v: &mir::VariantDef<'tcx>) -> Result<Type<'c>> {
        let name = self.program.symbol(v.symbol);
        let payload = record_incomplete(
            self.context,
            StringAttribute::new(self.context, name),
            ReussirRecordKind::Compound,
        );
        if record_is_complete(payload) {
            return Ok(payload);
        }
        let mut field_tys = Vec::with_capacity(v.fields.len());
        for &f in v.fields {
            field_tys.push(self.member_ty(f)?);
        }
        let is_field = vec![false; v.fields.len()];
        // A payload is a `[value]` compound, never a variant, so it never pins
        // `fixed` box sizing.
        record_complete_in_place(
            payload,
            &field_tys,
            &is_field,
            ReussirCapability::Value,
            /* fixed = */ false,
        );
        Ok(payload)
    }

    /// `!reussir.rc<closure<(inputs) -> ret>, shared>` — the pointer type for a
    /// closure value over `inputs`. A closure is always a shared `rc` box; the
    /// runtime has no separate capture environment, so `inputs` is the closure's
    /// full argument list (captures followed by parameters at creation, or just
    /// the remaining parameters once captures have been applied). A unit return is
    /// a closure with no output type.
    pub(super) fn closure_type(&self, inputs: &[Ty<'tcx>], ret: Ty<'tcx>) -> Result<Type<'c>> {
        Ok(self.rc_type(self.closure_inner(inputs, ret)?))
    }

    /// `!reussir.closure<(inputs) -> ret>` — the *bare* closure type, without the
    /// `rc` box. A closure is a shared `rc` value, so as a standalone value it is
    /// wrapped by [`closure_type`](Self::closure_type); as a record member it
    /// appears bare (a shared capability member the dialect boxes), the same way
    /// a `shared` record member is named by its inner type — see
    /// [`member_ty`](Self::member_ty).
    fn closure_inner(&self, inputs: &[Ty<'tcx>], ret: Ty<'tcx>) -> Result<Type<'c>> {
        let input_tys = inputs
            .iter()
            .map(|&t| self.mlir_ty(t))
            .collect::<Result<Vec<_>>>()?;
        let output = if is_unit(ret) {
            None
        } else {
            Some(self.mlir_ty(ret)?)
        };
        Ok(closure(self.context, &input_tys, output))
    }

    /// The `!reussir.array<dims x elem>` payload type of an array-typed `ty`
    /// (the type inside its `rc` box, and the element type of its views).
    pub(super) fn array_inner_of(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let TyKind::Array { elem, dims } = *ty.kind() else {
            return err("array payload requested for a non-array type");
        };
        let elem = scalar_ty(self.context, elem)?;
        let shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        Ok(array(&shape, elem))
    }

    /// `!reussir.rc<inner, shared>` — the pointer type for a heap-allocated,
    /// reference-counted value with the default (non-atomic) box layout.
    pub(super) fn rc_type(&self, inner: Type<'c>) -> Type<'c> {
        rc(inner, ReussirCapability::Shared, ReussirAtomicKind::Normal)
    }

    /// `!reussir.rc<inner, cap>` with the default (non-atomic) box layout — used
    /// to build the `flex`/`rigid` box types of a regional record.
    pub(super) fn rc_type_with_cap(&self, inner: Type<'c>, cap: ReussirCapability) -> Type<'c> {
        rc(inner, cap, ReussirAtomicKind::Normal)
    }

    /// `!reussir.region` — the arena-allocator handle a `regional` function or a
    /// `region-run` body threads to its regional allocations.
    pub(super) fn region_type(&self) -> Type<'c> {
        region(self.context)
    }

    /// `!reussir.nullable<pointer>` — a nullable wrapper around a pointer type
    /// (the `rc` link stored in a mutable `[field]` slot), which may hold null.
    pub(super) fn nullable_type(&self, pointer: Type<'c>) -> Type<'c> {
        nullable(pointer)
    }

    /// `!reussir.ref<inner, cap>` — a borrowed reference at an explicit
    /// capability, as produced by `reussir.rc.borrow` and `reussir.ref.project`:
    /// `shared` into a shared value, or `flex`/`rigid` into a regional one
    /// (region-local and mutable, or frozen).
    pub(super) fn ref_type_with_cap(&self, inner: Type<'c>, cap: ReussirCapability) -> Type<'c> {
        r#ref(inner, cap, ReussirAtomicKind::Normal)
    }

    /// `!reussir.ref<inner>` with unspecified capability — the form a spilled
    /// stack reference takes, used to acquire/drop an inline value in place.
    pub(super) fn unspecified_ref_type(&self, inner: Type<'c>) -> Type<'c> {
        r#ref(
            inner,
            ReussirCapability::Unspecified,
            ReussirAtomicKind::Normal,
        )
    }
}

/// The `rc` capability a regional record `ty` carries, read from its flexivity
/// coloring.
///
/// A regional `rc` is colored either `Flex` — region-local and mutable — or
/// `Rigid` — frozen: immutable and free to escape its region, but still a
/// *regional* `rc` (not a shared one). Those are the only two colors a regional
/// record may carry in expression position: full elaboration colors every
/// regional `rc`, so an unrefined `Regional` color, or an `Irrelevant`/absent
/// one, reaching codegen means the value was left uncolored. Rather than guess a
/// color, that is rejected as an elaboration invariant violation.
pub(super) fn regional_capability(ty: Ty<'_>) -> Result<ReussirCapability> {
    match ty.flexivity() {
        Some(Flexivity::Flex) => Ok(ReussirCapability::Flex),
        Some(Flexivity::Rigid) => Ok(ReussirCapability::Rigid),
        other => err(format!(
            "regional record reached codegen with an unrefined flexivity coloring \
             ({other:?}); elaboration must resolve it to flex or rigid"
        )),
    }
}

/// The element record a mutable `[field]` link points at. The link is typed
/// `Nullable<Record>` in the MIR; this strips the nullable wrapper to name the
/// pointee record (tolerating an already-bare record type).
pub(super) fn field_link_element(ty: Ty<'_>) -> Ty<'_> {
    match *ty.kind() {
        TyKind::Nullable(inner) => inner,
        _ => ty,
    }
}

/// Map a record's declared management to the MLIR record type's default member
/// capability.
fn capability(cap: DefaultCap) -> ReussirCapability {
    match cap {
        DefaultCap::Value => ReussirCapability::Value,
        DefaultCap::Shared => ReussirCapability::Shared,
        DefaultCap::Regional => ReussirCapability::Regional,
    }
}

/// Whether `ty` is the unit type (which carries no MLIR SSA value).
pub(super) fn is_unit(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Unit)
}

/// Whether `ty` is a closure type — a shared `rc<closure<…>>` value, which is
/// reference-counted like a `[shared]` record.
pub(super) fn is_closure(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Closure { .. })
}

/// The MLIR type for a ground scalar Reussir type. Records are handled by
/// [`TypeCtx::record_type`], which intercepts them before this fallback; reaching
/// the `Record` arm here is a bug.
fn scalar_ty<'c>(context: &'c Context, ty: Ty<'_>) -> Result<Type<'c>> {
    match *ty.kind() {
        TyKind::Int(IntTy::Signed(w)) | TyKind::Int(IntTy::Unsigned(w)) => {
            Ok(IntegerType::new(context, u32::from(w)).into())
        }
        TyKind::Fp(FpTy::Ieee(16)) => Ok(Type::float16(context)),
        TyKind::Fp(FpTy::Ieee(32)) => Ok(Type::float32(context)),
        TyKind::Fp(FpTy::Ieee(64)) => Ok(Type::float64(context)),
        TyKind::Fp(FpTy::Ieee(_)) => err("unsupported IEEE float width"),
        TyKind::Fp(FpTy::BFloat16) => Ok(Type::bfloat16(context)),
        TyKind::Fp(FpTy::Float8) => err("float8 has no standard MLIR builtin type"),
        TyKind::Bool => Ok(IntegerType::new(context, 1).into()),
        TyKind::Char => Ok(IntegerType::new(context, 32).into()),
        TyKind::Unit => err("unit has no MLIR value type"),
        TyKind::Str => Ok(str_type(context, ReussirLifeScope::Global)),
        TyKind::Record { .. } => err("record type reached scalar lowering without a layout"),
        TyKind::Array { .. } => err("array type reached scalar lowering without its rc wrapper"),
        TyKind::Nullable(_) => err("nullable type lowering not yet implemented"),
        // Intercepted by [`TypeCtx::mlir_ty`]; reaching here is a bug.
        TyKind::Closure { .. } => err("closure type reached scalar lowering without an rc wrapper"),
        TyKind::Bottom => err("bottom type reached lowering"),
        TyKind::Generic(_) | TyKind::Hole(_) => err("non-ground type reached lowering"),
    }
}

/// The numeric classification of a scalar type, used to pick the right cast op.
pub(super) struct NumClass {
    /// Bit width of the scalar.
    pub(super) width: u16,
    /// Whether to treat it as signed — selects `extsi`/`sitofp` over the
    /// unsigned variants. Floats are signed by convention.
    pub(super) signed: bool,
    /// Whether it is a floating-point type (vs. an integer or `bool`).
    pub(super) float: bool,
}

/// Classify a numeric scalar for cast selection, or error if it isn't one.
pub(super) fn num_class(ty: Ty<'_>) -> Result<NumClass> {
    let (width, signed, float) = match *ty.kind() {
        TyKind::Int(IntTy::Signed(w)) => (w, true, false),
        TyKind::Int(IntTy::Unsigned(w)) => (w, false, false),
        TyKind::Bool => (1, false, false),
        TyKind::Fp(FpTy::Ieee(w)) => (w, true, true),
        TyKind::Fp(FpTy::BFloat16) => (16, true, true),
        _ => return err("cast operand is not a numeric scalar"),
    };
    Ok(NumClass {
        width,
        signed,
        float,
    })
}
