//! Type lowering: ground Reussir types → MLIR types.
//!
//! Scalars map directly to builtin MLIR types. A record has no `DefTable` here,
//! so it resolves through a layout table ([`TypeCtx`]) keyed by its `(def, args)`
//! identity — the ground layout `mono` baked into the MIR. `[value]` records
//! build an identified `!reussir.record<…>`; shared/regional records and enums
//! arrive with the rc lowering. This module also holds the numeric classification
//! [`expr`](super::expr) uses to choose the right cast op.

use rustc_hash::FxHashMap;

use reussir_backend::dialect::ty::{ReussirCapability, ReussirRecordKind, record_complete};
use reussir_backend::melior::Context;
use reussir_backend::melior::ir::Type;
use reussir_backend::melior::ir::attribute::StringAttribute;
use reussir_backend::melior::ir::r#type::IntegerType;

use reussir_core::full::mir;
use reussir_core::semi::ctxt::DefaultCap;
use reussir_core::semi::ty::{DefId, FpTy, IntTy, Ty, TyKind};

use super::{LoweringError, Result, err};

/// The identity of a record instance — its definition and ground type arguments
/// (capability-canonicalized) — used to match a record-typed value to its layout.
type RecordInstanceKey<'tcx> = (DefId, &'tcx [Ty<'tcx>]);

/// Type-lowering state: the MLIR context to build types in, the program (whose
/// symbol table names record types), and the record instances indexed by their
/// `(def, args)` identity so a record-typed value can find its ground layout.
pub(super) struct TypeCtx<'c, 'p, 'tcx> {
    context: &'c Context,
    program: &'p mir::Program<'tcx>,
    records: FxHashMap<RecordInstanceKey<'tcx>, &'p mir::RecordInstance<'tcx>>,
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
        }
    }

    /// The record instance for a nominal record type, if known.
    pub(super) fn record_of(&self, ty: Ty<'tcx>) -> Option<&'p mir::RecordInstance<'tcx>> {
        match *ty.kind() {
            TyKind::Record { def, args, .. } => self.records.get(&(def, args)).copied(),
            _ => None,
        }
    }

    /// Lower a ground type to MLIR: records resolve through the layout table,
    /// everything else is a scalar.
    pub(super) fn mlir_ty(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        match *ty.kind() {
            TyKind::Record { .. } => self.record_type(ty),
            _ => scalar_ty(self.context, ty),
        }
    }

    /// Build the MLIR record type for a record instance. Only `[value]` records
    /// lower today; shared/regional records arrive with the rc lowering.
    pub(super) fn record_type(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let rec = self
            .record_of(ty)
            .ok_or_else(|| LoweringError("record instance has no resolved layout".into()))?;
        match rec.default_cap {
            DefaultCap::Value => {}
            DefaultCap::Shared => return err("shared (rc) record lowering not yet implemented"),
            DefaultCap::Regional => return err("regional record lowering not yet implemented"),
        }
        let members = match rec.layout {
            mir::RecordLayout::Compound(ms) => ms,
            mir::RecordLayout::Variant(_) => return err("a `[value]` record cannot be an enum"),
        };
        let mut member_tys = Vec::with_capacity(members.len());
        let mut member_is_field = Vec::with_capacity(members.len());
        for m in members {
            member_tys.push(self.mlir_ty(m.ty)?);
            member_is_field.push(m.is_field);
        }
        // Identify the record by its unique v0 symbol so distinct generic
        // instances stay distinct MLIR types.
        let name = StringAttribute::new(self.context, self.program.symbol(rec.symbol));
        Ok(record_complete(
            self.context,
            &member_tys,
            &member_is_field,
            Some(name),
            ReussirRecordKind::Compound,
            ReussirCapability::Value,
        ))
    }
}

/// Whether `ty` is the unit type (which carries no MLIR SSA value).
pub(super) fn is_unit(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Unit)
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
        TyKind::Unit => err("unit has no MLIR value type"),
        TyKind::Str => err("string type lowering not yet implemented"),
        TyKind::Record { .. } => err("record type reached scalar lowering without a layout"),
        TyKind::Nullable(_) => err("nullable type lowering not yet implemented"),
        TyKind::Closure { .. } => err("closure type lowering not yet implemented"),
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
