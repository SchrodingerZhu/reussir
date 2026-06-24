//! Type lowering: ground scalar Reussir types → MLIR types, plus the numeric
//! classification [`expr`](super::expr) uses to choose the right cast op.

use reussir_backend::melior::Context;
use reussir_backend::melior::ir::Type;
use reussir_backend::melior::ir::r#type::IntegerType;

use reussir_core::semi::ty::{FpTy, IntTy, Ty, TyKind};

use super::{Result, err};

/// Whether `ty` is the unit type (which carries no MLIR SSA value).
pub(super) fn is_unit(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Unit)
}

/// The MLIR type for a ground scalar Reussir type.
pub(super) fn mlir_ty<'c>(context: &'c Context, ty: Ty<'_>) -> Result<Type<'c>> {
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
        TyKind::Record { .. } => err("record type lowering not yet implemented"),
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
