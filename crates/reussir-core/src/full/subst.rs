//! Type substitution: ground a Semi [`Ty`] at a generic→type assignment.
//!
//! Monomorphization specializes a polymorphic function by mapping each generic
//! parameter ([`GenericId`]) to a concrete type. Because Full types *are* Semi
//! [`Ty`]s (just ground), substitution re-interns through the same [`TyCtxt`], so
//! the result keeps pointer-identity equality and feeds straight into mangling.
//!
//! Only *types* are substituted here; the expression tree is not rewritten in
//! place — [`crate::full::mono`] *lowers* it into the [`crate::full::mir`],
//! grounding each node's type with [`subst_ty`] as it goes.

use rustc_hash::FxHashMap;

use crate::semi::ty::{GenericId, Ty, TyCtxt, TyKind};

/// A ground assignment for a function's generic parameters.
pub type Subst<'tcx> = FxHashMap<GenericId, Ty<'tcx>>;

/// Substitute generics in a type, re-interning into `tcx`. An empty `subst`
/// short-circuits (the type is already ground).
pub fn subst_ty<'tcx>(tcx: &TyCtxt<'tcx>, ty: Ty<'tcx>, subst: &Subst<'tcx>) -> Ty<'tcx> {
    if subst.is_empty() {
        return ty;
    }
    match *ty.kind() {
        TyKind::Generic(id) => *subst
            .get(&id)
            .unwrap_or_else(|| panic!("subst: generic {id:?} is not bound by this instance")),
        TyKind::Record { def, args, flex } => {
            let args: Vec<Ty<'tcx>> = args.iter().map(|&a| subst_ty(tcx, a, subst)).collect();
            tcx.mk_record(def, &args, flex)
        }
        TyKind::Closure { params, ret } => {
            let params: Vec<Ty<'tcx>> = params.iter().map(|&p| subst_ty(tcx, p, subst)).collect();
            tcx.mk_closure(&params, subst_ty(tcx, ret, subst))
        }
        TyKind::Nullable(inner) => tcx.mk_nullable(subst_ty(tcx, inner, subst)),
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Unit
        | TyKind::Bottom => ty,
        TyKind::Hole(_) => panic!("subst: an inference hole survived zonking"),
    }
}
