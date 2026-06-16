//! The trait/impl registry and a trivial resolver.
//!
//! Phase 0 resolves only *ground* obligations: a concrete self type is matched
//! exactly against impls (a pointer comparison, thanks to interning), with
//! super-trait projection, plus the capability lattice. Candidate gathering from
//! in-scope assumptions, one-way matching of generic impls, where-clause
//! discharge against an environment, and suspension of obligations whose self
//! type is still a hole all arrive in Phase 1 — slotting into this same
//! [`TraitDb::select`] entry point.

use crate::ty::{Capability, Ty};

use super::def::{ImplDef, TraitDef};
use super::{Evidence, ImplId, Obligation, TraitId, TraitRef};

/// The collected trait program plus resolution.
#[derive(Default)]
pub struct TraitDb<'tcx> {
    traits: Vec<TraitDef<'tcx>>,
    impls: Vec<ImplDef<'tcx>>,
}

/// Why an obligation could not be discharged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError<'tcx> {
    /// No impl makes `τ : Trait` hold.
    NoImpl(TraitRef<'tcx>),
    /// A capability bound the self type does not meet.
    CapNotSatisfied(Ty<'tcx>, Capability),
}

impl<'tcx> TraitDb<'tcx> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_trait(&mut self, def: TraitDef<'tcx>) -> TraitId {
        let id = def.id;
        self.traits.push(def);
        id
    }

    pub fn add_impl(&mut self, def: ImplDef<'tcx>) -> ImplId {
        let id = def.id;
        self.impls.push(def);
        id
    }

    pub fn trait_def(&self, id: TraitId) -> &TraitDef<'tcx> {
        &self.traits[id.0 as usize]
    }

    /// Discharge an obligation, producing evidence.
    pub fn select(
        &self,
        obligation: &Obligation<'tcx>,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        match obligation {
            Obligation::Cap(ty, need) => self.discharge_cap(*ty, *need),
            Obligation::Trait(goal) => self.select_trait(goal),
        }
    }

    fn discharge_cap(
        &self,
        ty: Ty<'tcx>,
        need: Capability,
    ) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        match ty.capability() {
            Some(have) if have.satisfies(need) => Ok(Evidence::Cap),
            _ => Err(SelectError::CapNotSatisfied(ty, need)),
        }
    }

    fn select_trait(&self, goal: &TraitRef<'tcx>) -> Result<Evidence<'tcx>, SelectError<'tcx>> {
        // A direct impl of the goal trait for this exact self type. With
        // interned types this self-type check is a pointer comparison.
        for imp in &self.impls {
            if imp.trait_ref.trait_id == goal.trait_id && imp.self_ty == goal.self_ty() {
                let mut sub = Vec::with_capacity(imp.where_clauses.len());
                for clause in &imp.where_clauses {
                    sub.push(self.select(clause)?);
                }
                return Ok(Evidence::Impl {
                    impl_id: imp.id,
                    args: imp.trait_ref.args.clone(),
                    sub,
                });
            }
        }

        // Otherwise, reach the goal through a super-trait edge: some impl for
        // this self type implements a sub-trait whose super-trait closure
        // contains the goal. (This is what the old class DAG did by hand.)
        for imp in &self.impls {
            if imp.self_ty != goal.self_ty() {
                continue;
            }
            let sub_def = self.trait_def(imp.trait_ref.trait_id);
            if let Some(index) = self.super_index(sub_def, goal.trait_id) {
                let of = self.select_trait(&imp.trait_ref)?;
                return Ok(Evidence::Super {
                    of: Box::new(of),
                    index,
                });
            }
        }

        Err(SelectError::NoImpl(goal.clone()))
    }

    /// The position of `target` in `def`'s super-trait list, searching the
    /// transitive closure. `None` if `target` is not a super-trait of `def`.
    fn super_index(&self, def: &TraitDef<'tcx>, target: TraitId) -> Option<usize> {
        for (i, sup) in def.supertraits.iter().enumerate() {
            if sup.trait_id == target
                || self
                    .super_index(self.trait_def(sup.trait_id), target)
                    .is_some()
            {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::builtins::Builtins;
    use crate::ty::{Capability, FpTy, IntTy, TyCtxt};
    use crate::with_tcx;

    fn needs(trait_id: TraitId, self_ty: Ty<'_>) -> Obligation<'_> {
        Obligation::Trait(TraitRef {
            trait_id,
            args: vec![self_ty],
        })
    }

    #[test]
    fn primitive_membership_is_data_not_hardcoded() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let b = Builtins::register(&mut db, tcx);

            let i32 = tcx.mk_int(IntTy::Signed(32));
            let f32 = tcx.mk_fp(FpTy::Ieee(32));

            // i32 is Integral directly, and Num through the super-trait edge.
            assert!(db.select(&needs(b.integral, i32)).is_ok());
            assert!(db.select(&needs(b.num, i32)).is_ok());

            // f32 is FloatingPoint and (super) Num, but not Integral.
            assert!(db.select(&needs(b.floating_point, f32)).is_ok());
            assert!(db.select(&needs(b.num, f32)).is_ok());
            assert!(db.select(&needs(b.integral, f32)).is_err());

            // bool belongs to no numeric trait.
            assert!(db.select(&needs(b.num, tcx.mk_bool())).is_err());
        });
    }

    #[test]
    fn primitive_scalars_are_send_and_sync() {
        with_tcx(|tcx: &TyCtxt| {
            let mut db = TraitDb::new();
            let b = Builtins::register(&mut db, tcx);
            for ty in [tcx.mk_int(IntTy::Unsigned(64)), tcx.mk_bool(), tcx.mk_str()] {
                assert!(db.select(&needs(b.send, ty)).is_ok());
                assert!(db.select(&needs(b.sync, ty)).is_ok());
            }
        });
    }

    #[test]
    fn capability_bound_uses_the_lattice() {
        with_tcx(|tcx: &TyCtxt| {
            let db = TraitDb::new();
            let rigid = tcx.mk_record("R", &[], Capability::Rigid);
            let regional = tcx.mk_record("R", &[], Capability::Regional);
            // Rigid ⊒ Flex, Regional ⋣ Flex.
            assert!(db.select(&Obligation::Cap(rigid, Capability::Flex)).is_ok());
            assert!(
                db.select(&Obligation::Cap(regional, Capability::Flex))
                    .is_err()
            );
        });
    }
}
