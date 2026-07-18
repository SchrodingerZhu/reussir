//! The built-in ("lang item") traits and impls.
//!
//! These replace the formerly hard-coded primitive class hierarchy. Membership
//! now lives in ordinary [`TraitDef`]/[`ImplDef`] data resolved by the same
//! engine as any user trait — there is no special-cased DAG. The hierarchy is:
//!
//! ```text
//! Num            PtrLike      Sync
//! ├── Integral   (impl: i8..=u64)
//! └── FloatingPoint (impl: f16..f8)
//! ```
//!
//! `Sync` is the single thread-safety marker (auto) trait: a value may be
//! reached from any thread, concurrently. It subsumes both of Rust's `Send`
//! and `Sync` — duplicable rc handles make transfer-without-sharing
//! unprovable, so the two predicates collapse (see
//! `docs/design/thread-safety.md`). It is declared here for name resolution
//! and bound syntax only: ground `Sync` goals are answered *structurally* by
//! [`crate::semi::traits::sync`], never by impl search, so it has no impls.

use crate::semi::ty::{FpTy, GenericId, IntTy, Ty, TyCtxt};

use super::def::{ImplDef, TraitDef};
use super::{ImplId, TraitDb, TraitId, TraitRef};

/// Handles to the built-in traits, for the type checker to refer to by name.
pub struct Builtins {
    pub num: TraitId,
    pub integral: TraitId,
    pub floating_point: TraitId,
    pub ptr_like: TraitId,
    /// Marker (auto) trait: a value may be reached from any thread,
    /// concurrently. Subsumes both of Rust's `Send` and `Sync`
    /// (`docs/design/thread-safety.md`).
    pub sync: TraitId,
}

impl Builtins {
    /// Register the built-in traits and their primitive impls into `db`.
    pub fn register<'tcx>(db: &mut TraitDb<'tcx>, tcx: &TyCtxt<'tcx>) -> Builtins {
        // Every built-in trait has a single `Self` parameter, generic #0.
        let self_g = GenericId(0);
        let self_ref = |trait_id: TraitId| TraitRef {
            trait_id,
            args: vec![tcx.mk_generic(self_g)],
        };

        let mut next = 0u32;
        let mut fresh_trait = || {
            let id = TraitId(next);
            next += 1;
            id
        };
        let num = fresh_trait();
        let integral = fresh_trait();
        let floating_point = fresh_trait();
        let ptr_like = fresh_trait();
        let sync = fresh_trait();

        let declare = |id, name: &str, supertraits, db: &mut TraitDb<'tcx>| {
            db.add_trait(TraitDef {
                id,
                name: name.to_owned(),
                params: vec![self_g],
                supertraits,
                methods: vec![],
                assoc_tys: vec![],
            });
        };
        declare(num, "Num", vec![], db);
        declare(integral, "Integral", vec![self_ref(num)], db);
        declare(floating_point, "FloatingPoint", vec![self_ref(num)], db);
        declare(ptr_like, "PtrLike", vec![], db);
        declare(sync, "Sync", vec![], db);

        let mut next_impl = 0u32;
        let mut implement = |trait_id: TraitId, self_ty: Ty<'tcx>, db: &mut TraitDb<'tcx>| {
            let id = ImplId(next_impl);
            next_impl += 1;
            db.add_impl(ImplDef {
                id,
                generics: vec![],
                trait_ref: TraitRef {
                    trait_id,
                    args: vec![self_ty],
                },
                self_ty,
                where_clauses: vec![],
            });
        };

        let mut ints = Vec::new();
        for width in [8u16, 16, 32, 64] {
            ints.push(tcx.mk_int(IntTy::Signed(width)));
            ints.push(tcx.mk_int(IntTy::Unsigned(width)));
        }
        let mut fps = Vec::new();
        for fp in [
            FpTy::Ieee(16),
            FpTy::Ieee(32),
            FpTy::Ieee(64),
            FpTy::BFloat16,
            FpTy::Float8,
        ] {
            fps.push(tcx.mk_fp(fp));
        }

        for &ty in &ints {
            implement(integral, ty, db);
        }
        for &ty in &fps {
            implement(floating_point, ty, db);
        }
        Builtins {
            num,
            integral,
            floating_point,
            ptr_like,
            sync,
        }
    }
}
