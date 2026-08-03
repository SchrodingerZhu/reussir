//! The built-in ("lang item") traits and impls.
//!
//! These replace the formerly hard-coded primitive class hierarchy. Membership
//! now lives in ordinary [`TraitDef`]/[`ImplDef`] data resolved by the same
//! engine as any user trait — there is no special-cased DAG, and the table
//! below is a *declaration*: [`reussir_trait_dsl::builtin_traits!`] expands it
//! into the `Builtins` struct and its `register` function, assigning dense
//! `TraitId`/`ImplId`s in declaration order. The hierarchy is:
//!
//! ```text
//! Num            PtrLike      Sync
//! ├── Integral   (impl: i8..=u64)
//! └── FloatingPoint (impl: f16..f8)
//! ```
//!
//! [`TraitDef`]: super::TraitDef
//! [`ImplDef`]: super::ImplDef

use reussir_trait_dsl::builtin_traits;

builtin_traits! {
    /// The root numeric class: arithmetic and numeric literals.
    #[sealed]
    trait Num;
    /// Whole-number types; `Num` through the super-trait edge.
    #[sealed]
    trait Integral: Num;
    /// Floating-point types; `Num` through the super-trait edge.
    #[sealed]
    trait FloatingPoint: Num;
    /// Pointer-shaped values (nullable-niche eligible).
    #[sealed]
    trait PtrLike;
    /// Marker (auto) trait: a value may be reached from any thread,
    /// concurrently. It subsumes both of Rust's `Send` and `Sync` —
    /// duplicable rc handles make transfer-without-sharing unprovable, so
    /// the two predicates collapse (see `docs/design/thread-safety.md`).
    /// Declared for name resolution and bound syntax only: ground `Sync`
    /// goals are answered *structurally* by [`crate::semi::traits::sync`],
    /// never by impl search, so it has (and can have) no impls.
    #[sealed]
    #[structural]
    trait Sync;

    impl Integral for i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64;
    impl FloatingPoint for f16 | f32 | f64 | bf16 | f8;
}

#[cfg(test)]
mod tests {
    use super::Builtins;
    use crate::semi::traits::{TraitDb, TraitId};
    use crate::with_tcx;

    /// The generated table is the old hand-written one, exactly: dense
    /// declaration-ordered trait ids, 8 integral + 5 floating impls, all
    /// sealed, `Sync` impl-less.
    #[test]
    fn generated_table_matches_the_handwritten_contract() {
        with_tcx(|tcx| {
            let mut db = TraitDb::new();
            let mut defs = crate::semi::resolve::DefTable::new();
            let mut interner = lasso::Rodeo::<reussir_syntax::kind::TokenKey>::new();
            let b = Builtins::register(&mut db, &mut defs, &mut interner, tcx);
            assert_eq!(
                [b.num, b.integral, b.floating_point, b.ptr_like, b.sync],
                [TraitId(0), TraitId(1), TraitId(2), TraitId(3), TraitId(4)]
            );
            assert_eq!(db.traits_len(), 5);
            assert_eq!(db.impls_len(), 13, "8 integral + 5 floating impls");
            for i in 0..5u32 {
                assert!(db.trait_def(TraitId(i)).sealed);
            }
            assert!(
                db.impls().all(|imp| imp.trait_ref.trait_id != b.sync),
                "Sync is structural; it must have no impls"
            );
            let supers = |t: TraitId| -> Vec<TraitId> {
                db.trait_def(t)
                    .supertraits
                    .iter()
                    .map(|s| s.trait_id)
                    .collect()
            };
            assert_eq!(supers(b.integral), [b.num]);
            assert_eq!(supers(b.floating_point), [b.num]);
            assert!(supers(b.num).is_empty() && supers(b.sync).is_empty());
        });
    }
}
