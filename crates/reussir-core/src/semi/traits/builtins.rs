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

use reussir_syntax::Interner;
use reussir_syntax::kind::TokenKey;

use crate::semi::resolve::DefTable;
use crate::semi::ty::{FpTy, GenericId, IntTy, Ty, TyCtxt};
use crate::surface;

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
    /// Register the built-in traits and their primitive impls into `db`,
    /// declaring each trait in `defs`' trait namespace at the crate root —
    /// so a bare `Num` resolves from any module through the root fallback
    /// (prelude visibility), while a same-named user trait in a module
    /// shadows it for bare references.
    pub fn register<'tcx>(
        db: &mut TraitDb<'tcx>,
        defs: &mut DefTable,
        interner: &mut impl Interner<TokenKey>,
        tcx: &TyCtxt<'tcx>,
    ) -> Builtins {
        // Every built-in trait has a single `Self` parameter, generic #0 —
        // deliberately UNALLOCATED in the elaborator's generics table (the
        // builtin defs predate it, and allocating would shift every dump's
        // `$n` numbering).
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

        let mut declare =
            |id, name: &str, supertraits, db: &mut TraitDb<'tcx>, defs: &mut DefTable| {
                let key = interner.get_or_intern(name);
                let def = defs.declare_trait(key).expect("builtins precede user defs");
                db.add_trait(TraitDef {
                    id,
                    def,
                    visibility: surface::Visibility::Public,
                    sealed: true,
                    self_param: self_g,
                    params: vec![],
                    supertraits,
                    methods: vec![],
                    assoc_tys: vec![],
                    span: None,
                    file: reussir_syntax::source::FileId::ROOT,
                });
            };
        declare(num, "Num", vec![], db, defs);
        declare(integral, "Integral", vec![self_ref(num)], db, defs);
        declare(
            floating_point,
            "FloatingPoint",
            vec![self_ref(num)],
            db,
            defs,
        );
        declare(ptr_like, "PtrLike", vec![], db, defs);
        declare(sync, "Sync", vec![], db, defs);

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
