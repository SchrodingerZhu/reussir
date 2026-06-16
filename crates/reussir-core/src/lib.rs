//! Core language representation for Reussir.
//!
//! This crate is being filled in incrementally. The type system lives in three
//! layers: [`ty`] is the interned type IR (a [`ty::Ty`] is a pointer-comparable
//! handle), [`infer`] is the unification-variable solver, and [`traits`] is the
//! nominal, coherent trait system (built-ins included).

pub mod infer;
pub mod traits;
pub mod ty;
pub mod utils;

/// Run `f` inside a fresh type-arena scope, handing it an interning [`ty::TyCtxt`].
///
/// The arena's `with_scope` brands every handle with a generative lifetime, so
/// no `Ty` can escape `f` — exactly the property that makes pointer interning
/// sound. Real entry points (the elaborator) will wrap their work the same way.
#[cfg(test)]
pub(crate) fn with_tcx<R>(f: impl for<'tcx> FnOnce(&ty::TyCtxt<'tcx>) -> R) -> R {
    let mut arena = stumpalo::Arena::new();
    arena.with_scope(|arena_ref| f(&ty::TyCtxt::new(arena_ref)))
}
