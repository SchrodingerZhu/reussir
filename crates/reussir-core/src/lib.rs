//! Core language representation for Reussir.
//!
//! The compiler is organized by phase. [`semi`] owns the *Semi* representation:
//! the interned type IR ([`semi::ty`]), its unification solver ([`semi::infer`]),
//! the trait system ([`semi::traits`]), and the bidirectional elaboration of the
//! surface AST into a typed, still-polymorphic HIR. [`surface`] is the typed,
//! parser-facing syntax tree. The monomorphized, mangled *Full* representation
//! (`full::*`) is a separate, future phase.

pub mod semi;
pub mod surface;
pub mod utils;

/// Run `f` inside a fresh type-arena scope, handing it an interning
/// [`semi::ty::TyCtxt`].
///
/// The arena's `with_scope` brands every handle with a generative lifetime, so
/// no `Ty` can escape `f` — exactly the property that makes pointer interning
/// sound. Real entry points (the elaborator) wrap their work the same way.
#[cfg(test)]
pub(crate) fn with_tcx<R>(f: impl for<'tcx> FnOnce(&semi::ty::TyCtxt<'tcx>) -> R) -> R {
    let mut arena = stumpalo::Arena::new();
    arena.with_scope(|arena_ref| f(&semi::ty::TyCtxt::new(arena_ref)))
}
