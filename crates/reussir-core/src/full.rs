//! The *Full* phase: the monomorphized, mangled representation.
//!
//! Where [`crate::semi`] keeps generics open and may carry inference holes, the
//! Full phase is **ground**: every type argument is a concrete [`semi::ty::Ty`]
//! with no [`semi::ty::TyKind::Generic`] or [`semi::ty::TyKind::Hole`] left. That
//! invariant is what lets each item instance have a single, stable linker symbol
//! (see [`mangle`]) and what the backend lowering relies on.
//!
//! Full reuses the Semi type interner ([`semi::ty::TyCtxt`]) rather than
//! introducing a parallel type representation: a Full type *is* a Semi `Ty` that
//! happens to be ground, so monomorphization is a substitution that re-interns
//! into the same arena and downstream code keeps pointer-identity equality.
//!
//! [`semi`]: crate::semi

pub mod mangle;
pub mod mir;
pub mod mono;
pub mod print;
pub mod subst;

// The textual-IR parser, generated from `full/ir.lalrpop` at build time.
lalrpop_util::lalrpop_mod!(
    #[allow(clippy::all, dead_code, unused_imports)]
    pub ir,
    "/full/ir.rs"
);

#[cfg(test)]
mod ir_pipeline_tests {
    /// Smoke test: the lalrpop-generated parser is wired up and callable. The
    /// real IR productions replace this scaffolding.
    #[test]
    fn lalrpop_pipeline_is_wired() {
        assert_eq!(super::ir::NumParser::new().parse("42"), Ok(42));
    }
}
