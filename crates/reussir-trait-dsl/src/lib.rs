//! `builtin_traits!` — the declarative registration DSL for Reussir's
//! built-in ("lang item") traits and their primitive impls.
//!
//! The macro replaces the hand-written registration table in
//! `reussir-core::semi::traits::builtins` with a declaration that reads like
//! the surface language:
//!
//! ```ignore
//! builtin_traits! {
//!     #[sealed] trait Num;
//!     #[sealed] trait Integral: Num;
//!     #[sealed] trait FloatingPoint: Num;
//!     #[sealed] trait PtrLike;
//!     /// Marker (auto) trait: reachable from any thread, concurrently.
//!     #[sealed] #[structural] trait Sync;
//!     impl Integral for i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64;
//!     impl FloatingPoint for f16 | f32 | f64 | bf16 | f8;
//! }
//! ```
//!
//! Expansion produces, at the invocation site (inside `reussir-core`, whose
//! `crate::semi::…` paths the generated code references):
//!
//! * `pub struct Builtins` with one snake-cased `TraitId` field per declared
//!   trait, in declaration order (doc comments forwarded);
//! * `impl Builtins { pub fn register(db, defs, interner, tcx) -> Builtins }`
//!   declaring each trait in the `DefTable`'s trait namespace at the crate
//!   root and registering `TraitDef`s/`ImplDef`s with dense, declaration-
//!   ordered ids — the exact contract of the hand-written table it replaces.
//!
//! Declaration order **is** the id assignment: reordering the invocation
//! renumbers `TraitId`/`ImplId`. `#[sealed]` marks a trait unimplementable
//! by user code; `#[structural]` additionally forbids impls *in this table*
//! (the checker answers such traits structurally — `Sync`). Method
//! signatures inside a `trait { … }` body parse (the grammar is reserved for
//! the method-carrying future) but are rejected in expansion for now.

mod ast;
mod expand;

/// See the crate docs.
#[proc_macro]
pub fn builtin_traits(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
