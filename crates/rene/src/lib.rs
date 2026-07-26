//! `rene`: the Reussir package manager.
//!
//! A package is a directory with a `rene.ncl` manifest — a Nickel program
//! evaluating to the package description (name, dependencies, …) — and a
//! `src/` tree whose `lib.rr` is the crate root. Builds run in a build
//! directory (`reussir-build/` by default) whose status lives in a redb
//! database: the baked bundled `reussir-rt` runtime ([`rt`]), the package's
//! source graph as `rrc --scan-deps` reported it ([`deps`]), and the built
//! products of the manifest's declared targets ([`compile`]). Dependency
//! resolution and project scaffolding come later.

pub mod compile;
pub mod db;
pub mod deps;
pub mod manifest;
pub mod resolve;
pub mod rt;
pub mod tables;
