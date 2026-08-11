//! `rene`: the Reussir package manager.
//!
//! A package is a directory with a `rene.ncl` manifest — a Nickel program
//! evaluating to the package description (name, dependencies, …) — and
//! Reussir sources. Targets may name independent crate-root files, falling
//! back to `src/lib.rr` when no path is given. Builds run in a build directory
//! (`reussir-build/` by default) whose status lives in a redb
//! database: the baked bundled `reussir-rt` runtime ([`rt`]), the package's
//! source graph as `rrc --scan-deps` reported it ([`deps`]), and the built
//! products of the manifest's declared targets ([`compile`]). Dependencies
//! resolve over local paths ([`resolve`]), and `rene new` scaffolds a fresh
//! package ([`new`]).

pub mod compile;
pub mod core_src;
pub mod db;
pub mod deps;
pub mod exec;
pub mod fresh;
pub mod manifest;
pub mod new;
pub mod plan;
pub mod pool;
pub mod resolve;
pub mod rt;
pub mod tables;
