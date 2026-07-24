//! `rene`: the Reussir package manager.
//!
//! A package is a directory with a `rene.ncl` manifest — a Nickel program
//! evaluating to the package description (name, dependencies, …). Builds run
//! in a build directory (`reussir-build/` by default) whose status lives in a
//! redb database; the first build stage bakes the bundled `reussir-rt`
//! runtime there with the user's Rust toolchain. Driving `rrc`, dependency
//! resolution, and project scaffolding come later.

pub mod db;
pub mod manifest;
pub mod rt;
pub mod tables;
