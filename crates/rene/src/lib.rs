//! `rene`: the Reussir package manager.
//!
//! A package is a directory with a `rene.ncl` manifest — a Nickel program
//! evaluating to the package description (name, dependencies, …). This first
//! stage only locates and evaluates the manifest; driving `rrc`, dependency
//! resolution, and project scaffolding come later.

pub mod manifest;
