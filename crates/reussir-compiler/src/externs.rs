//! Loading dependency package interfaces (`--extern name=path.rri`).
//!
//! An `.rri` is a textual HIR document reduced to the export closure behind
//! a versioned header (`docs/design/rri.md`). This module owns the consumer
//! side of the contract's first leg: parsing the `--extern`/`--extern-src`
//! spellings and gating the header (format, package name, reserved names)
//! before anything is offered to resolution. What the loaded tables *feed*
//! (elaboration, monomorphization) arrives in later commits.

use std::path::PathBuf;

use reussir_core::full::interface::RRI_FORMAT;
use reussir_core::semi::hir::build::{self, Parsed};
use reussir_core::semi::ty::TyCtxt;

/// One `--extern` occurrence, joined with its optional `--extern-src` root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternSpec {
    pub name: String,
    pub path: PathBuf,
    /// The directory the interface's package-root-relative file table
    /// re-anchors onto; `None` loads the files as unfetchable virtual
    /// entries (locations degrade to name-only).
    pub src_root: Option<PathBuf>,
}

/// A `NAME=VALUE` flag argument, split at the first `=`: the flag's typed
/// form (`--extern dep=out/dep.rri`, `--extern-src dep=vendor/dep`), parsed
/// by the CLI itself via [`FromStr`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternPair {
    pub name: String,
    pub value: PathBuf,
}

impl std::str::FromStr for ExternPair {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.split_once('=') {
            Some((name, value)) if !name.is_empty() && !value.is_empty() => Ok(ExternPair {
                name: name.to_owned(),
                value: PathBuf::from(value),
            }),
            _ => Err(format!("`{s}`: expected `NAME=PATH`")),
        }
    }
}

/// Join and cross-check the `--extern` / `--extern-src` occurrences:
/// distinct extern names, and every source root naming a declared extern.
pub fn join_specs(
    externs: &[ExternPair],
    extern_srcs: &[ExternPair],
) -> Result<Vec<ExternSpec>, String> {
    let mut specs: Vec<ExternSpec> = Vec::with_capacity(externs.len());
    for pair in externs {
        if specs.iter().any(|s| s.name == pair.name) {
            return Err(format!("`--extern {}=...` given twice", pair.name));
        }
        specs.push(ExternSpec {
            name: pair.name.clone(),
            path: pair.value.clone(),
            src_root: None,
        });
    }
    for pair in extern_srcs {
        let Some(spec) = specs.iter_mut().find(|s| s.name == pair.name) else {
            return Err(format!(
                "`--extern-src {}=...` names no `--extern {}=...`",
                pair.name, pair.name
            ));
        };
        if spec.src_root.is_some() {
            return Err(format!("`--extern-src {}=...` given twice", pair.name));
        }
        spec.src_root = Some(pair.value.clone());
    }
    Ok(specs)
}

/// A dependency interface loaded into the consumer's `TyCtxt`.
pub struct LoadedExtern<'tcx> {
    pub name: String,
    pub src_root: Option<PathBuf>,
    pub parsed: Parsed<'tcx>,
}

impl LoadedExtern<'_> {
    /// The interface's file table, re-anchored. An `.rri` records its files
    /// package-root-relative on purpose (byte-stability across checkouts);
    /// this is DWARF's `comp_dir` split, and the source root is the base
    /// directory the environment supplies: with one, entries join onto it,
    /// so diagnostics and the DWARF of locally emitted instances point at
    /// the producing package's real sources. Without one, entries become
    /// unfetchable virtual names carrying the package — locations degrade
    /// to name-only, and never resolve against the consumer's own tree.
    pub fn re_anchored_files(&self) -> Vec<String> {
        self.parsed
            .files
            .iter()
            .map(|rel| {
                if rel.starts_with('<') {
                    // Already virtual at the producer; keep verbatim.
                    rel.clone()
                } else if let Some(root) = &self.src_root {
                    root.join(rel).display().to_string()
                } else {
                    format!("<{}/{rel}>", self.name)
                }
            })
            .collect()
    }

    /// A source cache over the re-anchored table — real paths registered for
    /// lazy loading, virtual entries name-only — so spans inside imported
    /// items keep resolving in consumer diagnostics. `None` when the
    /// interface was printed without locations.
    pub fn sources(&self) -> Option<reussir_syntax::source::SourceCache> {
        let files = self.re_anchored_files();
        if files.is_empty() {
            return None;
        }
        let mut cache = reussir_syntax::source::SourceCache::new();
        for name in &files {
            if name.starts_with('<') {
                cache.add_unavailable(name);
            } else {
                cache.add_lazy_file(name);
            }
        }
        Some(cache)
    }
}

/// Load every interface, gating each header: it must be present (a plain
/// HIR dump is not an interface), carry the supported format, and describe
/// the package the flag named. `core` and the compiling package's own name
/// are reserved.
pub fn load<'tcx>(
    tcx: &TyCtxt<'tcx>,
    specs: &[ExternSpec],
    root_package: Option<&str>,
) -> Result<Vec<LoadedExtern<'tcx>>, String> {
    let mut loaded = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.name == "core" {
            return Err("`--extern core=...`: `core` is the builtin package".into());
        }
        if Some(spec.name.as_str()) == root_package {
            return Err(format!(
                "`--extern {}=...` shadows the package being compiled",
                spec.name
            ));
        }
        let text = std::fs::read_to_string(&spec.path)
            .map_err(|e| format!("cannot read `{}`: {e}", spec.path.display()))?;
        let parsed = build::parse_program(tcx, &text)
            .map_err(|e| format!("`{}`: {e}", spec.path.display()))?;
        let Some(header) = &parsed.header else {
            return Err(format!(
                "`{}` is a plain HIR dump, not a package interface; \
                 recompile the dependency with `--emit rri`",
                spec.path.display()
            ));
        };
        if header.format != RRI_FORMAT {
            return Err(format!(
                "`{}`: interface format {} (from {}) is not the supported {}; \
                 rebuild the dependency with this compiler",
                spec.path.display(),
                header.format,
                header.producer,
                RRI_FORMAT
            ));
        }
        if header.package != spec.name {
            return Err(format!(
                "`{}` describes package `{}`, but was loaded as `--extern {}=...`",
                spec.path.display(),
                header.package,
                spec.name
            ));
        }
        loaded.push(LoadedExtern {
            name: spec.name.clone(),
            src_root: spec.src_root.clone(),
            parsed,
        });
    }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: &[&str]) -> Vec<ExternPair> {
        v.iter().map(|x| x.parse().unwrap()).collect()
    }

    #[test]
    fn pairs_split_at_the_first_equals() {
        let pair: ExternPair = "dep=out/dep=v1.rri".parse().unwrap();
        assert_eq!(pair.name, "dep");
        assert_eq!(pair.value, PathBuf::from("out/dep=v1.rri"));
        assert!("dep".parse::<ExternPair>().is_err());
        assert!("=x".parse::<ExternPair>().is_err());
        assert!("dep=".parse::<ExternPair>().is_err());
    }

    #[test]
    fn file_tables_re_anchor_onto_the_source_root() {
        reussir_core::in_arena(|tcx| {
            let text = r#"
                interface 1 package "dep" producer "t";
                0 = "src/lib.rr";
                1 = "<prelude>";
            "#;
            let parsed = build::parse_program(tcx, text).unwrap();
            let mut ext = LoadedExtern {
                name: "dep".into(),
                src_root: None,
                parsed,
            };
            // No root: unfetchable virtual names carrying the package;
            // producer-virtual entries pass through.
            assert_eq!(ext.re_anchored_files(), ["<dep/src/lib.rr>", "<prelude>"]);
            ext.src_root = Some(PathBuf::from("/vendor/dep"));
            let files = ext.re_anchored_files();
            assert_eq!(
                files[0],
                std::path::Path::new("/vendor/dep")
                    .join("src/lib.rr")
                    .display()
                    .to_string()
            );
            assert_eq!(files[1], "<prelude>");
            let cache = ext.sources().expect("table is non-empty");
            assert_eq!(cache.len(), 2);
        });
    }

    #[test]
    fn specs_join_and_cross_check() {
        let specs = p(&["dep=a.rri", "util=b.rri"]);
        let specs = join_specs(&specs, &p(&["dep=/src"])).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs[0].src_root.as_deref(),
            Some(std::path::Path::new("/src"))
        );
        assert_eq!(specs[1].src_root, None);

        assert!(join_specs(&p(&["dep=a", "dep=b"]), &[]).is_err());
        assert!(join_specs(&p(&["dep=a"]), &p(&["other=/x"])).is_err());
        assert!(join_specs(&p(&["dep=a"]), &p(&["dep=/x", "dep=/y"])).is_err());
    }
}
