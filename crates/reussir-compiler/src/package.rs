//! Package discovery: map a package root directory to its source files and
//! module paths by following `mod` declarations from `lib.rr`.
//!
//! The layout follows the reference frontend (Rust-flavoured):
//!
//! * `lib.rr` is the crate root, module path `[package]`.
//! * A file's **location** determines its module path — `foo.rr` and
//!   `foo/mod.rr` are `[package, foo]`, `foo/bar.rr` is `[package, foo, bar]`
//!   — while `mod name;` declarations only drive *discovery*: a declaration
//!   in file `F` looks for `dir(F)/name.rr`, then `dir(F)/name/mod.rr`.
//! * Duplicate declarations and cycles are tolerated (each file is loaded
//!   once, keyed by its canonical path); a declaration with no source file is
//!   an error naming both candidates.
//! * The package name `core` is reserved for the built-in core package.
//!
//! Files are read into the compilation's [`SourceCache`] and parsed with one
//! **shared interner** (cross-file resolution compares interned keys); the
//! parses are handed back so elaboration never re-parses.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reussir_syntax::source::{FileId, SourceCache};
use reussir_syntax::{MultiThreadedTokenInterner, Parse, parse_with_interner};

/// One discovered source file: its id in the cache, its module path (package
/// name first), and its parse.
pub struct PackageSourceFile {
    pub file: FileId,
    pub module: Vec<String>,
    pub parse: Parse,
}

/// A fully loaded package: every reachable file, read and parsed. The first
/// entry is `lib.rr` ([`FileId::ROOT`]).
pub struct PackageSource {
    pub cache: SourceCache,
    pub files: Vec<PackageSourceFile>,
}

/// A discovery failure. `Diagnostics` means parse errors were found in `file`
/// — the caller renders them against the returned cache.
pub enum PackageError {
    /// A driver-level error (bad name, missing file, unreadable file).
    Message(String),
    /// Syntax errors in one of the package's files.
    ParseErrors {
        cache: SourceCache,
        file: FileId,
        errors: Vec<reussir_syntax::diagnostics::ParseError>,
    },
}

/// Load the package rooted at `root` under `name`: read `lib.rr`, follow
/// `mod` declarations transitively, and parse every file with `interner`.
pub fn load_package(
    root: &Path,
    name: &str,
    interner: &Arc<MultiThreadedTokenInterner>,
) -> Result<PackageSource, PackageError> {
    if name == "core" {
        return Err(PackageError::Message(
            "package name 'core' is reserved for the built-in core package".into(),
        ));
    }
    let root = root.canonicalize().map_err(|e| {
        PackageError::Message(format!("cannot open package root {}: {e}", root.display()))
    })?;
    let lib = root.join("lib.rr");
    if !lib.is_file() {
        return Err(PackageError::Message(format!(
            "package root {} has no lib.rr",
            root.display()
        )));
    }

    let mut cache = SourceCache::new();
    let mut files = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    // Depth-first from the crate root, matching declaration order.
    let mut work: Vec<PathBuf> = vec![lib];

    while let Some(path) = work.pop() {
        let canonical = path
            .canonicalize()
            .map_err(|e| PackageError::Message(format!("cannot open {}: {e}", path.display())))?;
        // A duplicate `mod` declaration or a declaration cycle re-reaches a
        // loaded file; each file joins the package once.
        if !visited.insert(canonical.clone()) {
            continue;
        }
        let source = std::fs::read_to_string(&canonical).map_err(|e| {
            PackageError::Message(format!("failed to read {}: {e}", path.display()))
        })?;
        let file = cache.add_file(&canonical, source);
        let parse = parse_with_interner(cache.source(file), interner.clone());
        if !parse.ok() {
            let errors = parse.errors.clone();
            return Err(PackageError::ParseErrors {
                cache,
                file,
                errors,
            });
        }

        // Follow this file's `mod` declarations before parsing siblings
        // (depth-first, so `work` is a stack pushed in reverse order).
        let dir = canonical.parent().expect("a source file has a directory");
        let mut children = Vec::new();
        for decl in mod_decls(&parse) {
            let flat = dir.join(format!("{decl}.rr"));
            let nested = dir.join(&decl).join("mod.rr");
            let child = if flat.is_file() {
                flat
            } else if nested.is_file() {
                nested
            } else {
                return Err(PackageError::Message(format!(
                    "module '{decl}' declared in {} has no source file; expected one of: {}, {}",
                    canonical.display(),
                    flat.display(),
                    nested.display()
                )));
            };
            children.push(child);
        }
        children.reverse();
        work.extend(children);

        let module = module_path(&root, &canonical, name);
        files.push(PackageSourceFile {
            file,
            module,
            parse,
        });
    }

    Ok(PackageSource { cache, files })
}

/// The `mod` declarations of a parsed file, in source order.
fn mod_decls(parse: &Parse) -> Vec<String> {
    let program = reussir_core::surface::program(&parse.root);
    program
        .iter()
        .filter_map(|stmt| match stmt.kind() {
            reussir_core::surface::StmtKind::Mod(_, name) => {
                Some(parse.resolver().resolve(name).to_owned())
            }
            _ => None,
        })
        .collect()
}

/// A file's module path from its location under the package root:
/// `lib.rr` → `[pkg]`, `foo.rr` and `foo/mod.rr` → `[pkg, foo]`,
/// `foo/bar.rr` → `[pkg, foo, bar]`.
fn module_path(root: &Path, file: &Path, pkg: &str) -> Vec<String> {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let mut segs: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if segs.last().is_some_and(|s| s == "mod") {
        segs.pop();
    }
    if segs.as_slice() == ["lib"] {
        segs.clear();
    }
    let mut path = vec![pkg.to_owned()];
    path.extend(segs);
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn interner() -> Arc<MultiThreadedTokenInterner> {
        Arc::new(reussir_syntax::new_threaded_interner())
    }

    #[test]
    fn discovers_nested_modules_with_location_derived_paths() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "lib.rr", "mod utils;\npub fn e() -> u64 { 0 }");
        write(
            dir.path(),
            "utils/mod.rr",
            "mod math;\npub fn o() -> u64 { 1 }",
        );
        write(dir.path(), "utils/math.rr", "pub fn d(x: u64) -> u64 { x }");
        let pkg = load_package(dir.path(), "mypkg", &interner()).unwrap_or_else(|_| panic!());
        let modules: Vec<Vec<String>> = pkg.files.iter().map(|f| f.module.clone()).collect();
        assert_eq!(
            modules,
            vec![
                vec!["mypkg".to_owned()],
                vec!["mypkg".to_owned(), "utils".to_owned()],
                vec!["mypkg".to_owned(), "utils".to_owned(), "math".to_owned()],
            ]
        );
        assert_eq!(pkg.files[0].file, FileId::ROOT);
    }

    #[test]
    fn tolerates_duplicate_declarations_and_cycles() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "lib.rr",
            "mod a;\nmod a;\npub fn e() -> u64 { 0 }",
        );
        write(dir.path(), "a.rr", "mod b;\npub fn f() -> u64 { 1 }");
        write(dir.path(), "b.rr", "mod a;\npub fn g() -> u64 { 2 }");
        let pkg = load_package(dir.path(), "p", &interner()).unwrap_or_else(|_| panic!());
        // lib, a, b — each once despite the duplicate and the a↔b cycle.
        assert_eq!(pkg.files.len(), 3);
    }

    #[test]
    fn reports_a_missing_module_file_with_both_candidates() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "lib.rr",
            "mod missing;\npub fn e() -> u64 { 0 }",
        );
        let Err(PackageError::Message(msg)) = load_package(dir.path(), "p", &interner()) else {
            panic!("expected a missing-module error");
        };
        assert!(msg.contains("module 'missing'"), "{msg}");
        assert!(msg.contains("has no source file"), "{msg}");
        assert!(msg.contains("missing.rr"), "{msg}");
        assert!(msg.contains("mod.rr"), "{msg}");
    }

    #[test]
    fn rejects_the_reserved_core_package_name() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "lib.rr", "pub fn e() -> u64 { 0 }");
        let Err(PackageError::Message(msg)) = load_package(dir.path(), "core", &interner()) else {
            panic!("expected the reserved-name error");
        };
        assert!(
            msg.contains("'core' is reserved for the built-in core package"),
            "{msg}"
        );
    }
}
