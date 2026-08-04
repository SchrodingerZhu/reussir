//! The bundled `core` package sources.
//!
//! `core` is the language-coupled layer of the standard library, written in
//! Reussir and compiled like any other dependency — but shipped *inside*
//! rene (exactly like the `reussir-rt` bundle) and injected into every
//! build's dependency graph unless a package opts out with
//! `package.no_core`. The sources unpack into the build directory under a
//! content-hashed name, so a rene upgrade re-materializes them and the
//! ordinary per-package freshness machinery sees the change.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

/// Every file of the bundled package, path-relative to its root.
const FILES: &[(&str, &str)] = &[
    ("rene.ncl", include_str!("../../../library/core/rene.ncl")),
    (
        "src/lib.rr",
        include_str!("../../../library/core/src/lib.rr"),
    ),
    (
        "src/cmp.rr",
        include_str!("../../../library/core/src/cmp.rr"),
    ),
    (
        "src/intrinsic/mod.rr",
        include_str!("../../../library/core/src/intrinsic/mod.rr"),
    ),
    (
        "src/intrinsic/math.rr",
        include_str!("../../../library/core/src/intrinsic/math.rr"),
    ),
];

/// The bundle's content hash — the directory name component that keys
/// unpacked sources to this rene build.
fn content_hash() -> String {
    let mut hasher = DefaultHasher::new();
    for (path, body) in FILES {
        path.hash(&mut hasher);
        body.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Where the bundle lives (or will live) under `build_dir`. Pure — inspect
/// paths print without writing anything.
pub fn dir(build_dir: &Path) -> PathBuf {
    build_dir.join("core-src").join(content_hash())
}

/// Materialize the bundle under `build_dir`, idempotently, and return its
/// root.
pub fn unpack(build_dir: &Path) -> Result<PathBuf, String> {
    let root = dir(build_dir);
    for (rel, body) in FILES {
        let path = root.join(rel);
        if path.is_file() {
            continue;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(&path, body).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(root)
}
