//! The bundled `core` package sources.
//!
//! `core` is the language-coupled layer of the standard library, written in
//! Reussir and compiled like any other dependency — but shipped *inside*
//! rene, exactly like the `reussir-rt` bundle: a zstd-compressed tar packed
//! by `build.rs`, injected into every build's dependency graph unless a
//! package opts out with `package.no_core`. The sources unpack into the
//! build directory under a content-hashed name, so a rene upgrade
//! re-materializes them and the ordinary per-package freshness machinery
//! sees the change.

use std::path::{Path, PathBuf};

/// The bundled package source (tar.zst), packed by `build.rs` with fixed
/// entry metadata — the bytes depend only on the sources.
static CORE_BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/core-src.tar.zst"));

/// The bundle's content hash — the directory name component that keys
/// unpacked sources to this rene build.
fn content_hash() -> String {
    blake3::hash(CORE_BUNDLE).to_hex()[..16].to_string()
}

/// Where the bundle lives (or will live) under `build_dir`. Pure — inspect
/// paths print without writing anything.
pub fn dir(build_dir: &Path) -> PathBuf {
    build_dir.join("core-src").join(content_hash())
}

/// Materialize the bundle under `build_dir`, idempotently, and return its
/// root. The directory is content-keyed, so its manifest's presence means
/// the whole bundle is already there.
pub fn unpack(build_dir: &Path) -> Result<PathBuf, String> {
    let root = dir(build_dir);
    if root.join("rene.ncl").is_file() {
        return Ok(root);
    }
    let decoder = zstd::Decoder::new(CORE_BUNDLE)
        .map_err(|e| format!("cannot decode the bundled core source: {e}"))?;
    tar::Archive::new(decoder).unpack(&root).map_err(|e| {
        format!(
            "cannot unpack the bundled core source into {}: {e}",
            root.display()
        )
    })?;
    Ok(root)
}
