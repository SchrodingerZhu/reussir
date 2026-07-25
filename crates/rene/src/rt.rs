//! Baking the bundled `reussir-rt` runtime inside the build directory.
//!
//! The `rene` binary embeds the runtime *source* as a zstd-compressed tar
//! (packed by `build.rs` — source, not binaries, is what keeps the bundle
//! cross-platform: rlibs and symbol hashes are version-locked to the rustc
//! that produced them, so the runtime must be built by the user's toolchain;
//! see docs/design/system-rust-runtime.md). `prepare` unpacks it into
//! `<build-dir>/reussir-rt/` and runs `cargo build --release` there, yielding
//! the rlib+deps directory polymorphic FFI links against (`rustc -L`) and the
//! static library final executables link. The result is recorded in the
//! status database and reused until the bundle hash or the toolchain changes.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::db::BuildDir;
use crate::tables;

/// The runtime's directory name inside the build directory.
pub const RT_DIR: &str = "reussir-rt";

/// The bundled runtime source (tar.zst), packed by `build.rs`.
static RT_BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reussir-rt.tar.zst"));

/// Blake3 hex digest of the bundle — the staleness key for the baked runtime.
pub fn bundle_hash() -> String {
    blake3::hash(RT_BUNDLE).to_hex().to_string()
}

/// Unpack the bundled source into `dest` (creating `dest/reussir-rt/…`).
pub fn unpack(dest: &Path) -> std::io::Result<()> {
    let decoder = zstd::Decoder::new(RT_BUNDLE)?;
    tar::Archive::new(decoder).unpack(dest)
}

/// The baked runtime: the toolchain that built it and the artifacts consumers
/// need. Stored as JSON under [`tables::RT_ARTIFACTS_KEY`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtArtifacts {
    /// The rustc that baked the runtime — also the one `rrc`'s link step and
    /// polyffi texture compiles must use (`REUSSIR_RUSTC`), so the runtime
    /// and everything linked against it agree on one toolchain.
    pub rustc: PathBuf,
    /// `rustc --version` of that toolchain.
    pub rustc_version: String,
    /// The toolchain's target libdir (`rustc --print target-libdir`) — the
    /// "detected rust lib" directory polyffi links the standard library from.
    pub rust_libdir: PathBuf,
    /// The baked `deps` directory: `libreussir_rt`'s rlib and every
    /// dependency rlib, for polyffi's `rustc -L`.
    pub deps_dir: PathBuf,
    /// The static library final executables link the runtime from.
    pub staticlib: PathBuf,
    pub rlib: PathBuf,
}

impl RtArtifacts {
    /// The library directories to pass to `rrc` (`--polyffi-libdir`): the
    /// detected rust lib and the freshly baked runtime.
    pub fn libdirs(&self) -> [&Path; 2] {
        [&self.rust_libdir, &self.deps_dir]
    }
}

/// The resolved `cargo`/`rustc` pair used for baking.
struct Toolchain {
    cargo: PathBuf,
    rustc: PathBuf,
    version: String,
    rust_libdir: PathBuf,
}

impl Toolchain {
    /// `REUSSIR_CARGO`/`REUSSIR_RUSTC` first (the explicit override rrc also
    /// honors), then cargo's own `CARGO`/`RUSTC`, then `PATH` — which
    /// respects rustup shims, so a `rust-toolchain.toml` in the user's
    /// project still picks the toolchain naturally.
    fn resolve() -> Result<Self, String> {
        let pick = |specific: &str, generic: &str, default: &str| {
            std::env::var_os(specific)
                .or_else(|| std::env::var_os(generic))
                .map_or_else(|| PathBuf::from(default), PathBuf::from)
        };
        let cargo = pick("REUSSIR_CARGO", "CARGO", "cargo");
        let rustc = pick("REUSSIR_RUSTC", "RUSTC", "rustc");
        let version = capture(&rustc, &["--version"])?;
        let rust_libdir = PathBuf::from(capture(&rustc, &["--print", "target-libdir"])?);
        Ok(Toolchain {
            cargo,
            rustc,
            version,
            rust_libdir,
        })
    }
}

fn capture(program: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run `{} {}`: {e}", program.display(), args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "`{} {}` failed: {}",
            program.display(),
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Ensure the runtime is baked in `dir`, reusing the recorded artifacts when
/// the bundle and toolchain are unchanged. The caller holds the build
/// directory's lock (it *is* the open [`BuildDir`]), so no other instance can
/// race the extract/build/record sequence.
pub fn prepare(dir: &BuildDir) -> Result<RtArtifacts, String> {
    let toolchain = Toolchain::resolve()?;
    let hash = bundle_hash();

    if let Some(cached) = fresh_bake(dir, &toolchain, &hash)? {
        tracing::info!("reussir-rt is up to date");
        return Ok(cached);
    }

    let src_dir = dir.root().join(RT_DIR);
    if src_dir.exists() {
        // Stale source (older rene or toolchain switch): re-extract from
        // scratch rather than patching files into an unknown tree.
        std::fs::remove_dir_all(&src_dir)
            .map_err(|e| format!("cannot clear `{}`: {e}", src_dir.display()))?;
    }
    tracing::info!(dest = %src_dir.display(), "extracting bundled reussir-rt source");
    unpack(dir.root()).map_err(|e| format!("cannot unpack the runtime bundle: {e}"))?;

    tracing::info!(
        cargo = %toolchain.cargo.display(),
        rustc = %toolchain.version,
        "building reussir-rt (once per toolchain or runtime change)"
    );
    let (rlib, staticlib) = cargo_build(&toolchain, &src_dir)?;

    // Cargo may report the uplifted rlib (`target/release/lib….rlib`) rather
    // than the hashed copy in `deps/`; polyffi's `rustc -L` needs `deps/`,
    // where the dependency rlibs live alongside the runtime's.
    let deps_dir = {
        let parent = rlib.parent().unwrap_or(Path::new("."));
        if parent.file_name() == Some("deps".as_ref()) {
            parent.to_owned()
        } else {
            parent.join("deps")
        }
    };
    let artifacts = RtArtifacts {
        rustc: toolchain.rustc,
        rustc_version: toolchain.version,
        rust_libdir: toolchain.rust_libdir,
        deps_dir,
        staticlib,
        rlib,
    };
    let record = serde_json::to_string(&artifacts).map_err(|e| e.to_string())?;
    dir.set_status(&[
        (tables::RT_SOURCE_HASH_KEY, hash.as_str()),
        (tables::RT_ARTIFACTS_KEY, record.as_str()),
    ])
    .map_err(|e| e.to_string())?;
    tracing::info!(staticlib = %artifacts.staticlib.display(), "reussir-rt ready");
    Ok(artifacts)
}

/// The recorded bake, if it is still valid: same bundle, same rustc, and the
/// artifacts still on disk.
fn fresh_bake(
    dir: &BuildDir,
    toolchain: &Toolchain,
    hash: &str,
) -> Result<Option<RtArtifacts>, String> {
    let recorded_hash = dir
        .status(tables::RT_SOURCE_HASH_KEY)
        .map_err(|e| e.to_string())?;
    if recorded_hash.as_deref() != Some(hash) {
        return Ok(None);
    }
    let Some(record) = dir
        .status(tables::RT_ARTIFACTS_KEY)
        .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let Ok(artifacts) = serde_json::from_str::<RtArtifacts>(&record) else {
        // An older rene may have written a different shape; just rebake.
        return Ok(None);
    };
    let valid = artifacts.rustc_version == toolchain.version
        && artifacts.rlib.is_file()
        && artifacts.staticlib.is_file();
    Ok(valid.then_some(artifacts))
}

/// Run `cargo build --release` in `src_dir`, following the JSON message
/// stream for progress and artifact locations (no directory guessing).
/// Returns the runtime's (rlib, staticlib).
fn cargo_build(toolchain: &Toolchain, src_dir: &Path) -> Result<(PathBuf, PathBuf), String> {
    let mut child = Command::new(&toolchain.cargo)
        .args([
            "build",
            "--release",
            "--message-format=json-render-diagnostics",
        ])
        .current_dir(src_dir)
        // Pin the rustc cargo delegates to, so the recorded version and
        // libdir describe the compiler that actually built the artifacts.
        .env("RUSTC", &toolchain.rustc)
        .stdout(Stdio::piped())
        // Diagnostics and cargo's own progress stream to the user unchanged.
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot run `{}`: {e}", toolchain.cargo.display()))?;

    let mut rlib = None;
    let mut staticlib = None;
    let stdout = std::io::BufReader::new(child.stdout.take().expect("stdout is piped"));
    for line in stdout.lines() {
        let line = line.map_err(|e| format!("lost cargo's output stream: {e}"))?;
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if msg["reason"] != "compiler-artifact" {
            continue;
        }
        let name = msg["target"]["name"].as_str().unwrap_or_default();
        tracing::debug!(%name, "compiled");
        if name != "reussir-rt" && name != "reussir_rt" {
            continue;
        }
        for file in msg["filenames"].as_array().into_iter().flatten() {
            let Some(path) = file.as_str() else { continue };
            match Path::new(path).extension().and_then(|e| e.to_str()) {
                Some("rlib") => rlib = Some(PathBuf::from(path)),
                // `.a` everywhere but MSVC, `.lib` there.
                Some("a") | Some("lib") => staticlib = Some(PathBuf::from(path)),
                _ => {}
            }
        }
    }
    let status = child
        .wait()
        .map_err(|e| format!("cannot wait for cargo: {e}"))?;
    if !status.success() {
        return Err("building reussir-rt failed (see cargo's output above)".to_owned());
    }
    match (rlib, staticlib) {
        (Some(rlib), Some(staticlib)) => Ok((rlib, staticlib)),
        _ => Err("cargo succeeded but did not report the runtime's rlib and staticlib".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundle_unpacks_to_a_standalone_crate() {
        let tmp = tempfile::tempdir().unwrap();
        unpack(tmp.path()).unwrap();

        let manifest = std::fs::read_to_string(tmp.path().join(RT_DIR).join("Cargo.toml")).unwrap();
        // Workspace inheritance must be fully resolved…
        assert!(!manifest.contains("workspace = true"), "{manifest}");
        assert!(manifest.contains("edition = \"2024\""), "{manifest}");
        // …the crate must be its own workspace root (or an enclosing
        // workspace — e.g. a user project in one — captures and rejects it)…
        assert!(manifest.contains("[workspace]"), "{manifest}");
        // …and the nightly-only optimization feature dropped, so the user's
        // stable toolchain can build the runtime. (Prose comments mentioning
        // nightly survive; only the feature string must be gone.)
        assert!(!manifest.contains("\"nightly\""), "{manifest}");

        assert!(tmp.path().join(RT_DIR).join("src").join("lib.rs").is_file());
        // Versions are pinned through the bundled lock file (no vendoring;
        // the host cargo fetches, the lock decides versions).
        assert!(tmp.path().join(RT_DIR).join("Cargo.lock").is_file());
    }

    #[test]
    fn bundle_hash_is_stable_hex() {
        let h = bundle_hash();
        assert_eq!(h.len(), 64);
        assert_eq!(h, bundle_hash());
    }
}
