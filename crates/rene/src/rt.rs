//! Baking the bundled `reussir-rt` runtime inside the build directory.
//!
//! The `rene` binary embeds the runtime *source* as a zstd-compressed tar
//! (packed by `build.rs` — source, not binaries, is what keeps the bundle
//! cross-platform: rlibs and symbol hashes are version-locked to the rustc
//! that produced them, so the runtime must be built by the user's toolchain;
//! see docs/design/system-rust-runtime.md). `prepare` unpacks it into
//! `<build-dir>/<target>/reussir-rt/` and runs Cargo with `--target <target>`
//! there, yielding the rlib+deps directory polymorphic FFI
//! links against (`rustc -L`) and the static library final executables link.
//! The result is recorded per target in the status database and reused until
//! the bundle hash or the toolchain changes.

use std::path::{Path, PathBuf};
use std::process::Stdio;

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

/// The baked runtime: the target, toolchain that built it, and artifacts
/// consumers need. Stored as JSON under [`tables::rt_artifacts_key`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtArtifacts {
    /// The machine target this runtime was baked for.
    pub target: String,
    /// The rustc that baked the runtime — also the one `rrc`'s link step and
    /// polyffi texture compiles must use (`REUSSIR_RUSTC`), so the runtime
    /// and everything linked against it agree on one toolchain.
    pub rustc: PathBuf,
    /// `rustc --version` of that toolchain.
    pub rustc_version: String,
    /// The toolchain's target libdir (`rustc --print target-libdir`) — the
    /// "detected rust lib" directory polyffi links the standard library from.
    pub rust_libdir: PathBuf,
    /// Cargo dependency directories needed to consume the runtime. With an
    /// explicit `--target`, target libraries and host-built proc macros live
    /// in different trees; polyffi's `rustc -L` needs both.
    pub deps_dirs: Vec<PathBuf>,
    /// The static library final executables link the runtime from.
    pub staticlib: PathBuf,
    pub rlib: PathBuf,
}

impl RtArtifacts {
    /// The library directories to pass to `rrc` (`--polyffi-libdir`): the
    /// target standard library and every Cargo dependency directory needed
    /// by the freshly baked runtime.
    pub fn libdirs(&self) -> impl Iterator<Item = &Path> {
        std::iter::once(self.rust_libdir.as_path())
            .chain(self.deps_dirs.iter().map(PathBuf::as_path))
    }
}

/// The resolved `cargo`/`rustc` pair used for baking.
struct Toolchain {
    cargo: PathBuf,
    rustc: PathBuf,
    version: String,
    /// The resolved target the bake builds for, and so the key of cargo's
    /// target-specific configuration.
    target: String,
    rust_libdir: PathBuf,
}

impl Toolchain {
    /// `REUSSIR_CARGO`/`REUSSIR_RUSTC` first (the explicit override rrc also
    /// honors), then cargo's own `CARGO`/`RUSTC`, then `PATH` — which
    /// respects rustup shims, so a `rust-toolchain.toml` in the user's
    /// project still picks the toolchain naturally.
    async fn resolve(requested_target: Option<&str>) -> Result<Self, String> {
        let cargo = pick_program("REUSSIR_CARGO", "CARGO", "cargo");
        let rustc = pick_program("REUSSIR_RUSTC", "RUSTC", "rustc");
        let verbose_version = capture(&rustc, &["-vV"]).await?;
        let version = verbose_version
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned();
        let host = verbose_version
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .ok_or_else(|| {
                format!(
                    "`{} -vV` reported no `host:` line:\n{verbose_version}",
                    rustc.display()
                )
            })?
            .trim()
            .to_owned();
        let target = match requested_target {
            Some(target) => validate_target(target)?,
            None => validate_target(&host)?,
        };
        let rust_libdir = PathBuf::from(
            capture(
                &rustc,
                &["--print", "target-libdir", "--target", target.as_str()],
            )
            .await?,
        );
        Ok(Toolchain {
            cargo,
            rustc,
            version,
            target,
            rust_libdir,
        })
    }
}

fn pick_program(specific: &str, generic: &str, default: &str) -> PathBuf {
    std::env::var_os(specific)
        .or_else(|| std::env::var_os(generic))
        .map_or_else(|| PathBuf::from(default), PathBuf::from)
}

/// Resolve the target used by a command that does not bake the runtime:
/// an explicit/profile target wins, otherwise use the selected rustc's host.
pub async fn resolve_target(requested: Option<&str>) -> Result<String, String> {
    if let Some(target) = requested {
        return validate_target(target);
    }
    let rustc = pick_program("REUSSIR_RUSTC", "RUSTC", "rustc");
    let verbose_version = capture(&rustc, &["-vV"]).await?;
    verbose_version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| {
            format!(
                "`{} -vV` reported no `host:` line:\n{verbose_version}",
                rustc.display()
            )
        })
        .and_then(|host| validate_target(host.trim()))
}

/// Target triples become one build-directory path component and one status
/// key component. Accept Rust-style triples, including dotted ARM targets,
/// while rejecting path syntax and empty/special components.
fn validate_target(target: &str) -> Result<String, String> {
    if target.ends_with(".json") {
        return Err(format!(
            "invalid target triple `{target}`: custom target JSON specifications are not supported"
        ));
    }
    let valid = !target.is_empty()
        && target != "."
        && target != ".."
        && target
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    if valid {
        Ok(target.to_owned())
    } else {
        Err(format!(
            "invalid target triple `{target}`: expected only ASCII letters, digits, `-`, `_`, or `.`"
        ))
    }
}

/// Whether the bundled runtime must use its platform allocator instead of
/// mimalloc for `target`.
///
/// mimalloc's backend is C and does not support WebAssembly targets. WASI
/// supplies the libc allocator used by reussir-rt's no-default-features
/// backend; freestanding wasm targets are classified the same way so they
/// fail, if unsupported, at that platform allocator boundary rather than in
/// mimalloc's native C build.
fn target_should_disable_mimalloc(target: &str) -> bool {
    matches!(target.split('-').next(), Some("wasm32" | "wasm64"))
}

/// Whether `RENE_RT_PLATFORM_ALLOCATOR` forces the platform-allocator backend
/// on every target.
///
/// mimalloc's private heap is invisible to the sanitizers, so a program built
/// through rene cannot be checked with ASan/LSan/MSan while the runtime keeps
/// its default backend — the allocation and free events never reach
/// compiler-rt. This is the escape hatch for that: it selects the same
/// `--no-default-features` backend the wasm targets take, whose libc/UCRT
/// allocation the sanitizers do interpose.
///
/// Any value other than `0` enables it. It changes which allocator a built
/// program uses, so it is a diagnostic knob, not a supported build mode.
fn platform_allocator_forced() -> bool {
    std::env::var_os("RENE_RT_PLATFORM_ALLOCATOR").is_some_and(|value| value != "0")
}

async fn capture(program: &Path, args: &[&str]) -> Result<String, String> {
    let out = compio::process::Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .expect("pipe stdout")
        .stderr(Stdio::piped())
        .expect("pipe stderr")
        .output()
        .await
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
///
/// `linker`, when given, is pinned for the bake's own links (the runtime
/// dylib) — the same pinning `rrc --linker` gives the driver-level links,
/// needed where rustc's discovery resolves the wrong tool (a vcvars shell
/// whose PATH carries a coreutils `link` ahead of MSVC's). It is not part of
/// the bake's freshness: whichever working linker linked the runtime, the
/// artifacts are equivalent.
pub async fn prepare(
    dir: &BuildDir,
    requested_target: Option<&str>,
    linker: Option<&Path>,
    progress: &indicatif::MultiProgress,
) -> Result<RtArtifacts, String> {
    let toolchain = Toolchain::resolve(requested_target).await?;
    let hash = bundle_hash();

    if let Some(cached) = fresh_bake(dir, &toolchain, &hash)? {
        tracing::debug!("reussir-rt is up to date");
        return Ok(cached);
    }

    let target_dir = dir.root().join(&toolchain.target);
    let src_dir = target_dir.join(RT_DIR);
    if src_dir.exists() {
        // Stale source: re-extract from scratch rather than patching files
        // into an unknown tree. Other targets' trees remain intact.
        std::fs::remove_dir_all(&src_dir)
            .map_err(|e| format!("cannot clear `{}`: {e}", src_dir.display()))?;
    }
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("cannot create `{}`: {e}", target_dir.display()))?;
    tracing::debug!(dest = %src_dir.display(), "extracting bundled reussir-rt source");
    unpack(&target_dir).map_err(|e| format!("cannot unpack the runtime bundle: {e}"))?;

    tracing::debug!(
        cargo = %toolchain.cargo.display(),
        rustc = %toolchain.version,
        "building reussir-rt"
    );
    // The bake is the build's one slow cold step (a real cargo build, with
    // the network on a fresh directory); a spinner carries that while
    // cargo's own output is captured — and replayed if the bake fails.
    let bar = progress.add(
        indicatif::ProgressBar::new_spinner()
            .with_message("baking reussir-rt (once per toolchain or runtime change)"),
    );
    bar.enable_steady_tick(std::time::Duration::from_millis(120));
    let result = cargo_build(&toolchain, &src_dir, linker).await;
    bar.finish_and_clear();
    progress.remove(&bar);
    let (rlib, staticlib, deps_dirs) = result?;

    let artifacts = RtArtifacts {
        target: toolchain.target.clone(),
        rustc: toolchain.rustc,
        rustc_version: toolchain.version,
        rust_libdir: toolchain.rust_libdir,
        deps_dirs,
        staticlib,
        rlib,
    };
    let record = serde_json::to_string(&artifacts).map_err(|e| e.to_string())?;
    let source_key = tables::rt_source_hash_key(&artifacts.target);
    let artifacts_key = tables::rt_artifacts_key(&artifacts.target);
    dir.set_status(&[
        (source_key.as_str(), hash.as_str()),
        (artifacts_key.as_str(), record.as_str()),
    ])
    .map_err(|e| e.to_string())?;
    tracing::debug!(staticlib = %artifacts.staticlib.display(), "reussir-rt ready");
    Ok(artifacts)
}

/// The recorded bake, if it is still valid: same bundle, same rustc, and the
/// artifacts still on disk.
fn fresh_bake(
    dir: &BuildDir,
    toolchain: &Toolchain,
    hash: &str,
) -> Result<Option<RtArtifacts>, String> {
    let source_key = tables::rt_source_hash_key(&toolchain.target);
    let artifacts_key = tables::rt_artifacts_key(&toolchain.target);
    let recorded_hash = dir.status(&source_key).map_err(|e| e.to_string())?;
    if recorded_hash.as_deref() != Some(hash) {
        return Ok(None);
    }
    let Some(record) = dir.status(&artifacts_key).map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    let Ok(artifacts) = serde_json::from_str::<RtArtifacts>(&record) else {
        // An older rene may have written a different shape; just rebake.
        return Ok(None);
    };
    let valid = artifacts.target == toolchain.target
        // The path is part of an explicit toolchain selection. In
        // particular, changing REUSSIR_RUSTC from a cached bare `rustc` to a
        // rustup/Nix absolute path must not silently restore the old value
        // merely because both binaries report the same version string.
        && artifacts.rustc == toolchain.rustc
        && artifacts.rustc_version == toolchain.version
        && artifacts.rlib.is_file()
        && artifacts.staticlib.is_file()
        && artifacts.libdirs().all(Path::is_dir);
    Ok(valid.then_some(artifacts))
}

/// Run `cargo build --release` in `src_dir`, reading the JSON message
/// stream for the artifact locations (no directory guessing). The stream is
/// parsed once cargo exits — live progress reaches the user through the
/// inherited stderr regardless. Returns the runtime's (rlib, staticlib).
async fn cargo_build(
    toolchain: &Toolchain,
    src_dir: &Path,
    linker: Option<&Path>,
) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), String> {
    let mut cmd = compio::process::Command::new(&toolchain.cargo);
    cmd.args(["build", "--release", "--target", toolchain.target.as_str()]);
    // The bundled mimalloc backend builds C code and is not yet a WebAssembly
    // allocator. WASI supplies the libc malloc family used by the runtime's
    // size-recovering fallback, so select that backend for wasm targets.
    if target_should_disable_mimalloc(&toolchain.target) || platform_allocator_forced() {
        cmd.arg("--no-default-features");
    }
    cmd.arg("--message-format=json-render-diagnostics")
        .current_dir(src_dir)
        // Pin the rustc cargo delegates to, so the recorded version and
        // libdir describe the compiler that actually built the artifacts.
        .env("RUSTC", &toolchain.rustc);
    if let Some(linker) = linker {
        // Cargo's target-specific linker configuration, as an environment
        // variable: `CARGO_TARGET_<TARGET>_LINKER`. Unlike RUSTFLAGS this
        // composes with whatever the user's environment already carries.
        let key = format!(
            "CARGO_TARGET_{}_LINKER",
            toolchain
                .target
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() {
                        ch.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        );
        cmd.env(key, linker);
    }
    // Both streams are captured: the JSON messages on stdout carry the
    // artifact locations, and cargo's human progress on stderr stays out of
    // the progress bars — replayed below only if the bake fails.
    let out = cmd
        .stdout(Stdio::piped())
        .expect("pipe stdout")
        .stderr(Stdio::piped())
        .expect("pipe stderr")
        .output()
        .await
        .map_err(|e| format!("cannot run `{}`: {e}", toolchain.cargo.display()))?;

    let mut rlib = None;
    let mut staticlib = None;
    let mut deps_dirs = std::collections::BTreeSet::new();
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if msg["reason"] != "compiler-artifact" {
            continue;
        }
        let name = msg["target"]["name"].as_str().unwrap_or_default();
        tracing::debug!(%name, "compiled");
        for file in msg["filenames"].as_array().into_iter().flatten() {
            let Some(path) = file.as_str() else { continue };
            let path = Path::new(path);
            if let Some(parent) = path.parent()
                && parent.file_name() == Some("deps".as_ref())
            {
                // Includes both target dependencies and host-built proc
                // macros when Cargo separates them for `--target`.
                deps_dirs.insert(parent.to_owned());
            }
            if name != "reussir-rt" && name != "reussir_rt" {
                continue;
            }
            match path.extension().and_then(|e| e.to_str()) {
                Some("rlib") => rlib = Some(path.to_owned()),
                // `.a` everywhere but MSVC, `.lib` there.
                Some("a") | Some("lib") => staticlib = Some(path.to_owned()),
                _ => {}
            }
        }
    }
    if !out.status.success() {
        return Err(format!(
            "building reussir-rt failed:\n{}",
            String::from_utf8_lossy(&out.stderr).trim_end()
        ));
    }
    match (rlib, staticlib) {
        (Some(rlib), Some(staticlib)) => {
            // Cargo may report the uplifted runtime rlib
            // (`…/release/lib….rlib`) rather than its hashed `deps/` copy.
            // Ensure that target dependency directory is present even when
            // no dependency artifact message named it.
            let parent = rlib.parent().unwrap_or(Path::new("."));
            let runtime_deps = if parent.file_name() == Some("deps".as_ref()) {
                parent.to_owned()
            } else {
                parent.join("deps")
            };
            deps_dirs.insert(runtime_deps);
            Ok((rlib, staticlib, deps_dirs.into_iter().collect()))
        }
        _ => Err("cargo succeeded but did not report the runtime's rlib and staticlib".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_backend_follows_the_target_architecture() {
        for target in [
            "wasm32-wasip1",
            "wasm32-unknown-unknown",
            "wasm64-unknown-unknown",
        ] {
            assert!(target_should_disable_mimalloc(target), "{target}");
        }
        for target in [
            "x86_64-unknown-linux-gnu",
            "aarch64-apple-darwin",
            "wasmtime-unknown-linux-gnu",
        ] {
            assert!(!target_should_disable_mimalloc(target), "{target}");
        }
    }

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
    fn target_triples_are_safe_path_components() {
        for target in [
            "x86_64-unknown-linux-gnu",
            "wasm32-unknown-unknown",
            "thumbv8m.main-none-eabi",
        ] {
            assert_eq!(validate_target(target).unwrap(), target);
        }
        for target in ["", ".", "..", "../wasm32", "x/y", "target.json"] {
            assert!(validate_target(target).is_err(), "{target}");
        }
    }

    #[test]
    fn bundle_hash_is_stable_hex() {
        let h = bundle_hash();
        assert_eq!(h.len(), 64);
        assert_eq!(h, bundle_hash());
    }
}
