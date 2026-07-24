//! The package's source graph: what `rrc --scan-deps` reported, recorded in
//! the build directory so a later build can tell whether anything moved.
//!
//! The record is a *whole-graph snapshot*, not a per-file cache. Modifying a
//! source file can change the graph itself — a new `mod` declaration pulls in
//! a file nobody has seen, a deleted one drops a subtree — so a stale entry
//! never means "re-stat that one file": it means the graph is unknown again,
//! and the only way to learn it is to ask `rrc` afresh and replace the whole
//! table. Hence [`prepare`] has exactly two outcomes: everything matched (no
//! `rrc` at all), or a full re-scan.
//!
//! Staleness is decided from `stat` alone — modification time (plus size,
//! which the same call yields and which catches a same-timestamp rewrite) —
//! so an unchanged package costs one `stat` per file and no reads. The
//! Blake3 digests recorded alongside are content identity for consumers of
//! [`crate::db::BuildDir::sources`]; they are not part of the check, which
//! must stay read-free.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use crate::db::BuildDir;
use crate::manifest::Loaded;
use crate::tables;

/// The package's source directory, relative to the manifest: `src/lib.rr` is
/// the crate root, and its `mod` declarations discover the rest.
pub const SRC_DIR: &str = "src";

/// The separator a module path is stored under (its written form, `pkg::math`
/// — see [`tables::SOURCES`]).
pub const MODULE_SEPARATOR: &str = "::";

/// Split a stored module path back into its segments. An empty column is an
/// empty path, not a single empty segment.
pub fn split_module(module: &str) -> Vec<String> {
    if module.is_empty() {
        return Vec::new();
    }
    module.split(MODULE_SEPARATOR).map(str::to_owned).collect()
}

/// What is recorded for one file of the source graph (see
/// [`tables::SOURCES`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRecord {
    /// The file's module path, package name first (`rrc --scan-deps`).
    pub module: Vec<String>,
    /// Modification time in nanoseconds since the Unix epoch — the staleness
    /// key. A file whose mtime predates the epoch records 0 and so always
    /// compares equal to another such file; the size check still applies.
    pub mtime_ns: u64,
    pub size: u64,
    /// Raw Blake3 digest of the file's contents.
    pub hash: [u8; blake3::OUT_LEN],
}

/// One file of the graph: its path and its record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: String,
    pub record: SourceRecord,
}

impl SourceFile {
    /// The `{"path": …, "module": …, …}` form `rene inspect` prints — where
    /// the digest becomes hex, the one place it needs to be readable.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "path": self.path,
            "module": self.record.module,
            "mtime_ns": self.record.mtime_ns,
            "size": self.record.size,
            "hash": blake3::Hash::from(self.record.hash).to_hex().to_string(),
        })
    }
}

/// Why the recorded graph cannot be trusted — the cases that force a re-scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Staleness {
    /// Every recorded file still matches; the graph stands.
    UpToDate,
    /// No graph recorded yet (a fresh build directory, or one written before
    /// this table existed).
    Uninitialized,
    /// The evaluated manifest changed, so the package's identity or layout
    /// may have too.
    ConfigChanged,
    /// A file of the graph moved. The graph may have moved with it.
    FileChanged { path: String, change: FileChange },
}

/// How a recorded file stopped matching what is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChange {
    /// Gone, or no longer readable.
    Missing,
    Mtime,
    Size,
}

impl Staleness {
    pub fn is_up_to_date(&self) -> bool {
        matches!(self, Staleness::UpToDate)
    }

    /// The stable tag `rene inspect` reports as `state`.
    pub fn tag(&self) -> &'static str {
        match self {
            Staleness::UpToDate => "up-to-date",
            Staleness::Uninitialized => "uninitialized",
            Staleness::ConfigChanged => "config-changed",
            Staleness::FileChanged { .. } => "file-changed",
        }
    }
}

impl std::fmt::Display for Staleness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Staleness::UpToDate => f.write_str("sources are up to date"),
            Staleness::Uninitialized => f.write_str("no source graph recorded yet"),
            Staleness::ConfigChanged => f.write_str("the manifest configuration changed"),
            Staleness::FileChanged { path, change } => {
                let what = match change {
                    FileChange::Missing => "is missing",
                    FileChange::Mtime => "was modified",
                    FileChange::Size => "changed size",
                };
                write!(f, "`{path}` {what}")
            }
        }
    }
}

/// The Blake3 digest of the evaluated manifest — the "config checksum" whose
/// change invalidates the recorded graph. It covers the *whole* evaluated
/// configuration, not just the fields [`crate::manifest::Manifest`] names, so
/// a manifest edit that only a later `rene` stage understands still forces a
/// re-scan.
pub fn config_hash(dump: &str) -> String {
    blake3::hash(dump.as_bytes()).to_hex().to_string()
}

/// The package's source root: `src/` next to the manifest.
pub fn source_root(loaded: &Loaded) -> PathBuf {
    loaded
        .path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(SRC_DIR)
}

/// Compare the recorded graph against the manifest and the files on disk,
/// reading no file contents.
pub fn staleness(dir: &BuildDir, config_hash: &str) -> Result<Staleness, String> {
    let recorded_config = dir
        .status(tables::SOURCES_CONFIG_HASH_KEY)
        .map_err(|e| e.to_string())?;
    let files = dir.sources().map_err(|e| e.to_string())?;
    // Both halves must be present: a config hash without files (or the
    // reverse) is a half-written record from an interrupted scan.
    let Some(recorded_config) = recorded_config else {
        return Ok(Staleness::Uninitialized);
    };
    if files.is_empty() {
        return Ok(Staleness::Uninitialized);
    }
    if recorded_config != config_hash {
        return Ok(Staleness::ConfigChanged);
    }
    for file in files {
        let Ok(meta) = std::fs::metadata(&file.path) else {
            return Ok(Staleness::FileChanged {
                path: file.path,
                change: FileChange::Missing,
            });
        };
        if mtime_ns(&meta) != file.record.mtime_ns {
            return Ok(Staleness::FileChanged {
                path: file.path,
                change: FileChange::Mtime,
            });
        }
        if meta.len() != file.record.size {
            return Ok(Staleness::FileChanged {
                path: file.path,
                change: FileChange::Size,
            });
        }
    }
    Ok(Staleness::UpToDate)
}

/// The outcome of [`prepare`]: the graph in force, and how it was obtained.
pub struct Prepared {
    pub files: Vec<SourceFile>,
    /// Why the table was rebuilt, or [`Staleness::UpToDate`] if it was not.
    pub reason: Staleness,
}

impl Prepared {
    /// Did this run have to ask `rrc` for the graph?
    pub fn rescanned(&self) -> bool {
        !self.reason.is_up_to_date()
    }
}

/// Bring the recorded source graph up to date, re-scanning through `rrc` when
/// it cannot be trusted. A run where every recorded file still matches does
/// nothing at all — no subprocess, no write.
pub fn prepare(dir: &BuildDir, loaded: &Loaded) -> Result<Prepared, String> {
    let hash = config_hash(&loaded.dump);
    let reason = staleness(dir, &hash)?;
    if reason.is_up_to_date() {
        tracing::info!("package sources are up to date");
        return Ok(Prepared {
            files: dir.sources().map_err(|e| e.to_string())?,
            reason,
        });
    }

    tracing::info!(%reason, "scanning the package source graph");
    let root = source_root(loaded);
    let files = scan(&root, &loaded.manifest.package.name)?;
    dir.replace_sources(&files).map_err(|e| e.to_string())?;
    dir.set_status(&[(tables::SOURCES_CONFIG_HASH_KEY, hash.as_str())])
        .map_err(|e| e.to_string())?;
    tracing::info!(files = files.len(), "recorded the source graph");
    Ok(Prepared { files, reason })
}

/// Ask `rrc --scan-deps` for the package's source graph and stat every file
/// it names.
pub fn scan(root: &Path, package: &str) -> Result<Vec<SourceFile>, String> {
    let rrc = resolve_rrc();
    if !root.is_dir() {
        return Err(format!(
            "package source root `{}` does not exist; a package's crate root \
             is `{}/lib.rr`",
            root.display(),
            SRC_DIR
        ));
    }
    tracing::debug!(rrc = %rrc.display(), root = %root.display(), "scanning dependencies");
    let out = Command::new(&rrc)
        .arg("--package-root")
        .arg(root)
        .arg("--package-name")
        .arg(package)
        .arg("--scan-deps")
        .output()
        .map_err(|e| format!("cannot run `{}`: {e}", rrc.display()))?;
    if !out.status.success() {
        // rrc has already rendered its diagnostics to stderr; pass them on
        // rather than paraphrasing.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stderr = stderr.trim();
        return Err(if stderr.is_empty() {
            format!("`{} --scan-deps` failed", rrc.display())
        } else {
            format!("`{} --scan-deps` failed:\n{stderr}", rrc.display())
        });
    }
    let stdout = String::from_utf8(out.stdout)
        .map_err(|e| format!("`{} --scan-deps` printed invalid UTF-8: {e}", rrc.display()))?;
    parse_scan(&stdout)?
        .into_iter()
        .map(|(path, module)| record(path, module))
        .collect()
}

/// The `rrc` executable: `REUSSIR_RRC` if set (the override rene's runtime
/// bake uses for `cargo`/`rustc` too), else `rrc` through `PATH`.
fn resolve_rrc() -> PathBuf {
    std::env::var_os("REUSSIR_RRC").map_or_else(|| PathBuf::from("rrc"), PathBuf::from)
}

/// Pull `(path, module)` out of `rrc --scan-deps`' JSON, in the order it
/// reported them (discovery order).
fn parse_scan(stdout: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let graph: serde_json::Value = serde_json::from_str(stdout)
        .map_err(|e| format!("cannot parse the `--scan-deps` output: {e}"))?;
    let files = graph["files"]
        .as_array()
        .ok_or("the `--scan-deps` output has no `files` array")?;
    files
        .iter()
        .map(|file| {
            let path = file["path"]
                .as_str()
                .ok_or("a `--scan-deps` entry has no `path`")?
                .to_owned();
            let module = file["module"]
                .as_array()
                .map(|segs| {
                    segs.iter()
                        .map(|s| s.as_str().unwrap_or_default().to_owned())
                        .collect()
                })
                .unwrap_or_default();
            Ok((path, module))
        })
        .collect()
}

/// Stat and hash one scanned file.
fn record(path: String, module: Vec<String>) -> Result<SourceFile, String> {
    let meta = std::fs::metadata(&path).map_err(|e| format!("cannot stat `{path}`: {e}"))?;
    let contents = std::fs::read(&path).map_err(|e| format!("cannot read `{path}`: {e}"))?;
    Ok(SourceFile {
        record: SourceRecord {
            module,
            mtime_ns: mtime_ns(&meta),
            size: meta.len(),
            hash: *blake3::hash(&contents).as_bytes(),
        },
        path,
    })
}

/// A file's modification time in nanoseconds since the Unix epoch. Anything
/// unrepresentable — no mtime on this platform, a pre-epoch timestamp, or a
/// value past year 2554 — collapses to 0, which simply makes that file's
/// mtime uninformative (the size check still applies).
fn mtime_ns(meta: &std::fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .and_then(|d| u64::try_from(d.as_nanos()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, contents: &str) -> PathBuf {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
        path
    }

    /// A recorded graph of one file, as a scan would leave it.
    fn record_one(dir: &BuildDir, path: &Path, config: &str) {
        let file = record(path.display().to_string(), vec!["p".to_owned()]).unwrap();
        dir.replace_sources(std::slice::from_ref(&file)).unwrap();
        dir.set_status(&[(tables::SOURCES_CONFIG_HASH_KEY, config)])
            .unwrap();
    }

    #[test]
    fn config_hash_is_blake3_of_the_evaluated_manifest() {
        let hash = config_hash("{ package = { name = \"p\" } }");
        assert_eq!(hash.len(), 64);
        assert_eq!(
            hash,
            blake3::hash(b"{ package = { name = \"p\" } }")
                .to_hex()
                .to_string()
        );
        assert_ne!(hash, config_hash("{}"));
    }

    #[test]
    fn a_fresh_build_directory_is_uninitialized() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();
        assert_eq!(staleness(&dir, "cfg").unwrap(), Staleness::Uninitialized);
    }

    #[test]
    fn an_unchanged_graph_is_up_to_date() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/lib.rr", "pub fn e() -> u64 { 0 }");
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();
        record_one(&dir, &src, "cfg");
        assert_eq!(staleness(&dir, "cfg").unwrap(), Staleness::UpToDate);
    }

    #[test]
    fn a_changed_config_invalidates_the_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/lib.rr", "pub fn e() -> u64 { 0 }");
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();
        record_one(&dir, &src, "cfg");
        assert_eq!(
            staleness(&dir, "other-cfg").unwrap(),
            Staleness::ConfigChanged
        );
    }

    #[test]
    fn a_touched_or_deleted_file_invalidates_the_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/lib.rr", "pub fn e() -> u64 { 0 }");
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();
        record_one(&dir, &src, "cfg");

        // A rewrite moves the mtime (and here the size too); either way the
        // whole graph is suspect, not just this file.
        std::fs::write(&src, "pub fn e() -> u64 { 1 }\n").unwrap();
        match staleness(&dir, "cfg").unwrap() {
            Staleness::FileChanged { path, change } => {
                assert_eq!(path, src.display().to_string());
                assert!(matches!(change, FileChange::Mtime | FileChange::Size));
            }
            other => panic!("expected a file change, got {other:?}"),
        }

        std::fs::remove_file(&src).unwrap();
        assert!(matches!(
            staleness(&dir, "cfg").unwrap(),
            Staleness::FileChanged {
                change: FileChange::Missing,
                ..
            }
        ));
    }

    /// A same-mtime rewrite (coarse filesystem timestamps) still trips the
    /// size half of the same `stat`.
    #[test]
    fn a_size_change_alone_invalidates_the_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/lib.rr", "pub fn e() -> u64 { 0 }");
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();

        let mut file = record(src.display().to_string(), vec!["p".to_owned()]).unwrap();
        // Pretend the recording saw a shorter file at this very mtime.
        file.record.size -= 1;
        dir.replace_sources(std::slice::from_ref(&file)).unwrap();
        dir.set_status(&[(tables::SOURCES_CONFIG_HASH_KEY, "cfg")])
            .unwrap();

        assert!(matches!(
            staleness(&dir, "cfg").unwrap(),
            Staleness::FileChanged {
                change: FileChange::Size,
                ..
            }
        ));
    }

    #[test]
    fn parses_the_scan_deps_graph_in_discovery_order() {
        let files = parse_scan(
            r#"{
                 "package": "p",
                 "files": [
                   { "path": "/pkg/src/lib.rr", "module": ["p"] },
                   { "path": "/pkg/src/math.rr", "module": ["p", "math"] }
                 ]
               }"#,
        )
        .unwrap();
        assert_eq!(
            files,
            vec![
                ("/pkg/src/lib.rr".to_owned(), vec!["p".to_owned()]),
                (
                    "/pkg/src/math.rr".to_owned(),
                    vec!["p".to_owned(), "math".to_owned()]
                ),
            ]
        );
        assert!(parse_scan("not json").is_err());
        assert!(parse_scan(r#"{"package": "p"}"#).is_err());
    }

    #[test]
    fn a_record_carries_the_blake3_digest_of_the_contents() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/lib.rr", "contents");
        let file = record(src.display().to_string(), vec!["p".to_owned()]).unwrap();
        assert_eq!(file.record.size, 8);
        assert_eq!(file.record.hash, *blake3::hash(b"contents").as_bytes());
        // Hex is a rendering, not what the row carries.
        assert_eq!(
            file.to_json()["hash"],
            serde_json::json!(blake3::hash(b"contents").to_hex().to_string())
        );
    }

    /// Module paths round-trip through their stored `pkg::math` form.
    #[test]
    fn module_paths_round_trip_through_the_stored_column() {
        for module in [
            vec![],
            vec!["p".to_owned()],
            vec!["p".to_owned(), "math".to_owned(), "ops".to_owned()],
        ] {
            assert_eq!(split_module(&module.join(MODULE_SEPARATOR)), module);
        }
    }

    /// The graph reads back in scan order even when the paths sort the other
    /// way — the key is the scan position, not the path.
    #[test]
    fn the_recorded_graph_keeps_discovery_order() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = BuildDir::open(&tmp.path().join("build")).unwrap();
        let files: Vec<SourceFile> = ["zzz.rr", "aaa.rr", "mmm.rr"]
            .iter()
            .map(|name| {
                let path = write(tmp.path(), name, "pub fn e() -> u64 { 0 }");
                record(path.display().to_string(), vec!["p".to_owned()]).unwrap()
            })
            .collect();
        dir.replace_sources(&files).unwrap();
        assert_eq!(dir.sources().unwrap(), files);
    }
}
