//! The build directory and its status database.
//!
//! `reussir-build/rene.meta` (see [`crate::tables`] for the schema) records
//! what the build directory already contains. TurboKV holds an exclusive
//! cross-process file lock for as long as the database is open, and that lock
//! *is* the build-directory lock: `build` opens the database first and keeps
//! it open for the whole build. Other builds wait for that lock, while
//! `clean` refuses to delete a directory whose database it cannot open.
//!
//! Deleting the database directory removes its lock file, so `clean` cannot
//! hold the lock *through* deletion. Instead it narrows the window: while still
//! holding the lock it stamps [`tables::CLEANING_KEY`], then unlocks and
//! deletes. A `build` that slips in between sees the stamp and refuses the
//! directory — the same refusal an *interrupted* clean earns, so a
//! half-deleted directory is never trusted. This is still best-effort (a
//! racer can theoretically interleave with `remove_dir_all` itself), which
//! is the accepted trade-off for not inventing a second locking scheme.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use tokio::runtime::Runtime;
use turbokv::{Db, DbError as TurboError, DbOptions, WriteBatch};

use crate::{deps, tables};

/// The status database's directory name inside the build directory.
pub const DB_DIR: &str = "rene.meta";

/// An open build directory, holding the status database (and its lock).
pub struct BuildDir {
    root: PathBuf,
    // Taken by Drop so TurboKV can finish maintenance and release the lock
    // before the runtime shuts down. Dropping Db alone is not a clean close.
    db: Option<Db>,
    // Rene uses compio. Keep TurboKV's Tokio I/O behind this synchronous
    // adapter; the current-thread runtime runs only during database calls.
    runtime: Runtime,
    // Serialize the scan and batch of a replacement if callers share this
    // handle across threads. A batch alone cannot protect the preceding scan.
    sources_write: Mutex<()>,
}

/// Why the build directory could not be opened or cleaned.
#[derive(Debug)]
pub enum DbError {
    /// Another process holds the status database open.
    InUse(PathBuf),
    Other(String),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::InUse(path) => write!(
                f,
                "build directory is in use: another rene holds `{}`",
                path.display()
            ),
            DbError::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for DbError {}

impl BuildDir {
    /// Create/open the build directory rooted at `root` and take its lock.
    pub fn open(root: &Path) -> Result<Self, DbError> {
        std::fs::create_dir_all(root)
            .map_err(|e| DbError::Other(format!("cannot create `{}`: {e}", root.display())))?;
        let path = root.join(DB_DIR);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(other)?;
        // Acknowledge each batch only after its WAL has been synced,
        // preserving immediate commit durability for the cleaning marker.
        let db = runtime
            .block_on(Db::open_with_options(&path, DbOptions::paranoid()))
            .map_err(|e| match e {
                TurboError::DirectoryLocked { .. } => DbError::InUse(path.clone()),
                other => DbError::Other(format!("cannot open `{}`: {other}", path.display())),
            })?;
        Ok(BuildDir {
            root: root.to_owned(),
            db: Some(db),
            runtime,
            sources_write: Mutex::new(()),
        })
    }

    /// Open a build directory once its current build, if any, releases the
    /// database. TurboKV exposes a non-blocking cross-process lock, so builds
    /// turn `DirectoryLocked` into an asynchronous wait here rather than
    /// forcing every caller to arrange its own serialization.
    pub async fn open_wait(root: &Path) -> Result<Self, DbError> {
        let mut waiting = false;
        loop {
            match Self::open(root) {
                Ok(dir) => return Ok(dir),
                Err(DbError::InUse(path)) => {
                    if !waiting {
                        tracing::debug!(database = %path.display(), "waiting for the build directory");
                        waiting = true;
                    }
                    compio::runtime::time::sleep(Duration::from_millis(10)).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Open an *existing* build directory, or `None` if there is none. Unlike
    /// [`BuildDir::open`] this never creates one, so a read-only query
    /// (`rene inspect --frozen`) does not leave a build directory behind in a
    /// package that was never built.
    pub fn open_existing(root: &Path) -> Result<Option<Self>, DbError> {
        let path = root.join(DB_DIR);
        if !path.try_exists().map_err(other)? {
            return Ok(None);
        }
        Self::open(root).map(Some)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read a status value. `None` both for an absent key and for a fresh
    /// database whose status namespace was never written.
    pub fn status(&self, key: &str) -> Result<Option<String>, DbError> {
        self.runtime
            .block_on(self.database().get(format!("{}{key}", tables::STATUS)))
            .map_err(other)?
            .map(String::from_utf8)
            .transpose()
            .map_err(other)
    }

    /// Is the directory's teardown in flight or interrupted? Set by
    /// [`clean`]; a build must refuse the directory and send the user back
    /// to `rene clean`.
    pub fn is_cleaning(&self) -> Result<bool, DbError> {
        Ok(self.status(tables::CLEANING_KEY)?.is_some())
    }

    /// Write status values in one atomic, durable batch.
    pub fn set_status(&self, entries: &[(&str, &str)]) -> Result<(), DbError> {
        let mut batch = WriteBatch::new();
        for (key, value) in entries {
            batch.put(format!("{}{key}", tables::STATUS), value);
        }
        self.runtime
            .block_on(self.database().write_batch(&batch))
            .map_err(other)
    }

    /// The recorded source graph, in path order (TurboKV scans keys in byte
    /// order, and the path follows a fixed prefix). Empty for a fresh database
    /// or one whose sources namespace was never written.
    pub fn sources(&self) -> Result<Vec<deps::SourceFile>, DbError> {
        self.runtime
            .block_on(self.database().scan_prefix(tables::SOURCES))
            .map_err(other)?
            .into_iter()
            .map(|(key, value)| {
                Ok(deps::SourceFile {
                    path: String::from_utf8(key[tables::SOURCES.len()..].to_vec())
                        .map_err(other)?,
                    record: serde_json::from_slice(&value).map_err(other)?,
                })
            })
            .collect()
    }

    /// Replace the whole source graph in one batch of deletes and puts. A
    /// rebuilt graph is a new snapshot, and a file that dropped out must not
    /// survive as a phantom row that would keep invalidating later builds.
    pub fn replace_sources(&self, files: &[deps::SourceFile]) -> Result<(), DbError> {
        let _guard = self.sources_write.lock().map_err(other)?;
        let mut batch = WriteBatch::new();
        for (key, _) in self
            .runtime
            .block_on(self.database().scan_prefix(tables::SOURCES))
            .map_err(other)?
        {
            batch.delete(key);
        }
        for file in files {
            batch.put(
                format!("{}{}", tables::SOURCES, file.path),
                serde_json::to_vec(&file.record).map_err(other)?,
            );
        }
        self.runtime
            .block_on(self.database().write_batch(&batch))
            .map_err(other)
    }

    fn database(&self) -> &Db {
        self.db.as_ref().expect("database is open until drop")
    }
}

impl Drop for BuildDir {
    fn drop(&mut self) {
        if let Some(db) = self.db.take()
            && let Err(e) = self.runtime.block_on(db.close())
        {
            tracing::warn!("cannot close `{}`: {e}", self.root.join(DB_DIR).display());
        }
    }
}

fn other(e: impl std::fmt::Display) -> DbError {
    DbError::Other(e.to_string())
}

/// The outcome of [`clean`].
#[derive(Debug, PartialEq, Eq)]
pub enum CleanOutcome {
    /// There was no build directory.
    Missing,
    Removed,
}

/// Delete the build directory, unless its status database is held open by
/// another instance. While still holding the lock, the database is stamped
/// [`tables::CLEANING_KEY`] so a `build` racing into the unlock/delete window
/// (or arriving after an interrupted clean) refuses the directory.
pub fn clean(root: &Path) -> Result<CleanOutcome, DbError> {
    if !root.exists() {
        return Ok(CleanOutcome::Missing);
    }
    let path = root.join(DB_DIR);
    if path.exists() {
        // Take the lock (open, never create) and stamp the teardown marker.
        // Failures other than the lock — a corrupt or truncated database —
        // must not wedge `clean`, whose whole job is disposing of such state.
        match BuildDir::open_existing(root) {
            Ok(Some(dir)) => {
                if let Err(e) = dir.set_status(&[(tables::CLEANING_KEY, "true")]) {
                    tracing::warn!("cannot stamp `{}` as cleaning: {e}", path.display());
                }
            }
            Ok(None) => {}
            Err(e @ DbError::InUse(_)) => return Err(e),
            Err(e) => tracing::warn!("ignoring unreadable `{}`: {e}", path.display()),
        }
    }
    std::fs::remove_dir_all(root)
        .map_err(|e| DbError::Other(format!("cannot remove `{}`: {e}", root.display())))?;
    Ok(CleanOutcome::Removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrips_and_missing_keys_are_none() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = BuildDir::open(tmp.path()).unwrap();
        assert_eq!(dir.status("nope").unwrap(), None);
        dir.set_status(&[("a", "1"), ("b", "2")]).unwrap();
        assert_eq!(dir.status("a").unwrap().as_deref(), Some("1"));
        assert_eq!(dir.status("b").unwrap().as_deref(), Some("2"));
        assert_eq!(dir.status("nope").unwrap(), None);
        dir.set_status(&[("a", "updated"), ("empty", "")]).unwrap();
        drop(dir);

        let dir = BuildDir::open_existing(tmp.path()).unwrap().unwrap();
        assert_eq!(dir.status("a").unwrap().as_deref(), Some("updated"));
        assert_eq!(dir.status("b").unwrap().as_deref(), Some("2"));
        assert_eq!(dir.status("empty").unwrap().as_deref(), Some(""));
        assert_eq!(dir.status("nope").unwrap(), None);
    }

    #[test]
    fn source_replacement_persists_and_preserves_status() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = BuildDir::open(tmp.path()).unwrap();
        assert!(dir.sources().unwrap().is_empty());
        let source = deps::SourceFile {
            path: "status/λ.rr".to_owned(),
            record: deps::SourceRecord {
                module: vec!["pkg".to_owned(), "λ".to_owned()],
                mtime_ns: u64::MAX,
                size: u64::MAX,
                hash: [255; 32],
            },
        };
        let removed = deps::SourceFile {
            path: "sources/old.rr".to_owned(),
            ..source.clone()
        };
        dir.set_status(&[("sources/old.rr", "kept")]).unwrap();
        dir.replace_sources(&[source.clone(), removed]).unwrap();
        let updated = deps::SourceFile {
            record: deps::SourceRecord {
                module: Vec::new(),
                hash: [0; 32],
                ..source.record
            },
            ..source
        };
        dir.replace_sources(std::slice::from_ref(&updated)).unwrap();
        drop(dir);

        let dir = BuildDir::open_existing(tmp.path()).unwrap().unwrap();
        assert_eq!(dir.sources().unwrap(), vec![updated]);
        assert_eq!(
            dir.status("sources/old.rr").unwrap().as_deref(),
            Some("kept")
        );
        dir.replace_sources(&[]).unwrap();
        drop(dir);

        let dir = BuildDir::open_existing(tmp.path()).unwrap().unwrap();
        assert!(dir.sources().unwrap().is_empty());
        assert_eq!(
            dir.status("sources/old.rr").unwrap().as_deref(),
            Some("kept")
        );
    }

    #[test]
    fn metadata_uses_a_new_directory_and_leaves_the_old_cache_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join("rene.redb");
        std::fs::write(&legacy, b"old cache").unwrap();
        assert!(BuildDir::open_existing(tmp.path()).unwrap().is_none());
        assert!(!tmp.path().join(DB_DIR).exists());

        let dir = BuildDir::open(tmp.path()).unwrap();
        assert!(tmp.path().join("rene.meta").is_dir());
        assert!(dir.sources().unwrap().is_empty());
        assert_eq!(std::fs::read(legacy).unwrap(), b"old cache");
    }

    #[test]
    fn clean_removes_unreadable_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("build");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(DB_DIR), b"not a database directory").unwrap();
        assert!(BuildDir::open_existing(&root).is_err());
        assert_eq!(clean(&root).unwrap(), CleanOutcome::Removed);
        assert!(!root.exists());
    }

    #[test]
    fn the_database_lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let held = BuildDir::open(tmp.path()).unwrap();
        assert!(matches!(BuildDir::open(tmp.path()), Err(DbError::InUse(_))));
        drop(held);
        BuildDir::open(tmp.path()).unwrap();
    }

    #[test]
    fn clean_refuses_a_held_directory_and_removes_a_free_one() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("reussir-build");

        assert_eq!(clean(&root).unwrap(), CleanOutcome::Missing);

        let held = BuildDir::open(&root).unwrap();
        match clean(&root) {
            Err(DbError::InUse(_)) => {}
            other => panic!("expected InUse, got {other:?}"),
        }
        drop(held);

        std::fs::write(root.join("stray-artifact"), b"x").unwrap();
        assert_eq!(clean(&root).unwrap(), CleanOutcome::Removed);
        assert!(!root.exists());
    }

    #[test]
    fn an_interrupted_clean_leaves_the_marker_a_build_must_refuse() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("reussir-build");

        // Simulate a clean dying between the stamp and the delete.
        let dir = BuildDir::open(&root).unwrap();
        dir.set_status(&[(tables::CLEANING_KEY, "true")]).unwrap();
        drop(dir);

        // A later open sees the teardown marker (this is what `build` checks)…
        let reopened = BuildDir::open(&root).unwrap();
        assert!(reopened.is_cleaning().unwrap());
        drop(reopened);

        // …and `clean` disposes of the carcass.
        assert_eq!(clean(&root).unwrap(), CleanOutcome::Removed);
        assert!(!root.exists());
    }
}
