//! The build directory and its status database.
//!
//! `reussir-build/rene.redb` (see [`crate::tables`] for the schema) records
//! what the build directory already contains. redb holds an exclusive
//! cross-process file lock for as long as the database is open, and that lock
//! *is* the build-directory lock: `build` opens the database first and keeps
//! it open for the whole build, and `clean` refuses to delete a directory
//! whose database it cannot open.
//!
//! Deleting the database file releases its lock, so `clean` cannot hold the
//! lock *through* the deletion. Instead it narrows the window: while still
//! holding the lock it stamps [`tables::CLEANING_KEY`], then unlocks and
//! deletes. A `build` that slips in between sees the stamp and refuses the
//! directory — the same refusal an *interrupted* clean earns, so a
//! half-deleted directory is never trusted. This is still best-effort (a
//! racer can theoretically interleave with `remove_dir_all` itself), which
//! is the accepted trade-off for not inventing a second locking scheme.

use std::path::{Path, PathBuf};

use redb::{Database, DatabaseError, ReadableDatabase, ReadableTable, TableError};

use crate::{deps, tables};

/// The status database's file name inside the build directory.
pub const DB_FILE: &str = "rene.redb";

/// An open build directory, holding the status database (and its lock).
pub struct BuildDir {
    root: PathBuf,
    db: Database,
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
        std::fs::create_dir_all(root).map_err(|e| {
            DbError::Other(format!("cannot create `{}`: {e}", root.display()))
        })?;
        let path = root.join(DB_FILE);
        let db = Database::create(&path).map_err(|e| match e {
            DatabaseError::DatabaseAlreadyOpen => DbError::InUse(path.clone()),
            other => DbError::Other(format!("cannot open `{}`: {other}", path.display())),
        })?;
        Ok(BuildDir {
            root: root.to_owned(),
            db,
        })
    }

    /// Open an *existing* build directory, or `None` if there is none. Unlike
    /// [`BuildDir::open`] this never creates one, so a read-only query
    /// (`rene inspect --frozen`) does not leave a build directory behind in a
    /// package that was never built.
    pub fn open_existing(root: &Path) -> Result<Option<Self>, DbError> {
        let path = root.join(DB_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let db = Database::open(&path).map_err(|e| match e {
            DatabaseError::DatabaseAlreadyOpen => DbError::InUse(path.clone()),
            other => DbError::Other(format!("cannot open `{}`: {other}", path.display())),
        })?;
        Ok(Some(BuildDir {
            root: root.to_owned(),
            db,
        }))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read a status value. `None` both for an absent key and for a fresh
    /// database whose status table was never written.
    pub fn status(&self, key: &str) -> Result<Option<String>, DbError> {
        let txn = self.db.begin_read().map_err(other)?;
        let table = match txn.open_table(tables::STATUS) {
            Ok(table) => table,
            Err(TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(other(e)),
        };
        let value = table.get(key).map_err(other)?;
        Ok(value.map(|v| v.value().to_owned()))
    }

    /// Is the directory's teardown in flight or interrupted? Set by
    /// [`clean`]; a build must refuse the directory and send the user back
    /// to `rene clean`.
    pub fn is_cleaning(&self) -> Result<bool, DbError> {
        Ok(self.status(tables::CLEANING_KEY)?.is_some())
    }

    /// Write status values in one transaction.
    pub fn set_status(&self, entries: &[(&str, &str)]) -> Result<(), DbError> {
        let txn = self.db.begin_write().map_err(other)?;
        {
            let mut table = txn.open_table(tables::STATUS).map_err(other)?;
            for (key, value) in entries {
                table.insert(key, value).map_err(other)?;
            }
        }
        txn.commit().map_err(other)
    }

    /// The recorded source graph, in the discovery order the scan stored it
    /// under (redb iterates keys in order, so the recorded index is what
    /// restores it). Empty both for a fresh database and for one whose
    /// sources table was never written.
    pub fn sources(&self) -> Result<Vec<deps::SourceFile>, DbError> {
        let txn = self.db.begin_read().map_err(other)?;
        let table = match txn.open_table(tables::SOURCES) {
            Ok(table) => table,
            Err(TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(other(e)),
        };
        let mut indexed = Vec::new();
        for row in table.iter().map_err(other)? {
            let (key, value) = row.map_err(other)?;
            let stored: StoredSource = serde_json::from_str(value.value()).map_err(|e| {
                DbError::Other(format!("corrupt source record for `{}`: {e}", key.value()))
            })?;
            indexed.push((
                stored.index,
                deps::SourceFile {
                    path: key.value().to_owned(),
                    record: stored.record,
                },
            ));
        }
        indexed.sort_by_key(|(index, _)| *index);
        Ok(indexed.into_iter().map(|(_, file)| file).collect())
    }

    /// Replace the whole source graph. The table is emptied first: a rebuilt
    /// graph is a new snapshot, and a file that dropped out of it must not
    /// survive as a phantom row that would keep invalidating later builds.
    pub fn replace_sources(&self, files: &[deps::SourceFile]) -> Result<(), DbError> {
        let txn = self.db.begin_write().map_err(other)?;
        {
            let mut table = txn.open_table(tables::SOURCES).map_err(other)?;
            table.retain(|_, _| false).map_err(other)?;
            for (index, file) in files.iter().enumerate() {
                let stored = StoredSource {
                    index,
                    record: file.record.clone(),
                };
                let value = serde_json::to_string(&stored).map_err(other)?;
                table
                    .insert(file.path.as_str(), value.as_str())
                    .map_err(other)?;
            }
        }
        txn.commit().map_err(other)
    }
}

/// A [`tables::SOURCES`] row: the file's record plus its position in the
/// scan, so the graph reads back in discovery order rather than the key
/// order redb iterates.
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredSource {
    index: usize,
    #[serde(flatten)]
    record: deps::SourceRecord,
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
    let path = root.join(DB_FILE);
    if path.exists() {
        // Take the lock (open, never create) and stamp the teardown marker.
        // Failures other than the lock — a corrupt or truncated database —
        // must not wedge `clean`, whose whole job is disposing of such state.
        match Database::open(&path) {
            Ok(db) => {
                let dir = BuildDir {
                    root: root.to_owned(),
                    db,
                };
                if let Err(e) = dir.set_status(&[(tables::CLEANING_KEY, "true")]) {
                    tracing::warn!("cannot stamp `{}` as cleaning: {e}", path.display());
                }
            }
            Err(DatabaseError::DatabaseAlreadyOpen) => return Err(DbError::InUse(path)),
            Err(e) => tracing::warn!("ignoring unreadable `{}`: {e}", path.display()),
        }
    }
    std::fs::remove_dir_all(root).map_err(|e| {
        DbError::Other(format!("cannot remove `{}`: {e}", root.display()))
    })?;
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
    }

    #[test]
    fn the_database_lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let held = BuildDir::open(tmp.path()).unwrap();
        assert!(matches!(
            BuildDir::open(tmp.path()),
            Err(DbError::InUse(_))
        ));
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
