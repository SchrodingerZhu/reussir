//! Table definitions for the build-status database (`reussir-build/rene.redb`).
//!
//! Every table and key the database holds is declared here, so the on-disk
//! schema can be reviewed in one place. Values are JSON strings — the schema
//! of each value is owned by the module that writes it (referenced per key
//! below).

use redb::TableDefinition;

/// Build status: string keys (the `*_KEY` constants) to JSON-encoded values.
pub const STATUS: TableDefinition<&str, &str> = TableDefinition::new("status");

/// The package's source graph, as last reported by `rrc --scan-deps`.
///
/// Keyed by position in the scan, so redb's own key order *is* discovery
/// order — the graph is read whole and replaced whole (see
/// [`crate::deps`]), never looked up by path, so the path is a column rather
/// than the key. The row is a redb tuple, which encodes it natively:
///
/// ```text
/// (path, module, mtime_ns, size, blake3)
/// ```
///
/// `module` is the file's module path in its written form (`pkg::math`);
/// segments are identifiers, so joining is unambiguous. `blake3` is the raw
/// 32-byte digest — hex is for [`crate::deps::SourceFile::to_json`] to
/// render, not for storage to carry.
pub const SOURCES: TableDefinition<u64, (&str, &str, u64, u64, &[u8; 32])> =
    TableDefinition::new("sources");

/// Blake3 hex digest of the evaluated manifest the [`SOURCES`] snapshot was
/// taken under (a bare hex string). A mismatch invalidates the snapshot: a
/// changed configuration may change the package's layout.
pub const SOURCES_CONFIG_HASH_KEY: &str = "sources.config-hash";

/// Blake3 hex digest of the bundled `reussir-rt` source archive the baked
/// runtime was built from (a JSON string). Written by [`crate::rt`]; a
/// mismatch with the running `rene`'s bundle invalidates the bake.
pub const RT_SOURCE_HASH_KEY: &str = "rt.source-hash";

/// The baked runtime's artifact record, a JSON [`crate::rt::RtArtifacts`]:
/// toolchain identity plus the paths `build` reports to the user.
pub const RT_ARTIFACTS_KEY: &str = "rt.artifacts";

/// Teardown marker (JSON `true`), set by `clean` while it still holds the
/// lock, right before it deletes the directory. A database carrying it is a
/// directory whose deletion is in flight or was interrupted: `build` refuses
/// it and directs the user back to `rene clean`, which is the only thing
/// that clears the marker (by deleting the directory).
pub const CLEANING_KEY: &str = "cleaning";
