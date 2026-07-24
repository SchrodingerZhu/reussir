//! Table definitions for the build-status database (`reussir-build/rene.redb`).
//!
//! Every table and key the database holds is declared here, so the on-disk
//! schema can be reviewed in one place. Values are JSON strings — the schema
//! of each value is owned by the module that writes it (referenced per key
//! below).

use redb::TableDefinition;

/// Build status: string keys (the `*_KEY` constants) to JSON-encoded values.
pub const STATUS: TableDefinition<&str, &str> = TableDefinition::new("status");

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
