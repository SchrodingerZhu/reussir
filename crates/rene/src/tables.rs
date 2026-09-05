//! Key namespaces for the build-status database (`reussir-build/rene.meta`).
//!
//! Every namespace and key the database holds is declared here, so the on-disk
//! schema can be reviewed in one place. Values are JSON strings — the schema
//! of each value is owned by the module that writes it (referenced per key
//! below).

/// Build status: string keys (the `*_KEY` constants) to JSON-encoded values.
/// The prefix separates status keys from source paths in TurboKV's keyspace.
pub const STATUS: &str = "status/";

/// The package's source graph, as last reported by `rrc --scan-deps`: one
/// row per file, keyed by its path, holding what staleness is judged from.
/// Keys are this prefix followed by the UTF-8 path; values are JSON-encoded
/// [`crate::deps::SourceRecord`]s, including module segments and the digest's
/// 32 bytes. Hex is for [`crate::deps::SourceFile::to_json`] to render.
///
/// The graph is a *set* of files: nothing in rene depends on the order the
/// scan walked them in (staleness checks every row, and `rrc` rediscovers
/// the graph itself), so rows simply come back in path order. Written by
/// [`crate::deps`], always wholesale — the graph is a single snapshot, so a
/// rebuild replaces every row rather than updating one.
pub const SOURCES: &str = "sources/";

/// Blake3 hex digest of the evaluated manifest the [`SOURCES`] snapshot was
/// taken under (a bare hex string). A mismatch invalidates the snapshot: a
/// changed configuration may change the package's layout.
pub const SOURCES_CONFIG_HASH_KEY: &str = "sources.config-hash";

/// Status key for the Blake3 hex digest of the bundled `reussir-rt` source
/// archive baked for `target`. Each target keeps an independent record and
/// source tree, so switching targets does not evict a usable bake.
pub fn rt_source_hash_key(target: &str) -> String {
    format!("rt.{target}.source-hash")
}

/// Status key for a target's baked runtime artifact record, a JSON
/// [`crate::rt::RtArtifacts`]: target, toolchain identity, and artifact
/// paths.
pub fn rt_artifacts_key(target: &str) -> String {
    format!("rt.{target}.artifacts")
}

/// The status key of one built product's record, a JSON
/// [`crate::compile::ProductRecord`]: the fingerprint it was built under and
/// where the artifact landed. Records of targets that leave the manifest
/// simply go stale in place — nothing reads them again, and a config change
/// re-fingerprints the rest.
pub fn product_key(profile: &str, target: &str) -> String {
    format!("product.{profile}.{target}")
}

/// The status key of one built dependency's record, a JSON
/// [`crate::fresh::DepRecord`]: the components its freshness is judged
/// from, and where its interface and archive landed.
pub fn dep_product_key(profile: &str, name: &str) -> String {
    format!("product.{profile}.dep.{name}")
}

/// The status key of one dependency's recorded source graph, a JSON
/// `Vec<`[`crate::deps::SourceFile`]`>` — the per-dependency counterpart of
/// the root's [`SOURCES`] table, compact because dependencies are read-only
/// inputs scanned wholesale.
pub fn dep_sources_key(name: &str) -> String {
    format!("sources.dep.{name}")
}

/// Teardown marker (JSON `true`), set by `clean` while it still holds the
/// lock, right before it deletes the directory. A database carrying it is a
/// directory whose deletion is in flight or was interrupted: `build` refuses
/// it and directs the user back to `rene clean`, which is the only thing
/// that clears the marker (by deleting the directory).
pub const CLEANING_KEY: &str = "cleaning";
