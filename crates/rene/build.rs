//! Bundles the `reussir-rt` crate source into the `rene` binary as a
//! zstd-compressed tar, so `rene build` can materialize and bake the runtime
//! inside any project's build directory with the user's own Rust toolchain
//! (docs/design/system-rust-runtime.md). Shipping *source* is what makes the
//! bundle cross-platform: the produced rlib/staticlib are version-locked to
//! the rustc that builds them, so they must be baked on the user's machine,
//! never prebuilt here.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let crates_dir = manifest_dir.parent().unwrap();
    let rt_dir = crates_dir.join("reussir-rt");
    let workspace_dir = crates_dir.parent().unwrap();
    let workspace_toml = workspace_dir.join("Cargo.toml");
    let workspace_lock = workspace_dir.join("Cargo.lock");

    println!("cargo::rerun-if-changed={}", rt_dir.join("Cargo.toml").display());
    println!("cargo::rerun-if-changed={}", rt_dir.join("src").display());
    println!("cargo::rerun-if-changed={}", workspace_toml.display());
    println!("cargo::rerun-if-changed={}", workspace_lock.display());

    let manifest = standalone_manifest(&rt_dir.join("Cargo.toml"), &workspace_toml);
    // Dependencies are fetched by the host cargo (no vendoring), but their
    // versions are pinned: the workspace lock ships as the standalone crate's
    // lock file. Cargo accepts a superset Cargo.lock, keeping the pinned
    // versions of the reachable packages and pruning the rest.
    let lock = fs::read_to_string(&workspace_lock).expect("failed to read Cargo.lock");

    // Archive layout: everything under a `reussir-rt/` root, so unpacking into
    // the build directory yields `<build-dir>/reussir-rt/{Cargo.toml,src/**}`.
    let mut tar = tar::Builder::new(Vec::new());
    append_text(&mut tar, "reussir-rt/Cargo.toml", &manifest);
    append_text(&mut tar, "reussir-rt/Cargo.lock", &lock);
    tar.append_dir_all("reussir-rt/src", rt_dir.join("src"))
        .expect("failed to archive reussir-rt/src");
    let archive = tar.into_inner().expect("failed to finish the tar archive");

    let compressed =
        zstd::encode_all(&archive[..], 19).expect("failed to zstd-compress the runtime bundle");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("reussir-rt.tar.zst");
    fs::write(&out, compressed).expect("failed to write the runtime bundle");
}

fn append_text(tar: &mut tar::Builder<Vec<u8>>, path: &str, text: &str) {
    let mut header = tar::Header::new_gnu();
    header.set_size(text.len() as u64);
    header.set_mode(0o644);
    // Fixed mtime: the archive bytes (and thus the runtime staleness hash)
    // depend only on content, not on when the bundle was packed.
    header.set_mtime(0);
    header.set_cksum();
    tar.append_data(&mut header, path, text.as_bytes())
        .expect("failed to append the rewritten manifest");
}

/// Rewrite the in-workspace `reussir-rt` manifest into a standalone one:
/// resolve every `workspace = true` inheritance against the workspace
/// manifest, and pin an empty `[workspace]` table so the unpacked copy stays
/// its own workspace root even when the build directory sits inside some
/// enclosing cargo workspace (a user project often does).
fn standalone_manifest(crate_toml: &Path, workspace_toml: &Path) -> String {
    let mut doc: DocumentMut = fs::read_to_string(crate_toml)
        .unwrap()
        .parse()
        .expect("reussir-rt/Cargo.toml is not valid TOML");
    let ws: DocumentMut = fs::read_to_string(workspace_toml)
        .unwrap()
        .parse()
        .expect("workspace Cargo.toml is not valid TOML");

    let inherits = |item: &Item| {
        item.get("workspace")
            .and_then(Item::as_value)
            .and_then(|v| v.as_bool())
            == Some(true)
    };

    let pkg = doc["package"]
        .as_table_mut()
        .expect("[package] must be a table");
    let keys: Vec<String> = pkg.iter().map(|(k, _)| k.to_owned()).collect();
    for key in keys {
        if inherits(&pkg[&key]) {
            pkg[&key] = ws["workspace"]["package"][&key].clone();
        }
    }

    if let Some(deps) = doc.get_mut("dependencies").and_then(Item::as_table_like_mut) {
        let names: Vec<String> = deps.iter().map(|(k, _)| k.to_owned()).collect();
        for name in names {
            let spec = deps.get(&name).unwrap();
            if !inherits(spec) {
                continue;
            }
            // Only the plain `dep.workspace = true` form is handled; a
            // workspace dep that also sets `features`/`optional` locally
            // would need those keys merged over the workspace entry.
            assert!(
                spec.as_table_like().unwrap().len() == 1,
                "workspace dependency `{name}` carries extra keys; teach \
                 rene's build script to merge them"
            );
            deps.insert(&name, ws["workspace"]["dependencies"][&name].clone());
        }

        // The `nightly` feature of mlir_sync only gates optimizations (see the
        // comment in reussir-rt/Cargo.toml). The workspace pins a nightly
        // toolchain, but the bundled runtime is built by the *user's* rust —
        // drop the feature so a stable toolchain works
        // (docs/design/system-rust-runtime.md §2).
        if let Some(features) = deps
            .get_mut("mlir_sync")
            .and_then(|s| s.get_mut("features"))
            .and_then(Item::as_array_mut)
        {
            features.retain(|f| f.as_str() != Some("nightly"));
        }
    }

    // Cut the crate loose from any workspace it might land under: cargo
    // walks parent directories for a workspace root, and a foreign root
    // rejects the crate ("believes it's in a workspace when it's not").
    doc["workspace"] = toml_edit::Item::Table(toml_edit::Table::new());

    doc.to_string()
}
