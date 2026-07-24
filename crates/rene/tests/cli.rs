//! End-to-end tests of the `rene` binary: build-directory locking, the clean
//! protocol, the source-graph table (`inspect`, and the three conditions that
//! rebuild it), and the runtime bake — driven through fake `rrc`/`cargo`/
//! `rustc` by default, with the real toolchain under `--ignored`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rene::db::BuildDir;
use rene::tables;

fn demo_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("demo-pkg")
        .join("rene.ncl")
}

/// A scratch package the test may mutate: a manifest plus a two-file source
/// graph (`src/lib.rr` declaring `mod math;`). Returns the manifest path.
fn package(dir: &Path, name: &str) -> PathBuf {
    let src = dir.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        src.join("lib.rr"),
        "mod math;\npub fn entry(n: u64) -> u64 { math::add(n, 10) }\n",
    )
    .unwrap();
    std::fs::write(
        src.join("math.rr"),
        "pub fn add(a: u64, b: u64) -> u64 { a + b }\n",
    )
    .unwrap();
    write_manifest(dir, name, "0.1.0")
}

/// (Re)write the scratch package's manifest.
fn write_manifest(dir: &Path, name: &str, version: &str) -> PathBuf {
    let path = dir.join("rene.ncl");
    std::fs::write(
        &path,
        format!("{{ package = {{ name = \"{name}\", version = \"{version}\" }} }}\n"),
    )
    .unwrap();
    path
}

fn rene(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rene"));
    cmd.args(args);
    cmd
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to spawn rene")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn build_fails_cleanly_without_a_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run(rene(&["build"]).current_dir(tmp.path()));
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("rene.ncl"), "stderr: {}", stderr(&out));
}

#[test]
fn build_refuses_a_build_dir_held_by_another_instance() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");
    let _held = BuildDir::open(&root).unwrap();

    let out = run(rene(&["build", "--manifest-path"])
        .arg(demo_manifest())
        .arg("--build-dir")
        .arg(&root));
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("in use"), "stderr: {}", stderr(&out));
}

#[test]
fn build_refuses_a_build_dir_with_a_pending_clean() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");
    // An interrupted clean: marker stamped, directory still there.
    let dir = BuildDir::open(&root).unwrap();
    dir.set_status(&[(tables::CLEANING_KEY, "true")]).unwrap();
    drop(dir);

    let out = run(rene(&["build", "--manifest-path"])
        .arg(demo_manifest())
        .arg("--build-dir")
        .arg(&root));
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("rene clean"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn clean_is_a_noop_without_a_build_dir_and_removes_one_when_free() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");

    let out = run(rene(&["clean", "--build-dir"]).arg(&root));
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let dir = BuildDir::open(&root).unwrap();
    dir.set_status(&[("some", "state")]).unwrap();
    drop(dir);
    std::fs::write(root.join("artifact"), b"x").unwrap();

    let out = run(rene(&["clean", "--build-dir"]).arg(&root));
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(!root.exists());
}

#[test]
fn clean_errors_out_when_the_db_is_in_use() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");
    let _held = BuildDir::open(&root).unwrap();

    let out = run(rene(&["clean", "--build-dir"]).arg(&root));
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("in use"), "stderr: {}", stderr(&out));
    assert!(root.exists());
}

/// A fake toolchain: stand-in `rrc`, `cargo`, and `rustc` scripts that record
/// every invocation, so a test can assert exactly what a build re-ran.
#[cfg(unix)]
struct Fakes {
    /// The directory holding the scripts and their run logs.
    dir: PathBuf,
    rrc: PathBuf,
    cargo: PathBuf,
    rustc: PathBuf,
    /// The libdir the fake `rustc` reports as its target libdir.
    libdir: PathBuf,
}

#[cfg(unix)]
impl Fakes {
    fn new(dir: &Path) -> Self {
        use std::os::unix::fs::PermissionsExt;

        let libdir = dir.join("fake-rust-libdir");
        std::fs::create_dir_all(&libdir).unwrap();

        let script = |name: &str, body: String| -> PathBuf {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        };

        let rustc = script(
            "fake-rustc",
            format!(
                "case \"$1\" in\n\
                 --version) echo 'rustc 9.9.9 (fake)';;\n\
                 --print) echo '{}';;\n\
                 esac\n",
                libdir.display()
            ),
        );
        // Counts invocations, fabricates the artifacts, and reports them the
        // way `--message-format=json` does. Runs with cwd = the unpacked
        // source dir.
        let cargo = script(
            "fake-cargo",
            "echo run >> \"$(dirname \"$0\")/cargo-runs\"\n\
             mkdir -p target/release/deps\n\
             rlib=\"$PWD/target/release/deps/libreussir_rt-0000.rlib\"\n\
             staticlib=\"$PWD/target/release/libreussir_rt.a\"\n\
             touch \"$rlib\" \"$staticlib\"\n\
             printf '{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"reussir-rt\"},\"filenames\":[\"%s\",\"%s\"]}\\n' \"$rlib\" \"$staticlib\"\n"
                .to_owned(),
        );
        // Stands in for `rrc --scan-deps`: counts invocations and reports the
        // package root's `.rr` files as the source graph, crate root first —
        // the same JSON shape rrc emits, so a file added to the package shows
        // up in the next scan.
        let rrc = script(
            "fake-rrc",
            "echo run >> \"$(dirname \"$0\")/rrc-runs\"\n\
             root=\"\"; prev=\"\"\n\
             for arg in \"$@\"; do\n\
               [ \"$prev\" = \"--package-root\" ] && root=\"$arg\"\n\
               prev=\"$arg\"\n\
             done\n\
             printf '{\"package\":\"pkg\",\"files\":[{\"path\":\"%s\",\"module\":[\"pkg\"]}' \"$root/lib.rr\"\n\
             for f in \"$root\"/*.rr; do\n\
               case \"$f\" in */lib.rr) continue;; esac\n\
               [ -f \"$f\" ] || continue\n\
               base=$(basename \"$f\" .rr)\n\
               printf ',{\"path\":\"%s\",\"module\":[\"pkg\",\"%s\"]}' \"$f\" \"$base\"\n\
             done\n\
             printf ']}\\n'\n"
                .to_owned(),
        );

        Fakes {
            dir: dir.to_owned(),
            rrc,
            cargo,
            rustc,
            libdir,
        }
    }

    /// How many times the named fake ran (`rrc`, `cargo`).
    fn runs(&self, tool: &str) -> usize {
        match std::fs::read_to_string(self.dir.join(format!("{tool}-runs"))) {
            Ok(log) => log.lines().count(),
            Err(_) => 0,
        }
    }

    /// A `rene` invocation wired to the fake toolchain.
    fn rene(&self, args: &[&str], manifest: &Path, build_dir: &Path) -> Output {
        run(rene(args)
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--build-dir")
            .arg(build_dir)
            .env("REUSSIR_RRC", &self.rrc)
            .env("REUSSIR_CARGO", &self.cargo)
            .env("REUSSIR_RUSTC", &self.rustc))
    }
}

/// `rene inspect --frozen`'s report, parsed.
#[cfg(unix)]
fn frozen_report(fakes: &Fakes, manifest: &Path, build_dir: &Path) -> serde_json::Value {
    let out = fakes.rene(&["inspect", "--frozen"], manifest, build_dir);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    serde_json::from_slice(&out.stdout).expect("inspect --frozen prints JSON")
}

/// The full bake driven through the fake toolchain: cheap, and it pins the
/// toolchain-facing contract (JSON artifact messages in, libdir listing out,
/// status recorded, second build reuses the bake).
#[cfg(unix)]
#[test]
fn build_bakes_the_runtime_and_prints_the_libdirs() {
    let tmp_dir = tempfile::tempdir().unwrap();
    // Canonicalized: macOS tempdirs sit behind the `/var -> /private/var`
    // symlink, and the deps line rene prints comes from cargo's artifact
    // report, whose paths the fake cargo roots at the resolved `$PWD`.
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // Stdout is exactly the two `--polyffi-libdir` directories: the detected
    // rust lib, then the freshly baked runtime's deps dir.
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    let deps_dir = root.join("reussir-rt/target/release/deps");
    assert_eq!(
        lines,
        [
            fakes.libdir.display().to_string(),
            deps_dir.display().to_string()
        ]
    );

    // The bundle really was unpacked, and the bake recorded.
    assert!(root.join("reussir-rt/Cargo.toml").is_file());
    let dir = BuildDir::open(&root).unwrap();
    assert!(dir.status(tables::RT_SOURCE_HASH_KEY).unwrap().is_some());
    assert!(dir.status(tables::RT_ARTIFACTS_KEY).unwrap().is_some());
    drop(dir);

    // A second build must reuse the bake: same output, no new cargo run.
    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), stdout);
    assert_eq!(fakes.runs("cargo"), 1, "the second build re-ran cargo");
}

/// The first build records the graph; a build that finds every recorded file
/// untouched must not invoke `rrc` at all.
#[cfg(unix)]
#[test]
fn build_is_a_no_op_when_no_source_changed() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(fakes.runs("rrc"), 1, "the first build must scan");

    for _ in 0..2 {
        let out = fakes.rene(&["build"], &manifest, &root);
        assert!(out.status.success(), "stderr: {}", stderr(&out));
        assert!(
            stderr(&out).contains("up to date"),
            "stderr: {}",
            stderr(&out)
        );
    }
    assert_eq!(
        fakes.runs("rrc"),
        1,
        "an unchanged package must not be re-scanned"
    );
    assert_eq!(
        frozen_report(&fakes, &manifest, &root)["state"],
        "up-to-date"
    );
}

/// Trigger 3: a modified file. The rebuild re-scans the *whole* graph rather
/// than refreshing that file's entry — the edit may have changed the graph.
#[cfg(unix)]
#[test]
fn build_rescans_the_whole_graph_when_a_source_changes() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    assert!(
        fakes.rene(&["build"], &manifest, &root).status.success(),
        "the first build failed"
    );

    // Edit a file *and* declare a new module: exactly the case a per-file
    // mtime refresh would miss, because the graph itself grew.
    std::fs::write(
        tmp.join("src/lib.rr"),
        "mod math;\nmod extra;\npub fn entry(n: u64) -> u64 { math::add(n, 10) }\n",
    )
    .unwrap();
    std::fs::write(tmp.join("src/extra.rr"), "pub fn x() -> u64 { 7 }\n").unwrap();

    assert_eq!(
        frozen_report(&fakes, &manifest, &root)["state"],
        "file-changed"
    );

    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(fakes.runs("rrc"), 2, "the changed package must be re-scanned");

    // The refreshed table is the new graph, `extra.rr` included (the record
    // is a set of files, reported in path order).
    let report = frozen_report(&fakes, &manifest, &root);
    assert_eq!(report["state"], "up-to-date");
    let files: Vec<&str> = report["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert_eq!(
        files,
        [
            tmp.join("src/extra.rr").display().to_string(),
            tmp.join("src/lib.rr").display().to_string(),
            tmp.join("src/math.rr").display().to_string(),
        ]
    );
    // …and the next build is a no-op again.
    assert!(fakes.rene(&["build"], &manifest, &root).status.success());
    assert_eq!(fakes.runs("rrc"), 2);
}

/// Trigger 3, the other half: a file that vanished from the graph must not
/// linger in the table and keep invalidating later builds.
#[cfg(unix)]
#[test]
fn a_removed_file_drops_out_of_the_refreshed_graph() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    assert!(fakes.rene(&["build"], &manifest, &root).status.success());

    std::fs::remove_file(tmp.join("src/math.rr")).unwrap();
    std::fs::write(tmp.join("src/lib.rr"), "pub fn entry() -> u64 { 0 }\n").unwrap();
    assert_eq!(
        frozen_report(&fakes, &manifest, &root)["state"],
        "file-changed"
    );

    assert!(fakes.rene(&["build"], &manifest, &root).status.success());
    let report = frozen_report(&fakes, &manifest, &root);
    assert_eq!(report["state"], "up-to-date");
    assert_eq!(report["files"].as_array().unwrap().len(), 1);

    // The stale row is gone, so the build stays a no-op.
    assert!(fakes.rene(&["build"], &manifest, &root).status.success());
    assert_eq!(fakes.runs("rrc"), 2);
}

/// Trigger 2: the evaluated manifest changed. Nothing on disk moved, but the
/// configuration the graph was taken under did.
#[cfg(unix)]
#[test]
fn build_rescans_when_the_config_checksum_changes() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    assert!(fakes.rene(&["build"], &manifest, &root).status.success());
    let before = frozen_report(&fakes, &manifest, &root);
    assert_eq!(before["state"], "up-to-date");

    write_manifest(&tmp, "demo", "0.2.0");
    let after = frozen_report(&fakes, &manifest, &root);
    assert_eq!(after["state"], "config-changed");
    assert_ne!(before["config_hash"], after["config_hash"]);

    assert!(fakes.rene(&["build"], &manifest, &root).status.success());
    assert_eq!(fakes.runs("rrc"), 2, "a changed config must force a re-scan");
    assert_eq!(
        frozen_report(&fakes, &manifest, &root)["state"],
        "up-to-date"
    );
}

/// `inspect` reports the graph as JSON: module paths, sizes, and the Blake3
/// digest of every file. Without `--frozen` it refreshes a stale record.
#[cfg(unix)]
#[test]
fn inspect_reports_the_recorded_graph() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    // A plain `inspect` on an unrecorded package scans it — no runtime bake.
    let out = fakes.rene(&["inspect"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(fakes.runs("rrc"), 1);
    assert_eq!(fakes.runs("cargo"), 0, "inspect must not bake the runtime");

    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["package"], "demo");
    assert_eq!(report["state"], "up-to-date");
    assert_eq!(report["config_hash"].as_str().unwrap().len(), 64);

    let files = report["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0]["module"], serde_json::json!(["pkg"]));
    assert_eq!(files[1]["module"], serde_json::json!(["pkg", "math"]));
    for file in files {
        let path = file["path"].as_str().unwrap();
        let contents = std::fs::read(path).unwrap();
        assert_eq!(file["size"].as_u64().unwrap(), contents.len() as u64);
        assert_eq!(
            file["hash"].as_str().unwrap(),
            blake3::hash(&contents).to_hex().to_string(),
            "recorded digest must be Blake3 of the contents of {path}"
        );
    }

    // A second `inspect` finds nothing to do.
    assert!(fakes.rene(&["inspect"], &manifest, &root).status.success());
    assert_eq!(fakes.runs("rrc"), 1);
}

/// `--frozen` is a pure query: it reports an unrecorded package as
/// uninitialized without running `rrc` or creating a build directory.
#[cfg(unix)]
#[test]
fn inspect_frozen_neither_scans_nor_creates_a_build_directory() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    let report = frozen_report(&fakes, &manifest, &root);
    assert_eq!(report["state"], "uninitialized");
    assert_eq!(report["files"].as_array().unwrap().len(), 0);
    assert_eq!(fakes.runs("rrc"), 0);
    assert!(!root.exists(), "--frozen created `{}`", root.display());
}

/// A package without a source root is a configuration error, named as such.
#[cfg(unix)]
#[test]
fn build_reports_a_package_without_sources() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = write_manifest(&tmp, "empty", "0.1.0");
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("src") && stderr(&out).contains("lib.rr"),
        "stderr: {}",
        stderr(&out)
    );
}

/// The real thing: bake `reussir-rt` with the host toolchain. Slow (a full
/// release build of the runtime) and network-dependent (crates.io + the
/// pinned mlir-sync git revision), hence ignored by default:
/// `cargo test -p rene -- --ignored`.
///
/// The source scan still runs through the fake `rrc`: this test is about the
/// runtime bake, and a real `rrc` needs the whole LLVM/MLIR build.
#[cfg(unix)]
#[test]
#[ignore = "builds reussir-rt with the host cargo (slow, needs network)"]
fn build_bakes_the_runtime_with_the_host_toolchain() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");
    let fakes = Fakes::new(tmp.path());

    let out = run(rene(&["build", "--manifest-path"])
        .arg(demo_manifest())
        .arg("--build-dir")
        .arg(&root)
        .env("REUSSIR_RRC", &fakes.rrc));
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let stdout = String::from_utf8(out.stdout).unwrap();
    let libdirs: Vec<PathBuf> = stdout.lines().map(PathBuf::from).collect();
    assert_eq!(libdirs.len(), 2, "stdout: {stdout}");
    // The detected rust libdir holds the standard library…
    assert!(libdirs[0].is_dir());
    // …and the baked deps dir holds the runtime rlib polyffi links against.
    assert!(libdirs[1].is_dir());
    assert!(
        std::fs::read_dir(&libdirs[1])
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().starts_with("libreussir_rt")),
        "no libreussir_rt in {}",
        libdirs[1].display()
    );
}
