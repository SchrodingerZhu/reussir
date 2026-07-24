//! End-to-end tests of the `rene` binary against the demo package in
//! `tests/demo-pkg`: build-directory locking, the clean protocol, and the
//! runtime bake (with a fake toolchain by default; the real one under
//! `--ignored`).

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

/// The full bake driven through fake `cargo`/`rustc` scripts: cheap, and it
/// pins the toolchain-facing contract (JSON artifact messages in, libdir
/// listing out, status recorded, second build reuses the bake).
#[cfg(unix)]
#[test]
fn build_bakes_the_runtime_and_prints_the_libdirs() {
    use std::os::unix::fs::PermissionsExt;

    let tmp_dir = tempfile::tempdir().unwrap();
    // Canonicalized: macOS tempdirs sit behind the `/var -> /private/var`
    // symlink, and the deps line rene prints comes from cargo's artifact
    // report, whose paths the fake cargo roots at the resolved `$PWD`.
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let libdir = tmp.join("fake-rust-libdir");
    std::fs::create_dir_all(&libdir).unwrap();

    let script = |name: &str, body: String| -> PathBuf {
        let path = tmp.join(name);
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
    // Counts invocations, fabricates the artifacts, and reports them the way
    // `--message-format=json` does. Runs with cwd = the unpacked source dir.
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

    let build = |root: &Path| {
        run(rene(&["build", "--manifest-path"])
            .arg(demo_manifest())
            .arg("--build-dir")
            .arg(root)
            .env("REUSSIR_CARGO", &cargo)
            .env("REUSSIR_RUSTC", &rustc))
    };

    let out = build(&root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // Stdout is exactly the two `--polyffi-libdir` directories: the detected
    // rust lib, then the freshly baked runtime's deps dir.
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    let deps_dir = root.join("reussir-rt/target/release/deps");
    assert_eq!(
        lines,
        [
            libdir.display().to_string(),
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
    let out = build(&root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), stdout);
    let runs = std::fs::read_to_string(tmp.join("cargo-runs")).unwrap();
    assert_eq!(runs.lines().count(), 1, "the second build re-ran cargo");
}

/// The real thing: bake `reussir-rt` with the host toolchain. Slow (a full
/// release build of the runtime) and network-dependent (crates.io + the
/// pinned mlir-sync git revision), hence ignored by default:
/// `cargo test -p rene -- --ignored`.
#[test]
#[ignore = "builds reussir-rt with the host cargo (slow, needs network)"]
fn build_bakes_the_runtime_with_the_host_toolchain() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("reussir-build");

    let out = run(rene(&["build", "--manifest-path"])
        .arg(demo_manifest())
        .arg("--build-dir")
        .arg(&root));
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
