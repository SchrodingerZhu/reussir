//! End-to-end test of the `rene build` stub against the demo package in
//! `tests/demo-pkg`: locate/evaluate the manifest, print `TODO`, dump the
//! evaluated configuration.

use std::path::Path;
use std::process::Command;

fn demo_manifest() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("demo-pkg")
        .join("rene.ncl")
}

#[test]
fn build_prints_todo_then_dumps_the_config() {
    let out = Command::new(env!("CARGO_BIN_EXE_rene"))
        .args(["build", "--manifest-path"])
        .arg(demo_manifest())
        .output()
        .expect("failed to spawn rene");

    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("TODO"));

    // The rest is the evaluated manifest as JSON.
    let dump: serde_json::Value = serde_json::from_str(&lines.collect::<Vec<_>>().join("\n"))
        .expect("dump is not valid JSON");
    assert_eq!(dump["package"]["name"], "demo");
    assert_eq!(dump["dependencies"]["utils"]["path"], "vendor/utils");
}

#[test]
fn build_locates_the_manifest_from_a_subdirectory() {
    let out = Command::new(env!("CARGO_BIN_EXE_rene"))
        .arg("build")
        .current_dir(demo_manifest().parent().unwrap().join("vendor"))
        .output()
        .expect("failed to spawn rene");

    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    // `vendor/` has no manifest of its own; the walk-up finds the demo
    // package's. (`vendor/utils/` does have one — from there, that one wins.)
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("\"demo\""), "stdout: {stdout}");
}

#[test]
fn build_fails_cleanly_without_a_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_rene"))
        .arg("build")
        .current_dir(tmp.path())
        .output()
        .expect("failed to spawn rene");

    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("rene.ncl"), "stderr: {stderr}");
}
