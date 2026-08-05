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
/// (Unix-gated with its callers, the fake-toolchain tests; on Windows it
/// would be dead code under `-D warnings`.)
#[cfg(unix)]
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
#[cfg(unix)]
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
    assert!(
        stderr(&out).contains("rene.ncl"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn build_help_distinguishes_products_from_the_machine_target() {
    let out = run(&mut rene(&["build", "--help"]));
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let help = String::from_utf8(out.stdout).unwrap();
    assert!(help.contains("--bin <NAME>"), "{help}");
    assert!(help.contains("--lib <NAME>"), "{help}");
    assert!(help.contains("--target <TRIPLE>"), "{help}");
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
                 -vV) printf 'rustc 9.9.9 (fake)\\nhost: x86_64-unknown-fake\\n';;\n\
                 --print) echo '{}';;\n\
                 esac\n",
                libdir.display()
            ),
        );
        // Counts invocations, records the linker rene pinned through cargo's
        // target-specific environment (keyed by the fake host triple above),
        // fabricates the artifacts, and reports them the way
        // `--message-format=json` does. Runs with cwd = the unpacked source
        // dir.
        let cargo = script(
            "fake-cargo",
            "echo run >> \"$(dirname \"$0\")/cargo-runs\"\n\
             echo \"$*\" >> \"$(dirname \"$0\")/cargo-args\"\n\
             echo \"linker=$CARGO_TARGET_X86_64_UNKNOWN_FAKE_LINKER\" >> \"$(dirname \"$0\")/cargo-env\"\n\
             target=\"\"; prev=\"\"\n\
             for arg in \"$@\"; do\n\
               [ \"$prev\" = \"--target\" ] && target=\"$arg\"\n\
               prev=\"$arg\"\n\
             done\n\
             [ -n \"$target\" ] || exit 2\n\
             mkdir -p \"target/$target/release/deps\" target/release/deps\n\
             proc_macro=\"$PWD/target/release/deps/libnum_enum_derive-fake.so\"\n\
             rlib=\"$PWD/target/$target/release/deps/libreussir_rt-0000.rlib\"\n\
             staticlib=\"$PWD/target/$target/release/libreussir_rt.a\"\n\
             touch \"$proc_macro\" \"$rlib\" \"$staticlib\"\n\
             printf '{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"num_enum_derive\"},\"filenames\":[\"%s\"]}\\n' \"$proc_macro\"\n\
             printf '{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"reussir-rt\"},\"filenames\":[\"%s\",\"%s\"]}\\n' \"$rlib\" \"$staticlib\"\n"
                .to_owned(),
        );
        // Stands in for `rrc`. A `--scan-deps` invocation counts itself and
        // reports the package root's `.rr` files as the source graph, crate
        // root first — the same JSON shape rrc emits, so a file added to the
        // package shows up in the next scan. Anything else is a compile:
        // it logs the full argv (plus the REUSSIR_RUSTC rene handed it) and
        // fabricates the `-o` artifact.
        let rrc = script(
            "fake-rrc",
            "root=\"\"; out=\"\"; prev=\"\"; scan=no\n\
             for arg in \"$@\"; do\n\
               [ \"$prev\" = \"--package-root\" ] && root=\"$arg\"\n\
               [ \"$prev\" = \"-o\" ] && out=\"$arg\"\n\
               [ \"$arg\" = \"--scan-deps\" ] && scan=yes\n\
               prev=\"$arg\"\n\
             done\n\
             if [ \"$scan\" = no ]; then\n\
               echo \"$* rustc=$REUSSIR_RUSTC\" >> \"$(dirname \"$0\")/rrc-compiles\"\n\
               echo compiled > \"$out\"\n\
               exit 0\n\
             fi\n\
             echo run >> \"$(dirname \"$0\")/rrc-runs\"\n\
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

    /// The compile invocations the fake `rrc` served, one argv line each.
    fn compiles(&self) -> Vec<String> {
        match std::fs::read_to_string(self.dir.join("rrc-compiles")) {
            Ok(log) => log.lines().map(str::to_owned).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// [`compiles`](Self::compiles) without the implicit `core` package's
    /// invocations — most expectations are about the packages a test
    /// declares, with core's two compiles (interface + archive) asserted
    /// where the injection itself is the subject.
    fn package_compiles(&self) -> Vec<String> {
        self.compiles()
            .into_iter()
            .filter(|l| !l.contains("--package-name core"))
            .collect()
    }

    /// The cargo bake invocations, one argv line each.
    fn cargo_invocations(&self) -> Vec<String> {
        match std::fs::read_to_string(self.dir.join("cargo-args")) {
            Ok(log) => log.lines().map(str::to_owned).collect(),
            Err(_) => Vec::new(),
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

    // Stdout is exactly the `--polyffi-libdir` directories: the detected
    // Rust lib, Cargo's host proc-macro deps, then its target deps.
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    let target = "x86_64-unknown-fake";
    let target_deps = root.join(format!("{target}/reussir-rt/target/{target}/release/deps"));
    let host_deps = root.join(format!("{target}/reussir-rt/target/release/deps"));
    assert_eq!(
        lines,
        [
            fakes.libdir.display().to_string(),
            host_deps.display().to_string(),
            target_deps.display().to_string(),
        ]
    );

    // The bundle really was unpacked, and the bake recorded.
    assert!(root.join(target).join("reussir-rt/Cargo.toml").is_file());
    let dir = BuildDir::open(&root).unwrap();
    assert!(
        dir.status(&tables::rt_source_hash_key(target))
            .unwrap()
            .is_some()
    );
    assert!(
        dir.status(&tables::rt_artifacts_key(target))
            .unwrap()
            .is_some()
    );
    drop(dir);

    // A second build must reuse the bake: same output, no new cargo run.
    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), stdout);
    assert_eq!(fakes.runs("cargo"), 1, "the second build re-ran cargo");
    assert!(fakes.cargo_invocations()[0].contains("--target x86_64-unknown-fake"));

    // A compiler at a different explicit path is a different toolchain
    // selection even when it reports the same version. Do not restore the
    // old cached path and hand that stale value to rrc.
    let alternate_rustc = tmp.join("alternate-rustc");
    std::fs::copy(&fakes.rustc, &alternate_rustc).unwrap();
    let with_alternate = || {
        run(rene(&["build", "--manifest-path"])
            .arg(&manifest)
            .arg("--build-dir")
            .arg(&root)
            .env("REUSSIR_RRC", &fakes.rrc)
            .env("REUSSIR_CARGO", &fakes.cargo)
            .env("REUSSIR_RUSTC", &alternate_rustc))
    };
    let out = with_alternate();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(fakes.runs("cargo"), 2, "the rustc path did not rebake");
    let out = with_alternate();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(fakes.runs("cargo"), 2, "the new rustc path was not cached");
}

/// The profile supplies a default target, the CLI overrides it, and the two
/// target-prefixed runtime bakes remain independently reusable.
#[cfg(unix)]
#[test]
fn build_resolves_the_target_and_keeps_each_runtime_bake() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    std::fs::write(
        &manifest,
        r#"
        {
          package = { name = "demo", version = "0.1.0" },
          targets.demo.kind = 'executable,
          profiles.dev.default_target_triple = "wasm32-unknown-unknown",
        }
        "#,
    )
    .unwrap();
    let fakes = Fakes::new(&tmp);

    let default = fakes.rene(&["build"], &manifest, &root);
    assert!(default.status.success(), "stderr: {}", stderr(&default));
    assert_eq!(
        String::from_utf8(default.stdout).unwrap().trim(),
        root.join("dev/demo.wasm").display().to_string()
    );

    let override_target = "aarch64-unknown-linux-gnu";
    let overridden = fakes.rene(&["build", "--target", override_target], &manifest, &root);
    assert!(
        overridden.status.success(),
        "stderr: {}",
        stderr(&overridden)
    );
    assert_eq!(
        String::from_utf8(overridden.stdout).unwrap().trim(),
        root.join("dev/demo").display().to_string()
    );

    for target in ["wasm32-unknown-unknown", override_target] {
        assert!(root.join(target).join("reussir-rt/Cargo.toml").is_file());
        let dir = BuildDir::open(&root).unwrap();
        assert!(
            dir.status(&tables::rt_artifacts_key(target))
                .unwrap()
                .is_some()
        );
    }
    assert_eq!(
        fakes.cargo_invocations(),
        [
            "build --release --target wasm32-unknown-unknown --no-default-features --message-format=json-render-diagnostics",
            "build --release --target aarch64-unknown-linux-gnu --message-format=json-render-diagnostics",
        ]
    );
    let compiles = fakes.package_compiles();
    assert_eq!(compiles.len(), 2, "{compiles:#?}");
    assert!(
        compiles[0].contains("--target-triple wasm32-unknown-unknown"),
        "{compiles:#?}"
    );
    assert!(
        compiles[1].contains("--target-triple aarch64-unknown-linux-gnu"),
        "{compiles:#?}"
    );

    let inspected = fakes.rene(
        &[
            "inspect",
            "--frozen",
            "--commands",
            "--target",
            override_target,
        ],
        &manifest,
        &root,
    );
    assert!(inspected.status.success(), "stderr: {}", stderr(&inspected));
    let report: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(report["plan"]["target"], override_target);
    assert!(
        report["plan"]["env"]["REUSSIR_RUSTC"]
            .as_str()
            .unwrap()
            .contains("fake-rustc")
    );

    let default_again = fakes.rene(&["build"], &manifest, &root);
    assert!(
        default_again.status.success(),
        "stderr: {}",
        stderr(&default_again)
    );
    assert_eq!(fakes.runs("cargo"), 2, "the default target was rebaked");
    let compiles = fakes.package_compiles();
    assert_eq!(compiles.len(), 3, "{compiles:#?}");
    assert!(
        compiles[2].contains("--target-triple wasm32-unknown-unknown"),
        "{compiles:#?}"
    );
}

#[cfg(unix)]
#[test]
fn build_rejects_a_target_that_is_not_one_safe_triple_component() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    let manifest = package(&tmp, "demo");
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build", "--target", "../escape"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("invalid target triple `../escape`"),
        "stderr: {}",
        stderr(&out)
    );
    assert_eq!(fakes.runs("cargo"), 0);
    assert!(!tmp.join("escape").exists());
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
        // `-v`: the freshness narration lives at DEBUG now.
        let out = fakes.rene(&["-v", "build"], &manifest, &root);
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
    assert_eq!(
        fakes.runs("rrc"),
        2,
        "the changed package must be re-scanned"
    );

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
    assert_eq!(
        fakes.runs("rrc"),
        2,
        "a changed config must force a re-scan"
    );
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

/// (Re)write the scratch package's manifest with targets of every kind and a
/// `release` profile refining the built-in.
#[cfg(unix)]
fn write_targets_manifest(dir: &Path) -> PathBuf {
    let path = dir.join("rene.ncl");
    std::fs::write(
        &path,
        r#"
        {
          package = { name = "demo", version = "0.1.0" },
          targets = {
            demo = { kind = 'executable },
            shared = { kind = 'dynlib },
            archive = { kind = 'staticlib },
          },
          profiles = {
            release = { lto = "thin", codegen_units = 2, link_args = ["-lm"] },
          },
        }
        "#,
    )
    .unwrap();
    path
}

/// The whole flag surface, through the fakes: every declared target compiles
/// with the profile's knobs spelled as rrc flags, the link-only knobs go to
/// the linked kinds alone, artifacts land under `<build-dir>/<profile>/`,
/// and stdout lists them in declaration order.
#[cfg(unix)]
#[test]
fn build_compiles_the_declared_targets_with_the_profile_knobs() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    package(&tmp, "demo");
    let manifest = write_targets_manifest(&tmp);
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build", "--profile", "release"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // Stdout lists the artifacts in BTreeMap (declaration-name) order. The
    // fake rustc reports `x86_64-unknown-fake`, an ELF-like target regardless
    // of the platform running this test.
    let stdout = String::from_utf8(out.stdout).unwrap();
    let release = root.join("release");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            release.join("libarchive.a").display().to_string(),
            release.join("demo").display().to_string(),
            release.join("libshared.so").display().to_string(),
        ]
    );
    for line in stdout.lines() {
        assert!(Path::new(line).is_file(), "no artifact at {line}");
    }

    // The implicit core built once, ahead of everything: its interface and
    // archive, each with the reserved name lifted.
    let core: Vec<String> = fakes
        .compiles()
        .into_iter()
        .filter(|l| l.contains("--package-name core"))
        .collect();
    assert_eq!(core.len(), 2, "{core:#?}");
    assert!(core.iter().all(|l| l.contains("--core")), "{core:#?}");

    let compiles = fakes.package_compiles();
    assert_eq!(compiles.len(), 3, "{compiles:#?}");
    for line in &compiles {
        // The package, the profile's codegen knobs (the built-in's opt
        // refined by the manifest's lto/units), the PIC default, the baked
        // libdirs, and the toolchain rene pinned for the child.
        assert!(line.contains("--package-name demo"), "{line}");
        assert!(line.contains("-O aggressive"), "{line}");
        assert!(line.contains("--lto thin"), "{line}");
        assert!(line.contains("--codegen-units 2"), "{line}");
        assert!(line.contains("--relocation-mode pic"), "{line}");
        assert!(
            line.contains("--target-triple x86_64-unknown-fake"),
            "{line}"
        );
        assert!(line.contains("--polyffi-libdir"), "{line}");
        assert!(
            line.contains(&format!("rustc={}", fakes.rustc.display())),
            "{line}"
        );
    }
    let of_kind = |kind: &str| {
        compiles
            .iter()
            .find(|l| l.contains(&format!("--emit {kind}")))
            .unwrap_or_else(|| panic!("no {kind} compile in {compiles:#?}"))
    };
    // Link-only knobs reach the linked kinds and never the archive.
    assert!(of_kind("executable").contains("--link-arg=-lm"));
    assert!(of_kind("dynlib").contains("--link-arg=-lm"));
    assert!(!of_kind("staticlib").contains("--link-arg"));

    // A second build finds every product current: no new compiles, same
    // listing.
    let again = fakes.rene(&["build", "--profile", "release"], &manifest, &root);
    assert!(again.status.success(), "stderr: {}", stderr(&again));
    assert_eq!(String::from_utf8(again.stdout).unwrap(), stdout);
    assert_eq!(
        fakes.package_compiles().len(),
        3,
        "a fresh build re-compiled"
    );

    // A source edit re-fingerprints every product of the profile.
    std::fs::write(
        tmp.join("src/math.rr"),
        "pub fn add(a: u64, b: u64) -> u64 { b + a }\n",
    )
    .unwrap();
    let rebuilt = fakes.rene(&["build", "--profile", "release"], &manifest, &root);
    assert!(rebuilt.status.success(), "stderr: {}", stderr(&rebuilt));
    assert_eq!(fakes.package_compiles().len(), 6, "the edit must rebuild");
}

/// `--linker` pins one linker end to end: cargo's target-specific
/// environment for the runtime bake, and `rrc --linker` for the driver-level
/// links — the Windows story, where rustc's own discovery under a vcvars
/// shell resolves coreutils' `link` instead of MSVC's.
#[cfg(unix)]
#[test]
fn build_pins_the_linker_through_the_bake_and_the_compiles() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    package(&tmp, "demo");
    let manifest = write_targets_manifest(&tmp);
    let fakes = Fakes::new(&tmp);
    // Any existing file serves as "the linker" — nothing executes it here.
    let linker = fakes.rustc.display().to_string();

    let out = fakes.rene(&["build", "--linker", &linker], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // The bake saw it through CARGO_TARGET_<TARGET>_LINKER…
    let cargo_env = std::fs::read_to_string(tmp.join("cargo-env")).unwrap();
    assert_eq!(cargo_env.trim(), format!("linker={linker}"));

    // …and every linked compile got `--linker`; the archive did not.
    let compiles = fakes.package_compiles();
    assert_eq!(compiles.len(), 3, "{compiles:#?}");
    for line in &compiles {
        let linked = !line.contains("--emit staticlib");
        assert_eq!(
            line.contains(&format!("--linker {linker}")),
            linked,
            "{line}"
        );
    }
}

/// `--bin` and `--lib` narrow the build by target kind; unknown names,
/// kind mismatches, and unknown profiles are refused with their names.
#[cfg(unix)]
#[test]
fn build_selects_bins_and_libs_and_refuses_invalid_names() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    package(&tmp, "demo");
    let manifest = write_targets_manifest(&tmp);
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build", "--bin", "demo"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [root.join("dev").join("demo").display().to_string()]
    );
    let compiles = fakes.package_compiles();
    assert_eq!(compiles.len(), 1, "{compiles:#?}");
    // The dev built-in: unoptimized, debug info on.
    assert!(compiles[0].contains("-O none"), "{}", compiles[0]);
    assert!(compiles[0].contains("-g"), "{}", compiles[0]);

    let out = fakes.rene(
        &["build", "--lib", "shared", "--lib", "archive"],
        &manifest,
        &root,
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            root.join("dev/libshared.so").display().to_string(),
            root.join("dev").join("libarchive.a").display().to_string(),
        ]
    );
    assert_eq!(fakes.package_compiles().len(), 3);

    let out = fakes.rene(&["build", "--bin", "bogus"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("no binary target named `bogus`"),
        "stderr: {}",
        stderr(&out)
    );

    let out = fakes.rene(&["build", "--lib", "bogus"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("no library target named `bogus`"),
        "stderr: {}",
        stderr(&out)
    );

    let out = fakes.rene(&["build", "--bin", "shared"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("select it with `--lib shared`"),
        "stderr: {}",
        stderr(&out)
    );

    let out = fakes.rene(&["build", "--lib", "demo"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("select it with `--bin demo`"),
        "stderr: {}",
        stderr(&out)
    );

    let out = fakes.rene(&["build", "--profile", "bogus"], &manifest, &root);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("no profile named `bogus`"),
        "stderr: {}",
        stderr(&out)
    );
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

/// `-j`/`--jobs` caps the compile-process pool. `-j 1` serializes the pool
/// entirely; the build must still produce every artifact.
#[cfg(unix)]
#[test]
fn build_honors_a_single_job_cap() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");
    package(&tmp, "demo");
    let manifest = write_targets_manifest(&tmp);
    let fakes = Fakes::new(&tmp);

    let out = fakes.rene(&["build", "--jobs", "1"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 3, "stdout: {stdout}");
    for line in stdout.lines() {
        assert!(Path::new(line).is_file(), "no artifact at {line}");
    }
    assert_eq!(fakes.package_compiles().len(), 3);
}

/// The pool needs at least one process: `-j 0` is a usage error (exit 2),
/// reported by the parser before any build state is touched.
#[test]
fn build_rejects_a_zero_jobs_cap() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");

    let out = run(rene(&["build", "-j", "0"])
        .arg("--manifest-path")
        .arg(tmp.join("rene.ncl"))
        .arg("--build-dir")
        .arg(&root));
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    assert!(stderr(&out).contains("--jobs"), "stderr: {}", stderr(&out));
    assert!(!root.exists(), "usage error must not create the build dir");
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
    assert_eq!(libdirs.len(), 3, "stdout: {stdout}");
    // The detected rust libdir holds the standard library…
    assert!(libdirs[0].is_dir());
    // …while explicit `--target` keeps host proc macros and target libraries
    // in separate Cargo deps directories. One holds the runtime rlib polyffi
    // links against; ordering varies with the target triple's spelling.
    assert!(libdirs[1..].iter().all(|dir| dir.is_dir()));
    assert!(
        libdirs[1..].iter().any(|dir| {
            std::fs::read_dir(dir)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("libreussir_rt")
                })
        }),
        "no libreussir_rt in {:#?}",
        &libdirs[1..]
    );
}

/// The cross-package pipeline under the fake toolchain: the dependency's
/// interface and archive compile before the root's targets, the root sees
/// the cone (`--extern`/`--extern-src`) and links the archive
/// (`--link-lib`), records make the second build a no-op, and a touched
/// dependency source rebuilds it.
#[cfg(unix)]
#[test]
fn build_runs_the_dependency_pipeline() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp = tmp_dir.path().canonicalize().unwrap();
    let root = tmp.join("reussir-build");

    // The root package, depending on `util` by path.
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    std::fs::write(
        tmp.join("src/lib.rr"),
        "pub fn entry(n: u64) -> u64 { n }\n",
    )
    .unwrap();
    let manifest = tmp.join("rene.ncl");
    std::fs::write(
        &manifest,
        "{ package = { name = \"app\", version = \"0.1.0\" },\n\
         \x20 dependencies.util = { path = \"vendor/util\", version = \"^1.0\" },\n\
         \x20 targets.demo = { kind = 'executable } }\n",
    )
    .unwrap();
    let util = tmp.join("vendor/util");
    std::fs::create_dir_all(util.join("src")).unwrap();
    std::fs::write(
        util.join("src/lib.rr"),
        "pub fn twice(n: u64) -> u64 { n + n }\n",
    )
    .unwrap();
    std::fs::write(
        util.join("rene.ncl"),
        "{ package = { name = \"util\", version = \"1.2.0\" } }\n",
    )
    .unwrap();

    let fakes = Fakes::new(&tmp);
    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let compiles = fakes.package_compiles();
    // Dependency interface, dependency archive, root target — in order.
    assert_eq!(compiles.len(), 3, "{compiles:#?}");
    assert!(compiles[0].contains("--package-name util"), "{compiles:#?}");
    assert!(compiles[0].contains("--emit rri"), "{compiles:#?}");
    assert!(compiles[1].contains("--package-name util"), "{compiles:#?}");
    assert!(compiles[1].contains("--emit staticlib"), "{compiles:#?}");
    assert!(compiles[2].contains("--package-name app"), "{compiles:#?}");
    assert!(compiles[2].contains("--emit executable"), "{compiles:#?}");
    // The root sees the dependency's interface and sources, and links its
    // archive.
    let deps_dir = root.join("dev").join("deps");
    assert!(
        compiles[2].contains(&format!(
            "--extern util={}",
            deps_dir.join("util.rri").display()
        )),
        "{compiles:#?}"
    );
    assert!(
        compiles[2].contains(&format!("--extern-src util={}", util.join("src").display())),
        "{compiles:#?}"
    );
    assert!(
        compiles[2].contains(&format!(
            "--link-lib {}",
            deps_dir.join("libutil.a").display()
        )),
        "{compiles:#?}"
    );
    // The dependency's compiles never see link flags or their own extern
    // (the implicit `core` is the one interface every compile gets).
    assert!(!compiles[0].contains("--link-lib"), "{compiles:#?}");
    assert!(!compiles[1].contains("--extern util"), "{compiles:#?}");

    // Everything is recorded: the second build compiles nothing.
    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(
        fakes.package_compiles().len(),
        3,
        "stderr: {}\n{:#?}",
        stderr(&out),
        fakes.package_compiles()
    );

    // A touched dependency source re-runs its two compiles. The root's
    // target does NOT re-run: the fake fabricates byte-identical artifacts,
    // and the root's fingerprint hashes upstream *content* — the early
    // cutoff, demonstrated. (With a real compiler the archive's bytes would
    // change and the root would relink.)
    std::fs::write(
        util.join("src/lib.rr"),
        "pub fn twice(n: u64) -> u64 { n * 2 }\n",
    )
    .unwrap();
    let out = fakes.rene(&["build"], &manifest, &root);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let after = fakes.package_compiles();
    assert_eq!(after.len(), 5, "{after:#?}");
    assert!(after[3].contains("--emit rri"), "{after:#?}");
    assert!(after[4].contains("--emit staticlib"), "{after:#?}");
    assert!(
        after[3..]
            .iter()
            .all(|line| !line.contains("--package-name app")),
        "the root re-ran despite identical upstream bytes: {after:#?}"
    );
}

/// `rene new` end to end: the flags land in the scaffold, the manifest the
/// binary writes really evaluates, and stdout stays silent (it is reserved
/// for machine-readable listings).
#[test]
fn new_scaffolds_a_package_the_loader_accepts() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("hello");
    let out = run(rene(&["new", "--vcs", "none", "--lib"]).arg(&dir));
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(out.stdout.is_empty(), "stdout is for machine output");

    let loaded = rene::manifest::load(&dir.join("rene.ncl")).unwrap();
    assert_eq!(loaded.manifest.package.name, "hello");
    assert_eq!(
        loaded.manifest.targets["hello"].kind,
        rene::manifest::TargetKind::Dynlib
    );
    assert!(!dir.join(".gitignore").exists(), "--vcs none writes none");

    // Occupied now: a rerun must refuse rather than overwrite.
    let out = run(rene(&["new", "--vcs", "none"]).arg(&dir));
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("not empty"),
        "stderr: {}",
        stderr(&out)
    );
}

/// Interactive mode reads the answers from stdin; typed answers override
/// the flag-provided defaults, and empty ones keep them.
#[test]
fn new_interactive_answers_override_the_flag_defaults() {
    use std::io::Write as _;
    use std::process::Stdio;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("scaffold");
    // name: typed; version: kept; executable: declined (flag default was
    // yes); dynlib: accepted; staticlib: kept (no); triple: typed; git:
    // declined.
    let answers = "calc\n\nn\ny\n\nwasm32-wasip1\nn\n";
    let mut child = rene(&["new", "-i"])
        .arg(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(answers.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let manifest = rene::manifest::load(&dir.join("rene.ncl"))
        .unwrap()
        .manifest;
    assert_eq!(manifest.package.name, "calc");
    assert_eq!(manifest.package.version.as_deref(), Some("0.1.0"));
    assert_eq!(manifest.targets.len(), 1, "{:?}", manifest.targets);
    assert_eq!(
        manifest.targets["calc"].kind,
        rene::manifest::TargetKind::Dynlib
    );
    let dev = rene::manifest::resolve_profile(&manifest, "dev").unwrap();
    assert_eq!(dev.default_target_triple.as_deref(), Some("wasm32-wasip1"));
    assert!(!dir.join(".git").exists(), "git was declined");
}

/// The git default: a repository outside any work tree gets its own, a
/// scaffold nested inside one only gets the ignore file.
#[test]
fn new_inits_git_once_and_never_nested() {
    if !std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
    {
        eprintln!("skipping: no git on PATH");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let outer = tmp.path().join("outer");
    let out = run(rene(&["new"]).arg(&outer));
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(outer.join(".git").exists());
    let ignore = std::fs::read_to_string(outer.join(".gitignore")).unwrap();
    assert!(ignore.contains("/reussir-build/"), "{ignore}");

    let nested = outer.join("vendor").join("inner");
    let out = run(rene(&["new"]).arg(&nested));
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        !nested.join(".git").exists(),
        "must not nest a repository inside `outer`'s"
    );
    assert!(nested.join(".gitignore").exists());
}
