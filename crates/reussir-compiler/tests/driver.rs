//! Driver end-to-end tests: run the real `rrc` binary through the stage chain,
//! checking that each input/output pair produces the expected artifact and that
//! the re-entry points (`.mir`/`.mlir`/`.ll` inputs) rejoin the pipeline.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

const PROGRAM: &str = r#"
    pub enum List<T> { Nil, Cons(T, List<T>) }
    pub fn sum(list: List<i64>) -> i64 {
        match list {
            List::Nil => 0,
            List::Cons(x, xs) => x + sum(xs)
        }
    }
    extern "C" trampoline "sum_ffi" = sum;
"#;

/// A throwaway directory unique to one test, removed when the returned handle
/// drops.
fn scratch(tag: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("rrc-driver-{tag}-"))
        .tempdir()
        .expect("create scratch dir")
}

fn rrc(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rrc"))
        .args(args)
        .output()
        .expect("spawn rrc")
}

/// Compile `input` to `target` (`--emit target`), writing `out`; assert success.
fn emit(input: &Path, target: &str, out: &Path) {
    let output = rrc(&[
        input,
        Path::new("-o"),
        out,
        Path::new("--emit"),
        Path::new(target),
    ]);
    assert!(
        output.status.success(),
        "rrc {} -> {target} failed:\n{}",
        input.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Write `PROGRAM` to a `.rr` file in a fresh scratch dir. The returned
/// [`TempDir`] must be kept alive for the test's duration (it cleans up on drop).
fn source(tag: &str) -> (TempDir, PathBuf) {
    let dir = scratch(tag);
    let src = dir.path().join("prog.rr");
    std::fs::write(&src, PROGRAM).expect("write source");
    (dir, src)
}

#[test]
fn dumps_hir_mir_and_mlir_from_source() {
    let (dir, src) = source("dumps");

    let hir = dir.path().join("prog.hir");
    emit(&src, "hir", &hir);
    let hir_text = read(&hir);
    assert!(hir_text.contains("fn #sum"), "hir:\n{hir_text}");
    assert!(hir_text.contains("match"), "hir:\n{hir_text}");

    let mir = dir.path().join("prog.mir");
    emit(&src, "mir", &mir);
    let mir_text = read(&mir);
    // The MIR is mangled and monomorphized.
    assert!(mir_text.contains("sum"), "mir:\n{mir_text}");
    assert!(
        mir_text.contains("switch") || mir_text.contains("match"),
        "mir:\n{mir_text}"
    );

    let mlir = dir.path().join("prog.mlir");
    emit(&src, "mlir", &mlir);
    let mlir_text = read(&mlir);
    assert!(
        mlir_text.contains("reussir.record.dispatch"),
        "mlir:\n{mlir_text}"
    );
    assert!(mlir_text.contains("func.func"), "mlir:\n{mlir_text}");
}

#[test]
fn dumps_mlir_llvm_and_llvm_ir_from_source() {
    let (dir, src) = source("llvm");

    let mlir_llvm = dir.path().join("prog.mlir-llvm");
    emit(&src, "mlir-llvm", &mlir_llvm);
    let text = read(&mlir_llvm);
    // After the pipeline the module is the LLVM dialect: no reussir ops remain.
    assert!(text.contains("llvm."), "mlir-llvm:\n{text}");
    assert!(
        !text.contains("reussir.record.dispatch"),
        "mlir-llvm still high-level:\n{text}"
    );

    let ll = dir.path().join("prog.ll");
    emit(&src, "llvm-ir", &ll);
    let ll_text = read(&ll);
    assert!(ll_text.contains("define"), "ll:\n{ll_text}");
    // The exported C-ABI trampoline is present.
    assert!(ll_text.contains("sum_ffi"), "ll:\n{ll_text}");
}

#[test]
fn compiles_source_to_object() {
    let (dir, src) = source("obj");
    let obj = dir.path().join("prog.o");
    // Target inferred from the `.o` extension (no --emit).
    let output = rrc(&[&src, Path::new("-o"), &obj]);
    assert!(
        output.status.success(),
        "rrc -> obj failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::metadata(&obj).expect("object exists").len();
    assert!(bytes > 0, "object file is empty");
}

#[test]
fn reenters_from_mir_and_matches_source_lowering() {
    let (dir, src) = source("mir-reentry");

    // Dump MIR from source, then lower that MIR file to MLIR.
    let mir = dir.path().join("prog.mir");
    emit(&src, "mir", &mir);
    let via_mir = dir.path().join("via-mir.mlir");
    emit(&mir, "mlir", &via_mir);

    // Lowering straight from source should reach the same high-level ops.
    let via_src = dir.path().join("via-src.mlir");
    emit(&src, "mlir", &via_src);

    for path in [&via_mir, &via_src] {
        let text = read(path);
        assert!(
            text.contains("reussir.record.dispatch"),
            "{}: {text}",
            path.display()
        );
        assert!(text.contains("sum_ffi"), "{}: {text}", path.display());
    }
}

#[test]
fn mir_and_hir_round_trip_by_reprinting() {
    let (dir, src) = source("round-trip");

    // print(parse(t)) == t for MIR: dumping, re-parsing, and re-dumping is stable.
    let mir1 = dir.path().join("one.mir");
    emit(&src, "mir", &mir1);
    let mir2 = dir.path().join("two.mir");
    emit(&mir1, "mir", &mir2);
    assert_eq!(read(&mir1), read(&mir2), "MIR is not print/parse stable");

    // Same for HIR.
    let hir1 = dir.path().join("one.hir");
    emit(&src, "hir", &hir1);
    let hir2 = dir.path().join("two.hir");
    emit(&hir1, "hir", &hir2);
    assert_eq!(read(&hir1), read(&hir2), "HIR is not print/parse stable");
}

#[test]
fn reenters_from_mlir_and_llvm_ir_to_object() {
    let (dir, src) = source("obj-reentry");

    // .mlir -> .o : parse the dialect, run the pipeline, emit.
    let mlir = dir.path().join("prog.mlir");
    emit(&src, "mlir", &mlir);
    let obj_from_mlir = dir.path().join("from-mlir.o");
    emit(&mlir, "obj", &obj_from_mlir);
    assert!(
        std::fs::metadata(&obj_from_mlir)
            .expect("obj from mlir")
            .len()
            > 0
    );

    // .ll -> .o : parse LLVM IR and run the backend.
    let ll = dir.path().join("prog.ll");
    emit(&src, "llvm-ir", &ll);
    let obj_from_ll = dir.path().join("from-ll.o");
    emit(&ll, "obj", &obj_from_ll);
    assert!(std::fs::metadata(&obj_from_ll).expect("obj from ll").len() > 0);
}

#[test]
fn infers_mlir_llvm_from_the_output_extension() {
    let (dir, src) = source("mlir-llvm-ext");
    // No --emit: the `.mlir-llvm` extension selects the post-pipeline dump.
    let out = dir.path().join("prog.mlir-llvm");
    let output = rrc(&[&src, Path::new("-o"), &out]);
    assert!(
        output.status.success(),
        "rrc -> mlir-llvm (by extension) failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = read(&out);
    assert!(text.contains("llvm."), "mlir-llvm:\n{text}");
    assert!(
        !text.contains("reussir.record.dispatch"),
        "still high-level:\n{text}"
    );
}

#[test]
fn rejects_stdout_for_file_artifacts() {
    let (_dir, src) = source("stdout-obj");
    let output = rrc(&[
        &src,
        Path::new("-o"),
        Path::new("-"),
        Path::new("--emit"),
        Path::new("obj"),
    ]);
    assert!(
        !output.status.success(),
        "expected obj-to-stdout to be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("stdout"), "stderr:\n{stderr}");
}

#[test]
fn rejects_running_the_pipeline_backward() {
    let (dir, _src) = source("backward");
    // A `.mir` input cannot produce HIR (an earlier stage).
    let mir = dir.path().join("prog.mir");
    std::fs::write(&mir, "").expect("write empty mir");
    let output = rrc(&[
        &mir,
        Path::new("-o"),
        &dir.path().join("out.hir"),
        Path::new("--emit"),
        Path::new("hir"),
    ]);
    assert!(
        !output.status.success(),
        "expected backward pipeline to fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("only runs forward"), "stderr:\n{stderr}");
}

#[test]
fn compiles_to_the_wasm_target() {
    // Cross-compile to WebAssembly via `--target-triple`. This exercises the
    // cross-target path (all LLVM targets registered, wasm data layout feeding
    // the pipeline) end to end; the emitted object is a real wasm module even
    // though its runtime symbols stay unresolved until link.
    let (dir, src) = source("wasm");
    let obj = dir.path().join("prog.wasm.o");
    let output = rrc(&[
        &src,
        Path::new("--target-triple"),
        Path::new("wasm32-unknown-unknown"),
        Path::new("-o"),
        &obj,
    ]);
    assert!(
        output.status.success(),
        "wasm compile failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&obj).expect("read wasm object");
    // A WebAssembly object begins with the magic `\0asm` followed by the version.
    assert!(
        bytes.starts_with(b"\0asm"),
        "not a wasm object, leading bytes: {:02x?}",
        &bytes[..bytes.len().min(8)]
    );
}

/// `--scan-deps` describes the package source graph — `lib.rr` plus
/// everything reachable through `mod` declarations — as JSON with canonical
/// paths and module paths in discovery order, without compiling anything.
#[test]
fn scan_deps_lists_the_package_source_graph() {
    let dir = scratch("scan-deps");
    let write = |rel: &str, content: &str| {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        path
    };
    let lib = write("lib.rr", "mod utils;\npub fn e() -> u64 { 0 }");
    let utils = write("utils/mod.rr", "mod math;\npub fn o() -> u64 { 1 }");
    let math = write("utils/math.rr", "pub fn d(x: u64) -> u64 { x }");

    let output = rrc(&[
        Path::new("--package-root"),
        dir.path(),
        Path::new("--package-name"),
        Path::new("mypkg"),
        Path::new("--scan-deps"),
    ]);
    assert!(
        output.status.success(),
        "scan-deps failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Discovery canonicalizes (macOS tempdirs sit behind /var -> /private/var);
    // compare against the canonical form of each expected file.
    let canonical = |path: &Path| {
        path.canonicalize()
            .expect("canonicalize expected path")
            .display()
            .to_string()
    };
    let stdout = String::from_utf8(output.stdout).expect("scan-deps stdout is utf-8");
    let graph: serde_json::Value =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("bad JSON ({e}):\n{stdout}"));
    assert_eq!(
        graph,
        serde_json::json!({
            "package": "mypkg",
            "files": [
                { "path": canonical(&lib), "module": ["mypkg"] },
                { "path": canonical(&utils), "module": ["mypkg", "utils"] },
                { "path": canonical(&math), "module": ["mypkg", "utils", "math"] },
            ],
        }),
        "stdout:\n{stdout}"
    );

    // Rooting the same package at its lib.rr via `--package-name` (no
    // `--package-root`) scans the identical graph.
    let rooted = rrc(&[
        &lib,
        Path::new("--package-name"),
        Path::new("mypkg"),
        Path::new("--scan-deps"),
    ]);
    assert!(
        rooted.status.success(),
        "file-rooted scan-deps failed:\n{}",
        String::from_utf8_lossy(&rooted.stderr)
    );
    assert_eq!(
        String::from_utf8(rooted.stdout).expect("stdout is utf-8"),
        stdout,
        "file-rooted scan must match --package-root scan"
    );
}

/// `-o` redirects the `--scan-deps` listing into a file.
#[test]
fn scan_deps_writes_the_listing_to_the_output_file() {
    let dir = scratch("scan-deps-o");
    std::fs::write(dir.path().join("lib.rr"), "pub fn e() -> u64 { 0 }").unwrap();
    let out = dir.path().join("deps.txt");
    let output = rrc(&[
        Path::new("--package-root"),
        dir.path(),
        Path::new("--package-name"),
        Path::new("p"),
        Path::new("--scan-deps"),
        Path::new("-o"),
        &out,
    ]);
    assert!(
        output.status.success(),
        "scan-deps failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "listing must go to -o, not stdout"
    );
    let graph: serde_json::Value = serde_json::from_str(&read(&out)).expect("bad JSON in -o file");
    let expected = dir.path().join("lib.rr").canonicalize().unwrap();
    assert_eq!(
        graph["files"][0]["path"],
        serde_json::json!(expected.display().to_string())
    );
}

/// Without package mode there is no source graph to scan.
#[test]
fn scan_deps_requires_package_mode() {
    let (_dir, src) = source("scan-deps-nopkg");
    let output = rrc(&[&src, Path::new("--scan-deps")]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "expected a usage error, stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--package-root"), "stderr:\n{stderr}");
}

/// `--emit staticlib` archives the codegen units: one member per unit,
/// whatever the partitioning, and the loose objects never reach the user's
/// output directory.
#[test]
fn archives_the_codegen_units_into_a_static_library() {
    let (dir, src) = source("staticlib");

    for units in [1, 2, 4] {
        let lib = dir.path().join(format!("libprog.{units}.a"));
        let output = rrc(&[
            &src,
            Path::new("-o"),
            &lib,
            Path::new("--emit"),
            Path::new("staticlib"),
            Path::new("--relocation-mode"),
            Path::new("pic"),
            Path::new("--codegen-units"),
            Path::new(&units.to_string()),
        ]);
        assert!(
            output.status.success(),
            "staticlib with {units} units failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let bytes = std::fs::read(&lib).expect("read the archive");
        assert!(
            bytes.starts_with(b"!<arch>\n"),
            "not an archive: {:02x?}",
            &bytes[..bytes.len().min(8)]
        );
        // Members are named after the library and numbered by unit; the
        // scratch directory they were emitted into must not appear.
        let text = String::from_utf8_lossy(&bytes);
        for unit in 0..units {
            assert!(
                text.contains(&format!("libprog.{units}.{unit}.o")),
                "member {unit} of {units} missing from the archive"
            );
        }
        assert!(
            !text.contains("rrc-staticlib-"),
            "the scratch directory leaked into the archive"
        );
    }

    // Nothing but the archives: no stray `.o` siblings of `-o`.
    let strays: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".o"))
        .collect();
    assert!(strays.is_empty(), "left loose objects behind: {strays:?}");
}

/// Under `--lto` the members are bitcode instead of native objects, so the
/// link step can optimize across them.
#[test]
fn a_static_library_under_lto_archives_bitcode() {
    let (dir, src) = source("staticlib-lto");

    for mode in ["thin", "fat"] {
        let lib = dir.path().join(format!("lib{mode}.a"));
        let output = rrc(&[
            &src,
            Path::new("-o"),
            &lib,
            Path::new("--emit"),
            Path::new("staticlib"),
            Path::new("--lto"),
            Path::new(mode),
            Path::new("--relocation-mode"),
            Path::new("pic"),
            Path::new("--codegen-units"),
            Path::new("2"),
        ]);
        assert!(
            output.status.success(),
            "--lto {mode} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let bytes = std::fs::read(&lib).expect("read the archive");
        assert!(bytes.starts_with(b"!<arch>\n"));
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains(&format!("lib{mode}.0.bc")) && text.contains(&format!("lib{mode}.1.bc")),
            "--lto {mode} did not archive bitcode members"
        );
        // The magic of every LLVM bitcode wrapper — the members really are
        // bitcode, not objects wearing a `.bc` name.
        assert!(
            bytes.windows(4).any(|w| w == b"BC\xc0\xde"),
            "no bitcode magic in the --lto {mode} archive"
        );
    }
}

/// The archive is a pure function of its inputs: no timestamps, no scratch
/// paths, nothing that differs between two identical builds.
#[test]
fn a_static_library_is_reproducible() {
    let (dir, src) = source("staticlib-repro");
    let build = |name: &str| -> Vec<u8> {
        let lib = dir.path().join(name);
        let output = rrc(&[
            &src,
            Path::new("-o"),
            &lib,
            Path::new("--emit"),
            Path::new("staticlib"),
            Path::new("--relocation-mode"),
            Path::new("pic"),
            Path::new("--codegen-units"),
            Path::new("2"),
        ]);
        assert!(
            output.status.success(),
            "staticlib failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read(&lib).expect("read the archive")
    };
    // Same name both times: the member names derive from it, so a differing
    // name would differ for an uninteresting reason.
    let first = build("libsame.a");
    let second = build("libsame.a");
    assert_eq!(first, second, "two identical builds differ");
}

/// Like [`rrc`], with the toolchain environment scrubbed, so assertions
/// about a *missing* toolchain hold no matter what the developer's shell
/// exports. (The lit suite covers the links that succeed; these tests cover
/// the driver's own diagnostics, which must not depend on one.)
fn rrc_without_toolchain(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rrc"))
        .args(args)
        .env_remove("REUSSIR_RUSTC")
        .env_remove("REUSSIR_RUSTC_DEPS")
        .output()
        .expect("spawn rrc")
}

/// The link step's driver-level diagnostics, all raised before any linker
/// runs.
#[test]
fn link_step_driver_diagnostics() {
    let (dir, src) = source("link-products");

    // `PROGRAM` has no `#[main]`, so there is no entry point to link an
    // executable around; refused before rustc or the runtime is even looked
    // up, which is why this needs no toolchain.
    let out = dir.path().join("prog.exe");
    let output = rrc(&[
        &src,
        Path::new("-o"),
        &out,
        Path::new("--emit"),
        Path::new("executable"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("an executable needs an entry point"),
        "stderr:\n{stderr}"
    );
    assert!(!out.exists(), "the refused link produced a file anyway");

    // A shared-library extension resolves to the dynlib stage: the driver
    // heads for its link step (and, with the toolchain scrubbed, fails
    // looking for it) instead of emitting the default object.
    let out = dir.path().join("libprog.so");
    let output = rrc_without_toolchain(&[&src, Path::new("-o"), &out]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("for the link step"), "stderr:\n{stderr}");
    assert!(!out.exists(), "the failed link produced a file anyway");

    // The link knobs mean nothing without a link step.
    for (knob, name) in [
        ("--link-arg=-lm", "--link-arg"),
        ("--runtime-linkage=static", "--runtime-linkage"),
        ("--linker=rustc", "--linker"),
    ] {
        let out = dir.path().join("prog.o");
        let output = rrc(&[&src, Path::new("-o"), &out, Path::new(knob)]);
        assert_eq!(output.status.code(), Some(2), "{name} with an object");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(name) && stderr.contains("applies to `executable` and `dynlib`"),
            "stderr:\n{stderr}"
        );
    }
}

/// `--lto` describes how a packaged or linked product is built; it is
/// meaningless for the plain emission targets, and says so rather than
/// silently doing nothing.
#[test]
fn rejects_lto_without_a_packaged_or_linked_target() {
    let (dir, src) = source("lto-misuse");
    let out = dir.path().join("prog.o");
    let output = rrc(&[
        &src,
        Path::new("-o"),
        &out,
        Path::new("--lto"),
        Path::new("fat"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--lto applies to"), "stderr:\n{stderr}");
}
