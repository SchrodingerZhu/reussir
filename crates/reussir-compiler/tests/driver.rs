//! Driver end-to-end tests: run the real `rrc` binary through the stage chain,
//! checking that each input/output pair produces the expected artifact and that
//! the re-entry points (`.mir`/`.mlir`/`.ll` inputs) rejoin the pipeline.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

const PROGRAM: &str = r#"
    enum List<T> { Nil, Cons(T, List<T>) }
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
    assert!(output.stdout.is_empty(), "listing must go to -o, not stdout");
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
