use std::env;
use std::path::PathBuf;

// Locates the directory holding the staged Reussir static archives
// (libReussirCAPI.a and the libMLIRReussir* component archives). CMake sets
// REUSSIR_CAPI_LIB_DIR when it drives the build; a direct `cargo` invocation
// falls back to the in-tree `build/lib` next to the workspace root.
fn capi_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("REUSSIR_CAPI_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("../../build/lib")
}

// Reussir component archives the C API references: the dialect, analyses, the
// Rust-to-bitcode compiler, and every conversion/transformation pass exposed as
// a factory. They are whole-archived so registrations are retained and so the
// link order among them does not matter; their MLIR/LLVM references resolve
// against mlir-sys' static MLIR/LLVM, which is emitted after these.
const REUSSIR_ARCHIVES: &[&str] = &[
    "MLIRReussir",
    "MLIRReussirAnalysis",
    "ReussirRustCompiler",
    "MLIRReussirTypeConverter",
    "MLIRReussirBasicOpsLowering",
    "MLIRReussirSCFOpsLowering",
    "MLIRReussirRegionPatterns",
    "MLIRReussirAcquireDropExpansion",
    "MLIRReussirRcDecrementExpansion",
    "MLIRReussirTokenInstantiation",
    "MLIRReussirClosureOutlining",
    "MLIRReussirCompilePolymorphicFFI",
    "MLIRReussirAttachNativeTarget",
    "MLIRReussirRcCreateSink",
    "MLIRReussirRcCreateFusion",
    "MLIRReussirInferVariantTag",
    "MLIRReussirIncDecCancellation",
    "MLIRReussirTokenReuse",
    "MLIRReussirInvariantGroupAnalysis",
    "MLIRReussirUniqueCarryingRecursionAnalysis",
    "MLIRReussirTRMCRecursionAnalysis",
];

fn main() {
    let lib_dir = capi_lib_dir();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // ReussirCAPI (dialect handle + pass factories + helpers) and the Reussir
    // component archives are linked statically into the same binary as mlir-sys'
    // static MLIR, so there is a single MLIR runtime and dialect registry. They
    // are emitted before mlir-sys' MLIR archives so their references resolve.
    println!("cargo:rustc-link-lib=static:+whole-archive=ReussirCAPI");
    for archive in REUSSIR_ARCHIVES {
        println!("cargo:rustc-link-lib=static:+whole-archive={archive}");
    }

    // Cargo does not track the contents of native static libraries, so without
    // this it keeps reusing a binary linked against an out-of-date archive when
    // only the C++ side (libReussirCAPI.a / the component archives) is rebuilt.
    // Declaring each archive as a build input forces a relink when it changes.
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libReussirCAPI.a").display()
    );
    for archive in REUSSIR_ARCHIVES {
        println!(
            "cargo:rerun-if-changed={}",
            lib_dir.join(format!("lib{archive}.a")).display()
        );
    }

    println!("cargo:rerun-if-env-changed=REUSSIR_CAPI_LIB_DIR");
}
