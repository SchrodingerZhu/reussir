use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::path::PathBuf;

// Locates the directory holding the staged Reussir static archives
// (libReussirCAPI.a/ReussirCAPI.lib plus the MLIRReussir* and MLIRSync*
// component archives). CMake sets REUSSIR_CAPI_LIB_DIR when it drives the build;
// a direct `cargo` invocation falls back to the in-tree `build/lib` next to the
// workspace root.
fn capi_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("REUSSIR_CAPI_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("../../build/lib")
}

// Component archives the C API references: Reussir's dialect, analyses,
// Rust-to-bitcode compiler, and passes, plus the sync dialect and conversions
// used by lock-guarded cells. They are whole-archived so registrations are
// retained and so the link order among them does not matter; their MLIR/LLVM
// references resolve against mlir-sys' static MLIR/LLVM, which is emitted after
// these.
const REUSSIR_ARCHIVES: &[&str] = &[
    "MLIRReussir",
    "MLIRReussirAnalysis",
    "ReussirRustCompiler",
    "MLIRReussirTypeConverter",
    "MLIRReussirBasicOpsLowering",
    "MLIRReussirConvertToSTD",
    "MLIRReussirRegionPatterns",
    "MLIRReussirAcquireDropExpansion",
    "MLIRReussirRcDecrementExpansion",
    "MLIRReussirTokenInstantiation",
    "MLIRReussirClosureOutlining",
    "MLIRReussirCompilePolymorphicFFI",
    "MLIRReussirAttachNativeTarget",
    "MLIRReussirClosureBetaReduction",
    "MLIRReussirDefaultInliner",
    "MLIRReussirRcCreateSink",
    "MLIRReussirRcCreateFusion",
    "MLIRReussirRcDispatchFusion",
    "MLIRReussirInferVariantTag",
    "MLIRReussirIncDecCancellation",
    "MLIRReussirTokenReuse",
    "MLIRReussirSpecialPointerTag",
    "MLIRReussirInvariantGroupAnalysis",
    "MLIRReussirUniqueCarryingRecursionAnalysis",
    "MLIRReussirTRMCRecursionAnalysis",
    // mlir-sync components referenced by Reussir's type converter,
    // ConvertToSTD pass, and ConvertToLLVM interface.
    "MLIRSync",
    "MLIRSyncTypeConverter",
    "MLIRSyncConvertSyncToSTD",
    "MLIRSyncConvertSyncToLLVM",
    // The custom LLVM passes run by the JIT codegen helper (Jit.cpp).
    "ReussirLLVMAllocationSimplicationPass",
    "ReussirLLVMLinearRecurrencePass",
];

// TPDE static archives (and TPDE's own vendored dependencies: spdlog plus the
// fadec/disarm64 instruction encoders), referenced by Jit.cpp only when TPDE
// support is compiled in (ELF targets). They are linked when present; on
// platforms where TPDE is unavailable the archives are absent and Jit.cpp has no
// references to them, so they are simply skipped.
const TPDE_ARCHIVES: &[&str] = &["tpde_llvm", "tpde", "spdlog", "fadec", "disarm64"];

fn archive_candidates(lib_dir: &Path, archive: &str) -> [PathBuf; 3] {
    [
        lib_dir.join(format!("lib{archive}.a")),
        lib_dir.join(format!("{archive}.lib")),
        lib_dir.join(format!("lib{archive}.lib")),
    ]
}

fn archive_exists(lib_dir: &Path, archive: &str) -> bool {
    archive_candidates(lib_dir, archive)
        .into_iter()
        .any(|path| path.is_file())
}

fn hash_file(hasher: &mut blake3::Hasher, path: &Path) -> io::Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
    }
}

fn native_archive_fingerprint(lib_dir: &Path) -> io::Result<blake3::Hash> {
    let mut hasher = blake3::Hasher::new();
    for archive in std::iter::once("ReussirCAPI")
        .chain(REUSSIR_ARCHIVES.iter().copied())
        .chain(TPDE_ARCHIVES.iter().copied())
    {
        for path in archive_candidates(lib_dir, archive) {
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .expect("archive candidate has a file name")
                .as_encoded_bytes();
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name);
            hasher.update(&path.metadata()?.len().to_le_bytes());
            hash_file(&mut hasher, &path)?;
        }
    }
    Ok(hasher.finalize())
}

fn watch_archive(lib_dir: &Path, archive: &str) {
    for path in archive_candidates(lib_dir, archive) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    let lib_dir = capi_lib_dir();
    let fingerprint = native_archive_fingerprint(&lib_dir)
        .expect("failed to fingerprint native Reussir archives");

    // Cargo notices the archive changes below and invokes rustc again, but a
    // compiler cache can otherwise reuse the old final-link result because the
    // rustc command line only names each archive; it does not contain its
    // contents. Put the contents in rustc's arguments so that cache key changes
    // whenever any linked native archive does.
    println!("cargo:rustc-cfg=reussir_native_archive_fingerprint_{fingerprint}");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // ReussirCAPI (dialect handle + pass factories + helpers) and the Reussir
    // component archives are linked statically into the same binary as mlir-sys'
    // static MLIR, so there is a single MLIR runtime and dialect registry. They
    // are emitted before mlir-sys' MLIR archives so their references resolve.
    println!("cargo:rustc-link-lib=static:+whole-archive=ReussirCAPI");
    for archive in REUSSIR_ARCHIVES {
        println!("cargo:rustc-link-lib=static:+whole-archive={archive}");
    }
    for archive in TPDE_ARCHIVES {
        if archive_exists(&lib_dir, archive) {
            println!("cargo:rustc-link-lib=static:+whole-archive={archive}");
        }
    }

    // Cargo does not track the contents of native static libraries, so without
    // this it keeps reusing a binary linked against an out-of-date archive when
    // only the C++ side (libReussirCAPI.a / the component archives) is rebuilt.
    // Declaring each archive as a build input makes Cargo request a relink when
    // it changes; the fingerprint above makes that request miss compiler caches.
    watch_archive(&lib_dir, "ReussirCAPI");
    for archive in REUSSIR_ARCHIVES {
        watch_archive(&lib_dir, archive);
    }
    for archive in TPDE_ARCHIVES {
        watch_archive(&lib_dir, archive);
    }

    println!("cargo:rerun-if-env-changed=REUSSIR_CAPI_LIB_DIR");
}
