use std::env;
use std::path::PathBuf;

// Mirrors reussir-backend-sys' search-path resolution: CMake sets
// REUSSIR_CAPI_LIB_DIR, and a direct `cargo` invocation falls back to the
// in-tree `build/lib` next to the workspace root.
fn capi_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("REUSSIR_CAPI_LIB_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../build/lib")
}

fn main() {
    // Embed runtime search paths so this crate's binaries and test executables
    // find the staged Reussir C API (libReussirCAPI.so + the fabricated
    // libMLIR-C.so) and the system libMLIR.so they depend on. `rustc-link-arg`
    // applies to this package's binaries, tests, examples and benches.
    let lib_dir = capi_lib_dir();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    if let Ok(prefix) = env::var("MLIR_SYS_220_PREFIX") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    }

    println!("cargo:rerun-if-env-changed=REUSSIR_CAPI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLIR_SYS_220_PREFIX");
}
