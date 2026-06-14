use std::env;
use std::path::PathBuf;

// Locates the directory holding the staged Reussir C API shared libraries
// (libReussirCAPI.so and the fabricated libMLIR-C.so). CMake sets
// REUSSIR_CAPI_LIB_DIR when it drives the build; a direct `cargo` invocation
// falls back to the in-tree `build/lib` next to the workspace root.
fn capi_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("REUSSIR_CAPI_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("../../build/lib")
}

fn main() {
    let lib_dir = capi_lib_dir();
    // Link directives propagate to dependent crates, so the link search path and
    // the libReussirCAPI dependency are declared here. Runtime search paths
    // (rpath) cannot propagate and are emitted by each final-artifact crate's
    // own build script (see reussir-backend/build.rs).
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ReussirCAPI");

    println!("cargo:rerun-if-env-changed=REUSSIR_CAPI_LIB_DIR");
}
