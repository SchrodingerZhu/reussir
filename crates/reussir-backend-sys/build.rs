use std::env;
use std::path::PathBuf;

// Locates the directory holding the staged Reussir static archives
// (libReussirCAPI.a and libMLIRReussir.a). CMake sets REUSSIR_CAPI_LIB_DIR when
// it drives the build; a direct `cargo` invocation falls back to the in-tree
// `build/lib` next to the workspace root.
fn capi_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("REUSSIR_CAPI_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("../../build/lib")
}

fn main() {
    let lib_dir = capi_lib_dir();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // The Reussir dialect is linked statically into the same binary as mlir-sys'
    // static MLIR. ReussirCAPI carries the dialect handle; MLIRReussir is whole-
    // archived so the dialect's op/type/attr registrations are retained even
    // though only the handle symbol is referenced directly. Both are emitted
    // before mlir-sys' MLIR archives so their MLIR references resolve.
    println!("cargo:rustc-link-lib=static=ReussirCAPI");
    println!("cargo:rustc-link-lib=static:+whole-archive=MLIRReussir");

    println!("cargo:rerun-if-env-changed=REUSSIR_CAPI_LIB_DIR");
}
