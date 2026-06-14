use std::env;
use std::path::PathBuf;

// The melior `dialect!` macro that generates the Reussir op builders reads
// `REUSSIR_INCLUDE_DIR` at expansion time to locate the dialect's TableGen
// sources (it is added as a tablegen include directory, so `Reussir/IR/*.td`
// resolves there). CMake sets it when it drives the build; a direct `cargo`
// invocation falls back to the in-tree `include` directory.
//
// We re-export it via `cargo:rustc-env` so the value is present in the compiler
// environment the proc-macro runs in, regardless of how cargo was invoked.
fn main() {
    let include_dir = env::var("REUSSIR_INCLUDE_DIR").unwrap_or_else(|_| {
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        manifest
            .join("../../include")
            .canonicalize()
            .expect("the in-tree Reussir include directory should exist")
            .display()
            .to_string()
    });

    println!("cargo:rustc-env=REUSSIR_INCLUDE_DIR={include_dir}");
    println!("cargo:rerun-if-env-changed=REUSSIR_INCLUDE_DIR");
}
