fn main() {
    cxx_build::bridges(["src/lib.rs", "src/string_view.rs"])
        .includes(["../../include", "src"])
        .compile("reussir_bridge");
    let dst = cmake::Config::new("../..")
        .define("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        .define("LLVM_USE_LINKER", "lld")
        .build_target("MLIRReussirBridge")
        .build();
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=MLIRReussirBridge");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/string_view.rs");
    println!("cargo:rerun-if-changed=../../include");
    println!("cargo:rerun-if-changed=../../lib");
}
