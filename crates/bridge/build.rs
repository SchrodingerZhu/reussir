fn main() {
    cxx_build::bridges(["src/lib.rs", "src/string_view.rs"])
        .includes(["../../include", "src"])
        .compile("reussir_bridge");
    let mut config = cmake::Config::new("../..");
    if let Ok(llvm_dir) = std::env::var("LLVM_DIR") {
        config.define("LLVM_DIR", llvm_dir);
    }
    if let Ok(mlir_dir) = std::env::var("MLIR_DIR") {
        config.define("MLIR_DIR", mlir_dir);
    }
    if let Ok(cmake_cxx_compiler) = std::env::var("CMAKE_CXX_COMPILER") {
        config.define("CMAKE_CXX_COMPILER", cmake_cxx_compiler);
    }
    if let Ok(cmake_c_compiler) = std::env::var("CMAKE_C_COMPILER") {
        config.define("CMAKE_C_COMPILER", cmake_c_compiler);
    }
    if let Ok(llvm_use_linker) = std::env::var("LLVM_USE_LINKER") {
        config.define("LLVM_USE_LINKER", llvm_use_linker);
    }
    let dst = config.build_target("MLIRReussirBridge").build();
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=MLIRReussirBridge");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/string_view.rs");
    println!("cargo:rerun-if-changed=../../include");
    println!("cargo:rerun-if-changed=../../lib");
}
