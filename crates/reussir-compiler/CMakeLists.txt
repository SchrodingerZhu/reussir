# Builds the reussir-compiler Rust crate (the AOT driver: Reussir source ->
# native object / assembly / LLVM IR) with cargo.
#
# Like reussir-backend / reussir-jit / reussir-codegen, these targets are NOT
# part of the default `ALL` build. Invoke them explicitly:
#   ninja -C build reussir-compiler         # compile the crate + binary
#   ninja -C build reussir-compiler-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_COMPILER_CARGO_PROFILE "dev")
else()
    set(REUSSIR_COMPILER_CARGO_PROFILE "release")
endif()

# The same environment as reussir-codegen: the crate pulls in reussir-backend
# (mlir-sys / melior / the staged Reussir archives) and links LLVM via llvm-sys
# 221, all discovered through these prefixes.
set(REUSSIR_COMPILER_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    LLVM_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

add_custom_target(reussir-compiler
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_COMPILER_CARGO_ENV}
        cargo build --locked --profile ${REUSSIR_COMPILER_CARGO_PROFILE}
        --package reussir-compiler --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building reussir-compiler with cargo (${REUSSIR_COMPILER_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-compiler-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_COMPILER_CARGO_ENV}
        cargo test --locked --profile ${REUSSIR_COMPILER_CARGO_PROFILE}
        --package reussir-compiler
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing reussir-compiler with cargo"
    USES_TERMINAL
)
