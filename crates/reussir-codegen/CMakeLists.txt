# Builds the reussir-codegen Rust crate (Full MIR -> MLIR lowering) with cargo.
#
# Like reussir-backend / reussir-jit, these targets are NOT part of the default
# `ALL` build. Invoke them explicitly:
#   ninja -C build reussir-codegen         # compile the crate
#   ninja -C build reussir-codegen-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_CODEGEN_CARGO_PROFILE "dev")
else()
    set(REUSSIR_CODEGEN_CARGO_PROFILE "release")
endif()

# The union of the reussir-backend and reussir-jit environments: the crate pulls
# in reussir-backend (mlir-sys / melior / the `dialect!` macro / staged Reussir
# archives) and, for its end-to-end test, reussir-jit (which links LLVM via
# llvm-sys 221), all discovered through these prefixes.
set(REUSSIR_CODEGEN_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    ${REUSSIR_CARGO_CXX_ENV}
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_TABLEGEN_PREFIX}
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

add_custom_target(reussir-codegen
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_CODEGEN_CARGO_ENV}
        cargo build --locked --profile ${REUSSIR_CODEGEN_CARGO_PROFILE}
        --package reussir-codegen --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building reussir-codegen with cargo (${REUSSIR_CODEGEN_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-codegen-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_CODEGEN_CARGO_ENV}
        cargo test --locked --profile ${REUSSIR_CODEGEN_CARGO_PROFILE}
        --package reussir-codegen
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing reussir-codegen with cargo"
    USES_TERMINAL
)
