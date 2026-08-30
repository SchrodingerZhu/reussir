# Builds the reussir-jit Rust crate (the JIT execution engine) with cargo.
#
# Like reussir-backend, these targets are NOT part of the default `ALL` build.
# Invoke them explicitly:
#   ninja -C build reussir-jit         # compile the crate
#   ninja -C build reussir-jit-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_JIT_CARGO_PROFILE "dev")
else()
    set(REUSSIR_JIT_CARGO_PROFILE "release")
endif()

# Same toolchain/env as reussir-backend: the tests pull in reussir-backend (and
# thus mlir-sys / melior / the staged Reussir archives), and the crate links LLVM
# via llvm-sys, all discovered through these prefixes.
set(REUSSIR_JIT_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    ${REUSSIR_CARGO_CXX_ENV}
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_TABLEGEN_PREFIX}
    # llvm-sys 221 (LLVM 22.1) discovers llvm-config through this prefix.
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

add_custom_target(reussir-jit
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_JIT_CARGO_ENV}
        ${REUSSIR_CARGO} build ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_JIT_CARGO_PROFILE}
        --package reussir-jit --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building reussir-jit with cargo (${REUSSIR_JIT_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-jit-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_JIT_CARGO_ENV}
        ${REUSSIR_CARGO} test ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_JIT_CARGO_PROFILE}
        --package reussir-jit
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing reussir-jit with cargo"
    USES_TERMINAL
)
