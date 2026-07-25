# Builds the reussir-compiler Rust crate (the AOT driver: Reussir source ->
# native object / assembly / LLVM IR) with cargo.
#
# The CMake targets and the produced binary are named `rrc` (the driver's
# conventional short name), not `reussir-compiler`.
#
# Like reussir-backend / reussir-jit / reussir-codegen, these targets are NOT
# part of the default `ALL` build. Invoke them explicitly:
#   ninja -C build rrc         # compile the crate + binary
#   ninja -C build rrc-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_COMPILER_CARGO_PROFILE "dev")
    set(REUSSIR_COMPILER_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_COMPILER_CARGO_PROFILE "release")
    set(REUSSIR_COMPILER_CARGO_PROFILE_DIR "release")
endif()

# The same environment as reussir-codegen: the crate pulls in reussir-backend
# (mlir-sys / melior / the staged Reussir archives) and links LLVM via llvm-sys
# 221, all discovered through these prefixes.
set(REUSSIR_COMPILER_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    ${REUSSIR_CARGO_CXX_ENV}
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_TABLEGEN_PREFIX}
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

set(REUSSIR_RRC_BIN rrc${CMAKE_EXECUTABLE_SUFFIX})

add_custom_target(rrc-build
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_COMPILER_CARGO_ENV}
        cargo build --locked --profile ${REUSSIR_COMPILER_CARGO_PROFILE}
        --package reussir-compiler --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building rrc with cargo (${REUSSIR_COMPILER_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

# Stage the built binary next to the other lit tools so `%rrc` resolves it, the
# same way reussir-syntax is installed.
add_custom_target(rrc-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${REUSSIR_COMPILER_CARGO_PROFILE_DIR}/${REUSSIR_RRC_BIN}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REUSSIR_RRC_BIN}
    DEPENDS rrc-build
    COMMENT "Installing rrc executable"
)

add_custom_target(rrc DEPENDS rrc-build rrc-install)

add_custom_target(rrc-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_COMPILER_CARGO_ENV}
        cargo test --locked --profile ${REUSSIR_COMPILER_CARGO_PROFILE}
        --package reussir-compiler
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing rrc with cargo"
    USES_TERMINAL
)
