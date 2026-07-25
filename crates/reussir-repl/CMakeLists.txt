# Builds the reussir-repl Rust crate (the incremental REPL: persistent
# elaborator + ORC JIT) with cargo.
#
# The CMake targets and the produced binary are named `rrepl` (the REPL's
# conventional short name), not `reussir-repl`.
#
# Like reussir-backend / reussir-jit / reussir-codegen, these targets are NOT
# part of the default `ALL` build. Invoke them explicitly:
#   ninja -C build rrepl         # compile the crate + binary
#   ninja -C build rrepl-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_REPL_CARGO_PROFILE "dev")
    set(REUSSIR_REPL_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_REPL_CARGO_PROFILE "release")
    set(REUSSIR_REPL_CARGO_PROFILE_DIR "release")
endif()

# Where cargo puts the binary; exported for the lit `%rrepl` substitution.
set(REUSSIR_RREPL_BINARY
    ${CMAKE_BINARY_DIR}/target/${REUSSIR_REPL_CARGO_PROFILE_DIR}/rrepl${CMAKE_EXECUTABLE_SUFFIX}
    CACHE INTERNAL "Path of the rrepl binary built by cargo")

# The same environment as reussir-jit: the crate pulls in reussir-backend
# (mlir-sys / melior / the staged Reussir archives) and links LLVM via
# llvm-sys 221, all discovered through these prefixes.
set(REUSSIR_REPL_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    ${REUSSIR_CARGO_CXX_ENV}
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_TABLEGEN_PREFIX}
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

add_custom_target(rrepl
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_REPL_CARGO_ENV}
        cargo build --locked --profile ${REUSSIR_REPL_CARGO_PROFILE}
        --package reussir-repl --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building rrepl with cargo (${REUSSIR_REPL_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(rrepl-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_REPL_CARGO_ENV}
        cargo test --locked --profile ${REUSSIR_REPL_CARGO_PROFILE}
        --package reussir-repl
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing rrepl with cargo"
    USES_TERMINAL
)
