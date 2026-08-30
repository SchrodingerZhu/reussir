# Builds the reussir-backend Rust crate with cargo, wiring in the LLVM 22 prefix
# that mlir-sys / melior / tblgen require and the staged Reussir static archives.
#
# These targets are intentionally NOT part of the default `ALL` build: they pull
# heavy crates from the network. Invoke them explicitly:
#   ninja -C build reussir-backend         # compile the crate
#   ninja -C build reussir-backend-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_BACKEND_CARGO_PROFILE "dev")
else()
    set(REUSSIR_BACKEND_CARGO_PROFILE "release")
endif()

# mlir-sys / melior / tblgen / llvm-sys discover LLVM 22 via these prefixes and
# link MLIR/LLVM statically (the system development packages, including
# libpolly-<ver>-dev, must be installed).
set(REUSSIR_BACKEND_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    ${REUSSIR_CARGO_CXX_ENV}
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    # melior-macro 0.20 (the `dialect!` macro that generates the Reussir op
    # builders) still discovers llvm-config through the LLVM-21-era variable name
    # even though it targets LLVM 22; point it at the same prefix so its baked
    # include directory is correct.
    MLIR_SYS_210_PREFIX=${REUSSIR_LLVM_PREFIX}
    TABLEGEN_220_PREFIX=${REUSSIR_TABLEGEN_PREFIX}
    # `llvm-sys` 221 (the shared LLVM-IR boundary: translate, the backend LLVM
    # pipeline, polyffi gather/link) discovers LLVM through the 221 variable.
    LLVM_SYS_221_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    REUSSIR_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    RUSTFLAGS=-Awarnings)

add_custom_target(reussir-backend
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_BACKEND_CARGO_ENV}
        ${REUSSIR_CARGO} build ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_BACKEND_CARGO_PROFILE}
        --package reussir-backend --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Building reussir-backend with cargo (${REUSSIR_BACKEND_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-backend-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_BACKEND_CARGO_ENV}
        ${REUSSIR_CARGO} test ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_BACKEND_CARGO_PROFILE}
        --package reussir-backend
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ReussirCAPI MLIRReussir
    COMMENT "Testing reussir-backend with cargo"
    USES_TERMINAL
)
