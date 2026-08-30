# reussir-lsp is a pure-Rust stdio language server. It depends on the surface
# parser but not on LLVM/MLIR, so it shares Cargo's workspace target directory
# without inheriting the backend link environment.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_LSP_CARGO_PROFILE "dev")
    set(REUSSIR_LSP_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_LSP_CARGO_PROFILE "release")
    set(REUSSIR_LSP_CARGO_PROFILE_DIR "release")
endif()

set(REUSSIR_LSP_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    RUSTFLAGS=-Awarnings)
set(REUSSIR_LSP_BIN reussir-lsp${CMAKE_EXECUTABLE_SUFFIX})
set(REUSSIR_LSP_BINARY
    ${CMAKE_BINARY_DIR}/target/${REUSSIR_CARGO_TARGET_SUBDIR}${REUSSIR_LSP_CARGO_PROFILE_DIR}/${REUSSIR_LSP_BIN}
    CACHE INTERNAL "Path of the reussir-lsp binary built by cargo")

add_custom_target(reussir-lsp-build
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_LSP_CARGO_ENV}
        ${REUSSIR_CARGO} build ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_LSP_CARGO_PROFILE}
        --package reussir-lsp --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building reussir-lsp with cargo (${REUSSIR_LSP_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-lsp-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${REUSSIR_LSP_BINARY}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REUSSIR_LSP_BIN}
    DEPENDS reussir-lsp-build
    COMMENT "Installing reussir-lsp executable"
)

add_custom_target(reussir-lsp DEPENDS reussir-lsp-build reussir-lsp-install)

add_custom_target(reussir-lsp-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_LSP_CARGO_ENV}
        ${REUSSIR_CARGO} test ${REUSSIR_CARGO_TARGET_ARGS} --locked --profile ${REUSSIR_LSP_CARGO_PROFILE}
        --package reussir-lsp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Testing reussir-lsp with cargo"
    USES_TERMINAL
)
