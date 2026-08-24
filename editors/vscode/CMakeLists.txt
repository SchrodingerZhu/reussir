# The VS Code extension is intentionally not part of the default `all` target:
# consumers building only the compiler should not need Node.js. Explicit
# targets provide a reproducible Rust/Wasm build, protocol smoke test, and VSIX.

find_program(REUSSIR_NPM_EXECUTABLE NAMES npm npm.cmd)
if(NOT REUSSIR_NPM_EXECUTABLE)
    message(STATUS "npm not found; Reussir VS Code extension targets are disabled")
    return()
endif()

set(REUSSIR_VSCODE_NPM_STAMP ${CMAKE_CURRENT_BINARY_DIR}/npm-ci.stamp)
set(REUSSIR_VSCODE_VSIX ${CMAKE_BINARY_DIR}/reussir-vscode.vsix)
set(REUSSIR_VSCODE_CARGO_ENV CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target)

add_custom_command(
    OUTPUT ${REUSSIR_VSCODE_NPM_STAMP}
    COMMAND ${REUSSIR_NPM_EXECUTABLE} ci
    COMMAND ${CMAKE_COMMAND} -E touch ${REUSSIR_VSCODE_NPM_STAMP}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/package.json
        ${CMAKE_CURRENT_SOURCE_DIR}/package-lock.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing locked VS Code extension build dependencies"
    VERBATIM
)

add_custom_target(reussir-vscode-deps DEPENDS ${REUSSIR_VSCODE_NPM_STAMP})

add_custom_target(reussir-vscode-wasm
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_VSCODE_CARGO_ENV}
        ${REUSSIR_NPM_EXECUTABLE} run build:wasm
    DEPENDS reussir-vscode-deps
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the Rust/Wasm VS Code client"
    USES_TERMINAL
)

add_custom_target(reussir-vscode-build
    COMMAND ${REUSSIR_NPM_EXECUTABLE} run check
    COMMAND ${REUSSIR_NPM_EXECUTABLE} run compile
    DEPENDS reussir-vscode-wasm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Type-checking and bundling the Reussir VS Code extension"
    USES_TERMINAL
)

add_custom_target(reussir-vscode DEPENDS reussir-vscode-build)

add_custom_target(reussir-vscode-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_VSCODE_CARGO_ENV}
        cargo test --locked --manifest-path ${CMAKE_SOURCE_DIR}/Cargo.toml
        --package reussir-vscode-wasm
    COMMAND ${CMAKE_COMMAND} -E env
        REUSSIR_LSP_PATH=${REUSSIR_LSP_BINARY}
        ${REUSSIR_NPM_EXECUTABLE} run test:protocol
    DEPENDS reussir-vscode-build reussir-lsp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Testing the Rust/Wasm client against reussir-lsp"
    USES_TERMINAL
)

add_custom_target(reussir-vscode-package
    COMMAND ${CMAKE_COMMAND} -E env
        ${REUSSIR_VSCODE_CARGO_ENV}
        REUSSIR_VSIX_OUT=${REUSSIR_VSCODE_VSIX}
        ${REUSSIR_NPM_EXECUTABLE} run package:vsix
    DEPENDS reussir-vscode-deps
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Packaging ${REUSSIR_VSCODE_VSIX}"
    USES_TERMINAL
)
