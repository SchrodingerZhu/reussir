# Builds the rene Rust crate (the package manager) with cargo.
#
# Unlike the other Rust crates here, rene links no LLVM/MLIR: it drives `rrc`
# as a subprocess and bakes the bundled reussir-rt with the user's own
# toolchain. Its cargo environment is correspondingly bare — only the shared
# target directory, so it reuses the workspace's build artifacts.
#
# These targets are NOT part of the default `ALL` build (except the install
# step, which stages the binary for the lit suite). Invoke them explicitly:
#   ninja -C build rene         # compile the crate + binary
#   ninja -C build rene-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RENE_CARGO_PROFILE "dev")
    set(RENE_CARGO_PROFILE_DIR "debug")
else()
    set(RENE_CARGO_PROFILE "release")
    set(RENE_CARGO_PROFILE_DIR "release")
endif()

set(RENE_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    RUSTFLAGS=-Awarnings)

set(RENE_BIN rene${CMAKE_EXECUTABLE_SUFFIX})

add_custom_target(rene-build
    COMMAND ${CMAKE_COMMAND} -E env ${RENE_CARGO_ENV}
        cargo build --locked --profile ${RENE_CARGO_PROFILE}
        --package rene --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building rene with cargo (${RENE_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

# rene's build script packs the reussir-rt source tree into the binary, and
# its cargo build shares ${CMAKE_BINARY_DIR}/target with the runtime's own.
# Order the two so they never race that directory (the same reason rrepl
# carries this dependency).
add_dependencies(rene-build reussir-rt)

# Stage the binary next to the other lit tools so `%rene` resolves it.
add_custom_target(rene-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${RENE_CARGO_PROFILE_DIR}/${RENE_BIN}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${RENE_BIN}
    DEPENDS rene-build
    COMMENT "Installing rene executable"
)

add_custom_target(rene DEPENDS rene-build rene-install)

add_custom_target(rene-test
    COMMAND ${CMAKE_COMMAND} -E env ${RENE_CARGO_ENV}
        cargo test --locked --profile ${RENE_CARGO_PROFILE}
        --package rene
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Testing rene with cargo"
    USES_TERMINAL
)
