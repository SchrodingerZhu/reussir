# Build the reussir-bytecode-demo executable with cargo and stage it next to the
# other tools so the lit test suite can pick it up. The library itself is built
# as part of the same cargo package.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_BYTECODE_CARGO_PROFILE "dev")
    set(REUSSIR_BYTECODE_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_BYTECODE_CARGO_PROFILE "release")
    set(REUSSIR_BYTECODE_CARGO_PROFILE_DIR "release")
endif()

set(REUSSIR_BYTECODE_BIN reussir-bytecode-demo${CMAKE_EXECUTABLE_SUFFIX})

add_custom_target(reussir-bytecode-demo-build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS=-Awarnings
        cargo build --locked --profile ${REUSSIR_BYTECODE_CARGO_PROFILE} --package reussir-bytecode --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building reussir-bytecode with cargo (${REUSSIR_BYTECODE_CARGO_PROFILE} profile)"
)

add_custom_target(reussir-bytecode-demo-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${REUSSIR_BYTECODE_CARGO_PROFILE_DIR}/${REUSSIR_BYTECODE_BIN}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REUSSIR_BYTECODE_BIN}
    DEPENDS reussir-bytecode-demo-build
    COMMENT "Installing reussir-bytecode-demo executable"
)

add_custom_target(reussir-bytecode-demo DEPENDS reussir-bytecode-demo-build reussir-bytecode-demo-install)
