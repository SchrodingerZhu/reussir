# Build the reussir-syntax executable with cargo and stage it next to the
# other tools so the lit test suite can pick it up.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_SYNTAX_CARGO_PROFILE "dev")
    set(REUSSIR_SYNTAX_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_SYNTAX_CARGO_PROFILE "release")
    set(REUSSIR_SYNTAX_CARGO_PROFILE_DIR "release")
endif()

set(REUSSIR_SYNTAX_BIN reussir-syntax${CMAKE_EXECUTABLE_SUFFIX})

add_custom_target(reussir-syntax-build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS=-Awarnings
        cargo build --locked --profile ${REUSSIR_SYNTAX_CARGO_PROFILE} --package reussir-syntax --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building reussir-syntax with cargo (${REUSSIR_SYNTAX_CARGO_PROFILE} profile)"
)

add_custom_target(reussir-syntax-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${REUSSIR_SYNTAX_CARGO_PROFILE_DIR}/${REUSSIR_SYNTAX_BIN}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REUSSIR_SYNTAX_BIN}
    DEPENDS reussir-syntax-build
    COMMENT "Installing reussir-syntax executable"
)

add_custom_target(reussir-syntax DEPENDS reussir-syntax-build reussir-syntax-install)
