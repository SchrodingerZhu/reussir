# Determine build profile based on CMAKE_BUILD_TYPE.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_SYNTAX_CARGO_PROFILE "dev")
    set(REUSSIR_SYNTAX_CARGO_PROFILE_DIR "debug")
else()
    set(REUSSIR_SYNTAX_CARGO_PROFILE "release")
    set(REUSSIR_SYNTAX_CARGO_PROFILE_DIR "release")
endif()

if(WIN32)
    set(REUSSIR_SYNTAX_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(REUSSIR_SYNTAX_SHARED_LIB reussir_syntax.dll)
elseif(APPLE)
    set(REUSSIR_SYNTAX_OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(REUSSIR_SYNTAX_SHARED_LIB libreussir_syntax.dylib)
else()
    set(REUSSIR_SYNTAX_OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(REUSSIR_SYNTAX_SHARED_LIB libreussir_syntax.so)
endif()

add_custom_target(reussir-syntax-build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS=-Awarnings
        cargo build --locked --profile ${REUSSIR_SYNTAX_CARGO_PROFILE} --package reussir-syntax --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building reussir-syntax library with cargo (${REUSSIR_SYNTAX_CARGO_PROFILE} profile)"
)

add_custom_target(reussir-syntax-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${REUSSIR_SYNTAX_CARGO_PROFILE_DIR}/${REUSSIR_SYNTAX_SHARED_LIB}
        ${REUSSIR_SYNTAX_OUTPUT_DIR}/${REUSSIR_SYNTAX_SHARED_LIB}
    COMMENT "Installing reussir-syntax shared library"
    DEPENDS reussir-syntax-build
)
