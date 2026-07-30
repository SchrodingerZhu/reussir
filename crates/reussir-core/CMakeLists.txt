# Test target for the reussir-core Rust crate (elaboration, monomorphization,
# ownership analysis). The crate is pure Rust — no LLVM/MLIR linkage — so the
# target needs none of the backend environment. It is NOT part of the default
# `ALL` build; invoke it explicitly:
#   ninja -C build reussir-core-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_CORE_CARGO_PROFILE "dev")
else()
    set(REUSSIR_CORE_CARGO_PROFILE "release")
endif()

add_custom_target(reussir-core-test
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS=-Awarnings
        cargo test --locked --profile ${REUSSIR_CORE_CARGO_PROFILE}
        --package reussir-core
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Testing reussir-core with cargo"
    USES_TERMINAL
)
