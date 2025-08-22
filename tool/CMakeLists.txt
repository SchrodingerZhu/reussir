add_subdirectory(reussir-opt)

# Custom target for rrc Rust tool
# Determine build profile based on CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "dev")
    set(CARGO_PROFILE_DIR "debug")
else()
    set(CARGO_PROFILE "release")
    set(CARGO_PROFILE_DIR "release")
endif()

# Custom target to build rrc with cargo
add_custom_target(rrc-build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS="-Awarnings"
        LLVM_DIR=${LLVM_DIR}
        MLIR_DIR=${MLIR_DIR}
        CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        LLVM_USE_LINKER=${LLVM_USE_LINKER}
        cargo build --profile ${CARGO_PROFILE} --quiet
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rrc
    COMMENT "Building rrc with cargo (${CARGO_PROFILE} profile)"
    DEPENDS MLIRReussirBridge
)

# Custom target to copy and patch the binary
add_custom_target(rrc-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/target/${CARGO_PROFILE_DIR}/rrc
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rrc
    COMMENT "Installing rrc binary"
    DEPENDS rrc-build
)

# Main rrc target that depends on the complete build process
add_custom_target(rrc ALL
    DEPENDS rrc-install
)
