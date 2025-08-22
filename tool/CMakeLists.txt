add_subdirectory(reussir-opt)

# Custom target for rrc Rust tool
# Determine build profile based on CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "dev")
else()
    set(CARGO_PROFILE "release")
endif()

# Find patchelf
find_program(PATCHELF_EXECUTABLE patchelf)
if(NOT PATCHELF_EXECUTABLE)
    message(WARNING "patchelf not found, rpath patching will be skipped")
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
        ${CMAKE_BINARY_DIR}/target/${CARGO_PROFILE}/rrc
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rrc
    DEPENDS rrc-build
)

# Patch rpath if patchelf is available
if(PATCHELF_EXECUTABLE)
    add_custom_target(rrc-patch-rpath ALL
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/rrc/fix_rpath.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rrc
        DEPENDS rrc-install
    )
else()
    # Create a dummy target if patchelf is not available
    add_custom_target(rrc-patch-rpath ALL
        DEPENDS rrc-install
    )
endif()

# Main rrc target that depends on the complete build process
add_custom_target(rrc ALL
    DEPENDS rrc-patch-rpath
)
