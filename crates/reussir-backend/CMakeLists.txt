# Builds the reussir-backend Rust crate with cargo, wiring in the LLVM 22 prefix
# that mlir-sys / melior / tblgen require and the staged Reussir C API libraries.
#
# These targets are intentionally NOT part of the default `ALL` build: they pull
# heavy crates from the network and depend on the fabricated libMLIR-C.so, which
# is only validated on Linux so far. Invoke them explicitly:
#   ninja -C build reussir-backend         # compile the crate
#   ninja -C build reussir-backend-test    # run the crate's tests

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(REUSSIR_BACKEND_CARGO_PROFILE "dev")
else()
    set(REUSSIR_BACKEND_CARGO_PROFILE "release")
endif()

# The distro LLVM 22 packages advertise Polly in `llvm-config --link-static`
# but do not ship libPolly.a / libPollyISL.a. tblgen and llvm-sys bundle static
# LLVM archives at compile time, so provide empty stand-ins on the link path.
set(REUSSIR_LLVM_STUB_DIR ${CMAKE_BINARY_DIR}/llvm-stubs)
add_custom_command(
    OUTPUT ${REUSSIR_LLVM_STUB_DIR}/libPolly.a ${REUSSIR_LLVM_STUB_DIR}/libPollyISL.a
    COMMAND ${CMAKE_COMMAND} -E make_directory ${REUSSIR_LLVM_STUB_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${REUSSIR_LLVM_STUB_DIR}/empty.c
    COMMAND ${CMAKE_C_COMPILER} -c ${REUSSIR_LLVM_STUB_DIR}/empty.c
            -o ${REUSSIR_LLVM_STUB_DIR}/empty.o
    COMMAND ${CMAKE_AR} rcs ${REUSSIR_LLVM_STUB_DIR}/libPolly.a
            ${REUSSIR_LLVM_STUB_DIR}/empty.o
    COMMAND ${CMAKE_AR} rcs ${REUSSIR_LLVM_STUB_DIR}/libPollyISL.a
            ${REUSSIR_LLVM_STUB_DIR}/empty.o
    COMMENT "Creating empty Polly stand-in archives for static LLVM bundling"
    VERBATIM
)
add_custom_target(reussir-llvm-stubs
    DEPENDS ${REUSSIR_LLVM_STUB_DIR}/libPolly.a ${REUSSIR_LLVM_STUB_DIR}/libPollyISL.a)

# Shared environment for every cargo invocation against the backend crate.
set(REUSSIR_BACKEND_CARGO_ENV
    CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
    MLIR_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    MLIR_SYS_LINK_SHARED=1
    TABLEGEN_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    LLVM_SYS_220_PREFIX=${REUSSIR_LLVM_PREFIX}
    REUSSIR_CAPI_LIB_DIR=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    "RUSTFLAGS=-Awarnings -L ${REUSSIR_LLVM_STUB_DIR}")

add_custom_target(reussir-backend
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_BACKEND_CARGO_ENV}
        cargo build --locked --profile ${REUSSIR_BACKEND_CARGO_PROFILE}
        --package reussir-backend --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS reussir-llvm-stubs ReussirMLIRC ReussirCAPI
    COMMENT "Building reussir-backend with cargo (${REUSSIR_BACKEND_CARGO_PROFILE} profile)"
    USES_TERMINAL
)

add_custom_target(reussir-backend-test
    COMMAND ${CMAKE_COMMAND} -E env ${REUSSIR_BACKEND_CARGO_ENV}
        cargo test --locked --profile ${REUSSIR_BACKEND_CARGO_PROFILE}
        --package reussir-backend
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS reussir-llvm-stubs ReussirMLIRC ReussirCAPI
    COMMENT "Testing reussir-backend with cargo"
    USES_TERMINAL
)
