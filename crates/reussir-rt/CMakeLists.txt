# Determine build profile based on CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "dev")
    set(CARGO_PROFILE_DIR "debug")
else()
    set(CARGO_PROFILE "release")
    set(CARGO_PROFILE_DIR "release")
endif()

if(WIN32)
    set(REUSSIR_RT_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(REUSSIR_RT_SHARED_LIB reussir_rt.dll)
    set(REUSSIR_RT_SHARED_PREFIX reussir_rt)
    set(REUSSIR_RT_SHARED_EXT dll)
elseif(APPLE)
    set(REUSSIR_RT_OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(REUSSIR_RT_SHARED_LIB libreussir_rt.dylib)
    set(REUSSIR_RT_SHARED_PREFIX libreussir_rt)
    set(REUSSIR_RT_SHARED_EXT dylib)
else()
    set(REUSSIR_RT_OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(REUSSIR_RT_SHARED_LIB libreussir_rt.so)
    set(REUSSIR_RT_SHARED_PREFIX libreussir_rt)
    set(REUSSIR_RT_SHARED_EXT so)
endif()

# Custom target to build the default runtime with cargo.
add_custom_target(reussir-rt-build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target
        RUSTFLAGS=-Awarnings
        cargo build --locked --profile ${CARGO_PROFILE} --quiet
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building reussir-rt library with cargo (${CARGO_PROFILE} profile)"
)

# The default runtime install still copies the full deps directory because the
# existing build and test flow expects the auxiliary Rust artifacts there.
add_custom_target(reussir-rt-install ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/target/${CARGO_PROFILE_DIR}/deps
        ${REUSSIR_RT_OUTPUT_DIR}
    COMMENT "Installing reussir-rt libraries"
    DEPENDS reussir-rt-build
)

# Read rust-toolchain.toml to get the channel
file(READ "${CMAKE_SOURCE_DIR}/rust-toolchain.toml" RUST_TOOLCHAIN_CONTENT)
string(REGEX MATCH "channel = \"([^\"]+)\"" _ ${RUST_TOOLCHAIN_CONTENT})
set(RUST_CHANNEL ${CMAKE_MATCH_1})

if(NOT RUST_CHANNEL)
    message(FATAL_ERROR "Could not parse channel from rust-toolchain.toml")
endif()

message(STATUS "Target Rust channel: ${RUST_CHANNEL}")

# Define where we want the local toolchain to live (intermediate step)
set(LOCAL_RUSTUP_HOME "${CMAKE_BINARY_DIR}/.rustup")
set(LOCAL_CARGO_HOME "${CMAKE_BINARY_DIR}/.cargo")

# Detect the host triple from rustc -vV.
# This is needed to construct the correct toolchain path and to install
# the correct toolchain variant.
execute_process(
    COMMAND rustc -vV
    OUTPUT_VARIABLE RUSTC_VERSION_INFO
    RESULT_VARIABLE RUSTC_RESULT
)

if(NOT RUSTC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to run 'rustc -vV'. Ensure rustc is installed and in PATH.")
endif()

string(REGEX MATCH "host: ([a-zA-Z0-9_-]+)" _ ${RUSTC_VERSION_INFO})
set(RUST_HOST_TRIPLE ${CMAKE_MATCH_1})

if(NOT RUST_HOST_TRIPLE)
    message(FATAL_ERROR "Could not detect host triple from 'rustc -vV' output. Output was: ${RUSTC_VERSION_INFO}")
endif()

string(REPLACE "-" "_" RUST_CARGO_TARGET_ENV ${RUST_HOST_TRIPLE})
string(TOUPPER "${RUST_CARGO_TARGET_ENV}" RUST_CARGO_TARGET_ENV)

set(INSTALLED_TOOLCHAIN_PATH "${LOCAL_RUSTUP_HOME}/toolchains/${RUST_CHANNEL}-${RUST_HOST_TRIPLE}")
set(REUSSIR_RUNTIME_NEEDS_BUILD_STD OFF)
foreach(sanitizer IN LISTS REUSSIR_RUNTIME_SANITIZERS)
    string(TOLOWER "${sanitizer}" sanitizer)
    if(sanitizer STREQUAL "memory" OR sanitizer STREQUAL "thread")
        set(REUSSIR_RUNTIME_NEEDS_BUILD_STD ON)
    endif()
endforeach()

set(RUST_SOURCE_PATH "${INSTALLED_TOOLCHAIN_PATH}/lib/rustlib/src/rust/library")
set(RUSTUP_COMPONENT_ARGS "")
if(REUSSIR_RUNTIME_NEEDS_BUILD_STD)
    list(APPEND RUSTUP_COMPONENT_ARGS --component rust-src)
endif()

# Reuse the cached toolchain when it is already present in the build tree to
# avoid forcing network access during incremental builds.
if(EXISTS "${INSTALLED_TOOLCHAIN_PATH}/bin/rustc" AND
   (NOT REUSSIR_RUNTIME_NEEDS_BUILD_STD OR EXISTS "${RUST_SOURCE_PATH}"))
    message(STATUS "Using cached Rust toolchain ${RUST_CHANNEL}-${RUST_HOST_TRIPLE} from ${INSTALLED_TOOLCHAIN_PATH}")
    add_custom_target(reussir-rust-toolchain-fetch)
else()
    # We need to install the toolchain.
    # We use a custom command/target to ensure it's there.
    # Note: We install the specific toolchain variant matching the host triple
    # to ensure the installed path matches what we'll look for during copy.
    add_custom_target(reussir-rust-toolchain-fetch
        COMMAND ${CMAKE_COMMAND} -E env
            RUSTUP_HOME=${LOCAL_RUSTUP_HOME}
            CARGO_HOME=${LOCAL_CARGO_HOME}
            rustup toolchain install ${RUST_CHANNEL}-${RUST_HOST_TRIPLE} --profile minimal --no-self-update ${RUSTUP_COMPONENT_ARGS}
        COMMENT "Ensuring Rust toolchain ${RUST_CHANNEL}-${RUST_HOST_TRIPLE} is installed in ${LOCAL_RUSTUP_HOME}"
        USES_TERMINAL
    )
endif()

add_custom_target(reussir-rust-install ALL
    # Copy bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INSTALLED_TOOLCHAIN_PATH}/bin"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    # Copy lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INSTALLED_TOOLCHAIN_PATH}/lib"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    COMMENT "Installing rust libraries and tools from ${INSTALLED_TOOLCHAIN_PATH}"
    DEPENDS reussir-rt-build reussir-rust-toolchain-fetch
)

# Main reussir-rt target that depends on the complete build process
add_custom_target(reussir-rt ALL
    DEPENDS reussir-rt-install reussir-rust-install
)

set(REUSSIR_SUPPORTED_RUNTIME_SANITIZERS address leak memory thread)
set(REUSSIR_SANITIZED_RUNTIME_TARGETS "")
set(REUSSIR_RT_ASAN_LIBRARY "")
set(REUSSIR_RT_LSAN_LIBRARY "")
set(REUSSIR_RT_MSAN_LIBRARY "")
set(REUSSIR_RT_TSAN_LIBRARY "")

function(add_reussir_sanitized_runtime sanitizer suffix output_var)
    string(TOLOWER "${sanitizer}" sanitizer)
    set(target_dir "${CMAKE_BINARY_DIR}/target-${suffix}")
    set(output_lib "${REUSSIR_RT_OUTPUT_DIR}/sanitizers/${REUSSIR_RT_SHARED_PREFIX}.${suffix}.${REUSSIR_RT_SHARED_EXT}")
    set(rustflags "-Awarnings -Zsanitizer=${sanitizer}")
    set(target_linker "CARGO_TARGET_${RUST_CARGO_TARGET_ENV}_LINKER=${CMAKE_C_COMPILER}")
    set(target_rustflags "CARGO_TARGET_${RUST_CARGO_TARGET_ENV}_RUSTFLAGS=${rustflags}")
    set(cargo_extra_args "")

    if(sanitizer STREQUAL "memory" AND REUSSIR_MSAN_TRACK_ORIGINS)
        string(APPEND rustflags " -Cllvm-args=-msan-track-origins=2")
        set(target_rustflags "CARGO_TARGET_${RUST_CARGO_TARGET_ENV}_RUSTFLAGS=${rustflags}")
    endif()

    if(NOT WIN32 AND NOT APPLE)
        string(APPEND rustflags " -Clink-arg=-Wl,-soname,${REUSSIR_RT_SHARED_PREFIX}.${suffix}.${REUSSIR_RT_SHARED_EXT}")
        set(target_rustflags "CARGO_TARGET_${RUST_CARGO_TARGET_ENV}_RUSTFLAGS=${rustflags}")
    endif()

    if(sanitizer STREQUAL "memory" OR sanitizer STREQUAL "thread")
        list(APPEND cargo_extra_args -Zbuild-std)
    endif()

    add_custom_target(reussir-rt-${suffix}-build
        COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${target_dir}
            ${target_linker}
            ${target_rustflags}
            cargo build ${cargo_extra_args} --locked --target ${RUST_HOST_TRIPLE} --profile ${CARGO_PROFILE} --quiet --no-default-features
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building sanitized runtime ${output_lib}"
        DEPENDS reussir-rust-toolchain-fetch
    )

    add_custom_target(reussir-rt-${suffix}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${REUSSIR_RT_OUTPUT_DIR}/sanitizers"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${target_dir}/${RUST_HOST_TRIPLE}/${CARGO_PROFILE_DIR}/${REUSSIR_RT_SHARED_LIB}
            ${output_lib}
        COMMENT "Installing sanitized runtime ${output_lib}"
        DEPENDS reussir-rt-${suffix}-build
    )

    list(APPEND REUSSIR_SANITIZED_RUNTIME_TARGETS reussir-rt-${suffix})
    set(REUSSIR_SANITIZED_RUNTIME_TARGETS "${REUSSIR_SANITIZED_RUNTIME_TARGETS}" PARENT_SCOPE)
    set(${output_var} "${output_lib}" PARENT_SCOPE)
endfunction()

foreach(sanitizer IN LISTS REUSSIR_RUNTIME_SANITIZERS)
    string(TOLOWER "${sanitizer}" sanitizer)
    if(NOT sanitizer IN_LIST REUSSIR_SUPPORTED_RUNTIME_SANITIZERS)
        message(FATAL_ERROR
            "Unsupported runtime sanitizer '${sanitizer}'. "
            "Expected one of: ${REUSSIR_SUPPORTED_RUNTIME_SANITIZERS}")
    endif()

    if(sanitizer STREQUAL "address")
        add_reussir_sanitized_runtime(address asan REUSSIR_RT_ASAN_LIBRARY)
    elseif(sanitizer STREQUAL "leak")
        add_reussir_sanitized_runtime(leak lsan REUSSIR_RT_LSAN_LIBRARY)
    elseif(sanitizer STREQUAL "memory")
        add_reussir_sanitized_runtime(memory msan REUSSIR_RT_MSAN_LIBRARY)
    elseif(sanitizer STREQUAL "thread")
        add_reussir_sanitized_runtime(thread tsan REUSSIR_RT_TSAN_LIBRARY)
    endif()
endforeach()

if(REUSSIR_SANITIZED_RUNTIME_TARGETS)
    add_custom_target(reussir-rt-sanitizers
        DEPENDS ${REUSSIR_SANITIZED_RUNTIME_TARGETS})
else()
    add_custom_target(reussir-rt-sanitizers
        COMMAND ${CMAKE_COMMAND} -E echo "No sanitized Reussir runtimes configured"
        VERBATIM
    )
endif()

set(REUSSIR_RT_ASAN_LIBRARY "${REUSSIR_RT_ASAN_LIBRARY}" PARENT_SCOPE)
set(REUSSIR_RT_LSAN_LIBRARY "${REUSSIR_RT_LSAN_LIBRARY}" PARENT_SCOPE)
set(REUSSIR_RT_MSAN_LIBRARY "${REUSSIR_RT_MSAN_LIBRARY}" PARENT_SCOPE)
set(REUSSIR_RT_TSAN_LIBRARY "${REUSSIR_RT_TSAN_LIBRARY}" PARENT_SCOPE)
set(REUSSIR_SANITIZED_RUNTIME_TARGETS "${REUSSIR_SANITIZED_RUNTIME_TARGETS}" PARENT_SCOPE)
