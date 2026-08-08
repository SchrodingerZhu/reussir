# The in-repo runtime build for the dev/test flow. Shipped compilers no
# longer stage a runtime or a Rust toolchain from here: `rene` bakes the
# bundled reussir-rt source with the user's own toolchain
# (crates/rene/src/rt.rs). The lit suites and unit tests point straight at
# the cargo-managed target directory this file builds into.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "dev")
    set(CARGO_PROFILE_DIR "debug")
else()
    set(CARGO_PROFILE "release")
    set(CARGO_PROFILE_DIR "release")
endif()

# The runtime builds into its own managed target directory, not the shared
# ${CMAKE_BINARY_DIR}/target: reussir-jit builds reussir-rt there too, with
# default features off, and the dylib crate type makes cargo drop the
# metadata hash from the artifact names — the two feature variants would
# clobber each other's libreussir_rt.rlib. A dedicated directory keeps the
# uplifted dylib/staticlib and the deps/ tree (polyffi's `rustc -L`)
# coherent, whatever else builds concurrently.
set(REUSSIR_RT_TARGET_DIR ${CMAKE_BINARY_DIR}/target-rt/${CARGO_PROFILE_DIR})
set(REUSSIR_RT_TARGET_DIR "${REUSSIR_RT_TARGET_DIR}" PARENT_SCOPE)

# The developer toolchain, resolved once: the test suites pin rrc's polyffi
# compiles to it (REUSSIR_RUSTC), and its target libdir is where the runtime
# dylib's dynamically linked libstd lives at link and load time. Going
# through PATH respects rustup shims, so rust-toolchain.toml still selects
# the pinned toolchain.
find_program(REUSSIR_RUSTC_PATH rustc REQUIRED)
execute_process(
    COMMAND ${REUSSIR_RUSTC_PATH} --print target-libdir
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE REUSSIR_RUST_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE rustc_libdir_result)
if(NOT rustc_libdir_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to run '${REUSSIR_RUSTC_PATH} --print target-libdir'. "
        "Ensure rustc is installed and in PATH.")
endif()
set(REUSSIR_RUST_LIBDIR "${REUSSIR_RUST_LIBDIR}" PARENT_SCOPE)

add_custom_target(reussir-rt ALL
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/target-rt
        RUSTFLAGS=-Awarnings
        cargo build --locked --profile ${CARGO_PROFILE} --quiet
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building reussir-rt with cargo (${CARGO_PROFILE} profile)"
)

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

# Detect the host triple from rustc -vV; the sanitized runtime builds pass it
# explicitly so the per-target cargo environment variables apply.
execute_process(
    COMMAND ${REUSSIR_RUSTC_PATH} -vV
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
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

set(REUSSIR_SUPPORTED_RUNTIME_SANITIZERS address leak memory thread)
set(REUSSIR_SANITIZED_RUNTIME_TARGETS "")
set(REUSSIR_RT_ASAN_LIBRARY "")
set(REUSSIR_RT_LSAN_LIBRARY "")
set(REUSSIR_RT_MSAN_LIBRARY "")
set(REUSSIR_RT_TSAN_LIBRARY "")

# The sanitized variants need a nightly toolchain (-Zsanitizer), and the
# memory/thread ones additionally the rust-src component (-Zbuild-std). Both
# come from the developer toolchain now; nothing is fetched here.
function(add_reussir_sanitized_runtime sanitizer suffix output_var)
    string(TOLOWER "${sanitizer}" sanitizer)
    set(target_dir "${CMAKE_BINARY_DIR}/target-${suffix}")
    set(output_lib "${REUSSIR_RT_OUTPUT_DIR}/sanitizers/${REUSSIR_RT_SHARED_PREFIX}.${suffix}.${REUSSIR_RT_SHARED_EXT}")
    set(rustflags "-Awarnings -Zsanitizer=${sanitizer}")
    # Only Windows needs a library-path override (see below); elsewhere the
    # linker driver already knows where its own sanitizer runtimes live.
    set(target_sanitizer_lib "")
    # Point rustc at the C compiler as its linker driver, so the sanitizer
    # runtime it asks for is the one this Clang ships. That is wrong on an
    # `*-pc-windows-msvc` target: rustc emits MSVC-style linker arguments
    # there (`/NOLOGO`, `/OUT:…`, bare `kernel32.lib`), and handing those to
    # the clang driver fails immediately with "no such file or directory:
    # '/NOLOGO'". Leave the target's own linker in place on MSVC — rustc
    # resolves `link.exe` itself and links the ASan runtime from its
    # `-Zsanitizer` support.
    if(MSVC OR RUST_HOST_TRIPLE MATCHES "windows-msvc")
        set(target_linker "")
        # ...but then rustc resolves `clang_rt.asan_dynamic-x86_64.lib` through
        # the vcvars `LIB` path, which is Visual Studio's *own* ASan runtime.
        # Microsoft's fork and this Clang's are different implementations behind
        # one DLL name: the MSVC one additionally exports
        # `__sanitizer_cov_*_cleanup__dll`, which Clang's does not. A runtime
        # linked against MSVC's then fails to load next to a Clang-instrumented
        # test program — `0xC0000139`, entry point not found — because the test
        # brings Clang's copy of that DLL. Point rustc at this Clang's
        # compiler-rt so both halves of the process agree on one ASan.
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -print-resource-dir
            OUTPUT_VARIABLE _clang_resource_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _clang_resource_result)
        if(NOT _clang_resource_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to run '${CMAKE_C_COMPILER} -print-resource-dir'; needed to "
                "resolve the sanitizer runtime the sanitized reussir-rt links against.")
        endif()
        file(TO_NATIVE_PATH "${_clang_resource_dir}/lib/windows" _clang_sanitizer_libdir)
        # Through `LIB` rather than a `-Clink-arg=/LIBPATH:` in RUSTFLAGS:
        # cargo splits RUSTFLAGS on whitespace, so a resource directory under
        # `C:\Program Files\...` would arrive as several rustc arguments and no
        # amount of quoting for link.exe would help. `LIB` is a single value
        # link.exe reads directly, and prepending keeps the vcvars entries that
        # supply the CRT and Windows SDK.
        set(_vcvars_lib "$ENV{LIB}")
        # Semicolons separate LIB entries and also CMake list elements; escape
        # them so the value survives as one argument to `cmake -E env`.
        string(REPLACE ";" "\\;" _vcvars_lib "${_vcvars_lib}")
        set(target_sanitizer_lib "LIB=${_clang_sanitizer_libdir}\\;${_vcvars_lib}")
    else()
        set(target_linker "CARGO_TARGET_${RUST_CARGO_TARGET_ENV}_LINKER=${CMAKE_C_COMPILER}")
    endif()
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
            # Cargo gives a global `RUSTFLAGS` precedence over the
            # target-specific form, so an ambient value — `-D warnings` in the
            # CI workflows — silently *replaces* the flags below rather than
            # adding to them. The sanitized build then loses `-Zsanitizer` and
            # its custom SONAME, the copied library advertises the ordinary
            # `libreussir_rt.so`, and the loader resolves the uninstrumented
            # runtime from `target-rt/release/deps`. The suite goes green
            # while sanitizing nothing, which is the failure
            # `sanitizer/runtime-allocator-asan.rr` exists to catch — and did,
            # on this very build.
            --unset=RUSTFLAGS
            --unset=CARGO_ENCODED_RUSTFLAGS
            CARGO_TARGET_DIR=${target_dir}
            ${target_linker}
            ${target_sanitizer_lib}
            ${target_rustflags}
            cargo build ${cargo_extra_args} --locked --target ${RUST_HOST_TRIPLE} --profile ${CARGO_PROFILE} --quiet --no-default-features
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building sanitized runtime ${output_lib}"
    )

    if(WIN32)
        # Windows binds a DLL by the name recorded in the import library, and
        # cargo's import library records `reussir_rt.dll`. Staging the runtime
        # under a new name therefore breaks the pairing silently: a program
        # linked against a copied import library imports `reussir_rt.dll` and
        # the loader supplies the *unsanitized* runtime sitting on the same
        # search path, so instrumented code runs against an allocator the
        # sanitizer cannot see and the suite passes while checking far less
        # than it appears to.
        #
        # So keep cargo's own names and give each variant its own directory.
        # The import library stays valid, and which runtime a test loads is
        # decided by search order — see `REUSSIR_ASAN_RUNTIME_DIR` in
        # tests/integration, which puts this directory ahead of the ordinary
        # one for sanitizer RUN lines only.
        set(variant_dir "${REUSSIR_RT_OUTPUT_DIR}/sanitizers/${suffix}")
        set(staged_dll "${variant_dir}/${REUSSIR_RT_SHARED_LIB}")
        set(staged_implib "${variant_dir}/${REUSSIR_RT_SHARED_PREFIX}.dll.lib")
        add_custom_target(reussir-rt-${suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${variant_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${target_dir}/${RUST_HOST_TRIPLE}/${CARGO_PROFILE_DIR}/${REUSSIR_RT_SHARED_LIB}
                ${staged_dll}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${target_dir}/${RUST_HOST_TRIPLE}/${CARGO_PROFILE_DIR}/${REUSSIR_RT_SHARED_PREFIX}.dll.lib
                ${staged_implib}
            COMMENT "Installing sanitized runtime ${staged_dll}"
            DEPENDS reussir-rt-${suffix}-build
        )
        set(${output_var} "${staged_implib}" PARENT_SCOPE)
        set(REUSSIR_RT_${suffix}_DIR "${variant_dir}" PARENT_SCOPE)
    else()
        add_custom_target(reussir-rt-${suffix}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${REUSSIR_RT_OUTPUT_DIR}/sanitizers"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${target_dir}/${RUST_HOST_TRIPLE}/${CARGO_PROFILE_DIR}/${REUSSIR_RT_SHARED_LIB}
                ${output_lib}
            COMMENT "Installing sanitized runtime ${output_lib}"
            DEPENDS reussir-rt-${suffix}-build
        )
        set(${output_var} "${output_lib}" PARENT_SCOPE)
    endif()
    add_dependencies(reussir-rt-${suffix} reussir-rt-${suffix}-build)

    list(APPEND REUSSIR_SANITIZED_RUNTIME_TARGETS reussir-rt-${suffix})
    set(REUSSIR_SANITIZED_RUNTIME_TARGETS "${REUSSIR_SANITIZED_RUNTIME_TARGETS}" PARENT_SCOPE)
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
    add_dependencies(reussir-rt-sanitizers ${REUSSIR_SANITIZED_RUNTIME_TARGETS})
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
