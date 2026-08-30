# Cross-compiling the cargo-built components.
#
# When REUSSIR_CARGO_BUILD_TARGET is set (e.g. x86_64-pc-windows-msvc),
# every cargo invocation in the crate CMakeLists gains `--target` and the
# artifact paths move from target/<profile>/ to target/<triple>/<profile>/.
# For the MSVC target the driver becomes `cargo xwin`, which supplies the
# CRT/SDK include and link environment (the windows-cross dev shell provides
# cargo-xwin, the host proc-macro prefixes, and Wine as the test runner).
#
# Empty (the default) leaves native builds byte-for-byte unchanged.
set(REUSSIR_CARGO_BUILD_TARGET "" CACHE STRING
  "Rust --target triple for the cargo-built components (empty = host)")

set(REUSSIR_CARGO cargo)
set(REUSSIR_CARGO_TARGET_ARGS "")
set(REUSSIR_CARGO_TARGET_SUBDIR "")
if(REUSSIR_CARGO_BUILD_TARGET)
  set(REUSSIR_CARGO_TARGET_ARGS --target ${REUSSIR_CARGO_BUILD_TARGET})
  set(REUSSIR_CARGO_TARGET_SUBDIR "${REUSSIR_CARGO_BUILD_TARGET}/")
  if(REUSSIR_CARGO_BUILD_TARGET MATCHES "windows-msvc")
    # flock serializes the cargo-xwin invocations ninja runs in parallel:
    # each one re-creates the clang-cl symlink in its cache dir, and two
    # doing so concurrently race to a "failed to remove file" crash. Each
    # cargo build parallelizes internally, so the wall-clock cost is small.
    set(REUSSIR_CARGO flock ${CMAKE_BINARY_DIR}/cargo-xwin.lock cargo xwin)

    # Stage the conda runtime DLLs beside the built executables: Wine
    # searches the exe's own directory first, which is the only resolution
    # path that survives into child processes (rene.exe spawning rrc.exe
    # inherits a scrubbed Windows PATH — WINEPATH on the launching shell
    # does not reach it).
    if(DEFINED ENV{REUSSIR_MSVC_LLVM_PREFIX})
      file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
      foreach(_dll zlib.dll zstd.dll libxml2.dll)
        if(EXISTS "$ENV{REUSSIR_MSVC_LLVM_PREFIX}/bin/${_dll}")
          file(COPY "$ENV{REUSSIR_MSVC_LLVM_PREFIX}/bin/${_dll}"
               DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        endif()
      endforeach()
      message(STATUS "Staged conda runtime DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
  endif()
  message(STATUS "Cargo components cross-compile for ${REUSSIR_CARGO_BUILD_TARGET} (driver: ${REUSSIR_CARGO})")
endif()
