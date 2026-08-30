# Cross toolchain: Linux host -> x86_64-pc-windows-msvc, GNU-driver clang.
#
# Deliberately NOT clang-cl: CMake's C++ module dependency scanning is only
# wired for the GNU frontend variant (Modules/Compiler/Clang-CXX.cmake), and
# lib/Bridge ships CXX_MODULES file sets — with clang-cl the generate step
# fails with "the compiler does not provide a way to discover the import
# graph". The GNU driver targets the same MSVC ABI (COFF objects, MSVC STL,
# ucrt via -fms-runtime-lib=dll) and links through lld-link via -fuse-ld=lld.
#
# The MSVC CRT and Windows SDK come from cargo-xwin's splat cache (the same
# license-gated download the Rust cross build uses); point REUSSIR_XWIN_DIR
# elsewhere to override.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(DEFINED ENV{REUSSIR_XWIN_DIR})
  set(_reussir_xwin "$ENV{REUSSIR_XWIN_DIR}")
else()
  set(_reussir_xwin "$ENV{HOME}/.cache/cargo-xwin/xwin")
endif()
if(NOT EXISTS "${_reussir_xwin}/crt/include")
  message(FATAL_ERROR
    "xwin CRT/SDK splat not found at ${_reussir_xwin}. Run any "
    "`cargo xwin build` once (or `xwin splat`) to populate it, or set "
    "REUSSIR_XWIN_DIR.")
endif()

# xwin ships only the release CRT (no msvcrtd.lib), so pin the runtime to
# the release DLL CRT for every configuration and keep try_compile off the
# Debug default it would otherwise use during compiler detection.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_TRY_COMPILE_CONFIGURATION RelWithDebInfo)

# Target executables run through Wine: gtest_discover_tests enumerates the
# unit-test binary at build time, and ctest executes it, both via this
# emulator. (The lit suites run PE binaries through a binfmt_misc
# registration instead; see the windows-cross-test workflow.)
set(CMAKE_CROSSCOMPILING_EMULATOR wine)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
# GNU archiver syntax (cmake drives <AR> qc ... for the GNU frontend
# variant); the resulting ar archives are valid inputs for lld-link.
set(CMAKE_AR llvm-ar)
set(CMAKE_RANLIB llvm-ranlib)
set(CMAKE_RC_COMPILER llvm-rc)

string(JOIN " " _reussir_cross_flags
  --target=x86_64-pc-windows-msvc
  -fms-runtime-lib=dll
  "-isystem ${_reussir_xwin}/crt/include"
  "-isystem ${_reussir_xwin}/sdk/include/ucrt"
  "-isystem ${_reussir_xwin}/sdk/include/um"
  "-isystem ${_reussir_xwin}/sdk/include/shared"
  "-isystem ${_reussir_xwin}/sdk/include/winrt")
set(CMAKE_C_FLAGS_INIT "${_reussir_cross_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_reussir_cross_flags}")

string(JOIN " " _reussir_cross_ldflags
  --target=x86_64-pc-windows-msvc
  -fuse-ld=lld
  "-L${_reussir_xwin}/crt/lib/x86_64"
  "-L${_reussir_xwin}/sdk/lib/um/x86_64"
  "-L${_reussir_xwin}/sdk/lib/ucrt/x86_64")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_reussir_cross_ldflags}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_reussir_cross_ldflags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_reussir_cross_ldflags}")

# Libraries and headers come only from the prefixes the build passes in
# (the conda MSVC LLVM prefix); host programs stay findable.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
