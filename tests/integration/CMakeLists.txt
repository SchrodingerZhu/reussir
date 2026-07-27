
message(STATUS "finding tools in ${LLVM_TOOLS_BINARY_DIR}")

function(find_llvm_tool VAR TOOL)
    set(tool_names "${TOOL}${CMAKE_EXECUTABLE_SUFFIX}" "${TOOL}")
    foreach(extra_name IN LISTS ARGN)
        list(APPEND tool_names "${extra_name}")
    endforeach()
    find_program(${VAR}
        NAMES ${tool_names}
        HINTS ${LLVM_TOOLS_BINARY_DIR}
        REQUIRED)
    message(STATUS "${TOOL} path: ${${VAR}}")
endfunction()

find_llvm_tool(filecheck_path FileCheck filecheck filecheck.exe)
find_program(not_executable_path
    NAMES "not${CMAKE_EXECUTABLE_SUFFIX}" not
    HINTS ${LLVM_TOOLS_BINARY_DIR})
if(not_executable_path)
    set(not_path "${not_executable_path}")
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(not_script_path "${CMAKE_CURRENT_BINARY_DIR}/reussir-lit-not.py")
    file(WRITE "${not_script_path}" [=[
import subprocess
import sys


def main():
    args = sys.argv[1:]
    expect_crash = False
    if args and args[0] == "--crash":
        expect_crash = True
        args = args[1:]
    if not args:
        print("usage: not [--crash] command [args...]", file=sys.stderr)
        return 2

    try:
        completed = subprocess.run(args)
    except FileNotFoundError as err:
        print(f"not: {err}", file=sys.stderr)
        return 1

    code = completed.returncode
    if expect_crash:
        crashed = code < 0 or code >= 0x80000000
        return 0 if crashed else 1
    return 0 if code != 0 else 1


if __name__ == "__main__":
    sys.exit(main())
]=])
    if(WIN32)
        set(not_wrapper_path "${CMAKE_CURRENT_BINARY_DIR}/reussir-lit-not.cmd")
        file(WRITE "${not_wrapper_path}" "@echo off\r\n\"${Python3_EXECUTABLE}\" \"${not_script_path}\" %*\r\n")
    else()
        set(not_wrapper_path "${CMAKE_CURRENT_BINARY_DIR}/reussir-lit-not")
        file(WRITE "${not_wrapper_path}" "#!/bin/sh\nexec \"${Python3_EXECUTABLE}\" \"${not_script_path}\" \"$@\"\n")
        file(CHMOD "${not_wrapper_path}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endif()
    set(not_path "${not_wrapper_path}")
endif()
message(STATUS "not path: ${not_path}")
find_llvm_tool(opt_path opt)
find_llvm_tool(llc_path llc)
find_llvm_tool(lli_path lli)
# The static-library suite inspects what `rrc --emit staticlib` archived:
# llvm-ar lists and extracts members, llvm-nm reads the symbol table (over
# native objects and, under --lto, bitcode).
find_llvm_tool(llvm_ar_path llvm-ar)
find_llvm_tool(llvm_nm_path llvm-nm)
find_llvm_tool(llvm_bcanalyzer_path llvm-bcanalyzer)
find_llvm_tool(llvm_dis_path llvm-dis)
# The COFF comdat suite reads section characteristics (IMAGE_SCN_LNK_COMDAT)
# off cross-compiled objects.
find_llvm_tool(llvm_readobj_path llvm-readobj)

# lldb drives the end-to-end debug-info suite (tests/integration/debuginfo):
# it runs compiled executables under the debugger, with the Reussir formatters
# preloaded, and FileChecks the printed variables. Any reasonably recent lldb
# can read the DWARF the pipeline emits — it need not match the LLVM major.
# Optional: when absent the `lldb` lit feature stays off and those tests are
# marked unsupported.
find_program(lldb_path
    NAMES "lldb-${LLVM_VERSION_MAJOR}" "lldb${CMAKE_EXECUTABLE_SUFFIX}" lldb
    HINTS ${LLVM_TOOLS_BINARY_DIR})
if(lldb_path)
    message(STATUS "lldb path: ${lldb_path}")
else()
    set(lldb_path "")
    message(STATUS "lldb not found: debuginfo lit tests will be skipped")
endif()
set(reussir_lldb_formatters_path
    "${CMAKE_SOURCE_DIR}/tool/lldb/reussir_formatters.py")

# gdb is the pure-DWARF oracle for the same suite: it decodes the
# `DW_TAG_variant_part` encoding natively (no formatters), so its tests pin
# the debug info itself rather than any script. Optional like lldb.
find_program(gdb_path NAMES gdb)
if(gdb_path)
    message(STATUS "gdb path: ${gdb_path}")
else()
    set(gdb_path "")
    message(STATUS "gdb not found: gdb debuginfo lit tests will be skipped")
endif()

if(REUSSIR_RUNTIME_SANITIZERS AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "Sanitizer integration tests require Clang as C compiler because they "
    "compile generated LLVM IR with `%cc -x ir`.")
endif()

set(cc_runtime_flags "")

# The launcher link (frontend/main_attribute.rr) has rustc drive the MSVC
# linker. rustc's own discovery, under a vcvars environment, degrades to a
# bare PATH scan for `link.exe` — and the CI shells carry Git-for-Windows'
# coreutils /usr/bin/link ahead of MSVC, which then rejects the link line.
# Resolve the real linker here, where the vcvars variables are dependable,
# and hand it to rustc explicitly; when there is no vcvars environment
# (a plain dev box), leave it to rustc's registry-based discovery.
set(REUSSIR_MSVC_LINKER_PATH "")
if(WIN32 AND DEFINED ENV{VCToolsInstallDir})
  file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _vc_tools_dir)
  set(_vc_host "$ENV{VSCMD_ARG_HOST_ARCH}")
  set(_vc_tgt "$ENV{VSCMD_ARG_TGT_ARCH}")
  if(NOT _vc_host)
    set(_vc_host x64)
  endif()
  if(NOT _vc_tgt)
    set(_vc_tgt x64)
  endif()
  find_program(reussir_msvc_linker link.exe
    HINTS "${_vc_tools_dir}/bin/Host${_vc_host}/${_vc_tgt}"
    NO_DEFAULT_PATH)
  if(reussir_msvc_linker)
    set(REUSSIR_MSVC_LINKER_PATH "${reussir_msvc_linker}")
    message(STATUS "MSVC linker for rustc-driven links: ${REUSSIR_MSVC_LINKER_PATH}")
  endif()
endif()

# The runtime artifacts live in the cargo-managed target directory the
# reussir-rt build fills (crates/reussir-rt/CMakeLists.txt). Its deps/
# subdirectory carries everything the suites need in one place — the runtime
# dylib/staticlib the link lines take, and the rlib tree polyffi's `rustc -L`
# resolves `extern crate reussir_rt` from — the same content the old flow
# copied into the build tree. Nothing is staged anymore; the runtime dylib
# links libstd dynamically, so link lines also need the toolchain's own
# libdir (rpath-link at link time; lit puts it on the platform library path
# for RUN lines, see lit.cfg.py).
set(REUSSIR_RT_DEPS_DIR ${REUSSIR_RT_TARGET_DIR}/deps)
if (WIN32)
  set(extra_sys_libs "-Wl,/nodefaultlib:libucrt -Wl,/defaultlib:ucrt -lntdll -lws2_32 -ladvapi32 -lbcrypt -luserenv")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    # Rust MSVC targets use DLL CRT imports; keep lit-linked test drivers matched.
    set(cc_runtime_flags "-fms-runtime-lib=dll")
  endif()
  set(REUSSIR_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR})
  set(REUSSIR_RT_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR}/reussir_rt.dll)
  set(rpath_flag "")
elseif(APPLE)
  set(extra_sys_libs "-framework Security -framework CoreFoundation")
  set(REUSSIR_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR})
  set(REUSSIR_RT_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR}/libreussir_rt.dylib)
  set(rpath_flag "-Wl,-rpath,${REUSSIR_RT_DEPS_DIR} -Wl,-rpath,${REUSSIR_RUST_LIBDIR}")
else()
  set(REUSSIR_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR})
  set(REUSSIR_RT_LIBRARY_PATH ${REUSSIR_RT_DEPS_DIR}/libreussir_rt.so)
  set(rpath_flag "-Wl,-rpath,${REUSSIR_RT_DEPS_DIR} -Wl,-rpath-link,${REUSSIR_RUST_LIBDIR}")
endif()

# The developer toolchain's rustc, resolved by crates/reussir-rt (a PATH
# lookup respecting rustup shims) — no toolchain is staged into build/bin.
set(rustc_path ${REUSSIR_RUSTC_PATH})
set(reussir_rrc_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rrc${CMAKE_EXECUTABLE_SUFFIX})
# The package manager (`crates/rene`), driving rrc for the rene/ suite.
set(rene_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rene${CMAKE_EXECUTABLE_SUFFIX})
# The Rust REPL (`crates/reussir-repl`, binary `rrepl`).
set(rrepl_path ${REUSSIR_RREPL_BINARY})
set(reussir_syntax_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/reussir-syntax${CMAKE_EXECUTABLE_SUFFIX})
set(reussir_rt_asan_path ${REUSSIR_RT_ASAN_LIBRARY})
set(reussir_rt_lsan_path ${REUSSIR_RT_LSAN_LIBRARY})
set(reussir_rt_msan_path ${REUSSIR_RT_MSAN_LIBRARY})
set(reussir_rt_tsan_path ${REUSSIR_RT_TSAN_LIBRARY})

set(rpath_san_flag "-Wl,-rpath,${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/sanitizers")

set(asan_flags "-fsanitize=address -fno-omit-frame-pointer -g")
set(lsan_flags "-fsanitize=leak -fno-omit-frame-pointer -g")
set(msan_flags "-fsanitize=memory -fno-omit-frame-pointer -g -fno-optimize-sibling-calls")
set(tsan_flags "-fsanitize=thread -fno-omit-frame-pointer -g")
if(REUSSIR_MSAN_TRACK_ORIGINS)
  string(APPEND msan_flags " -fsanitize-memory-track-origins=2")
endif()

set(asan_env "ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:halt_on_error=1")
set(lsan_env "LSAN_OPTIONS=abort_on_error=1:report_objects=1")
set(msan_env "MSAN_OPTIONS=abort_on_error=1:halt_on_error=1:poison_in_dtor=1")
set(tsan_env "TSAN_OPTIONS=abort_on_error=1:halt_on_error=1")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

if(DEFINED ENV{MSYSTEM_PREFIX})
  # e.g. MSYSTEM_PREFIX = /mingw64 or /clang64
  message(STATUS "MSYS environment detected: $ENV{MSYSTEM_PREFIX}")

  # Convert /mingw64 -> C:/msys64/mingw64
  file(TO_CMAKE_PATH "$ENV{MSYSTEM_PREFIX}" _msys_prefix_normalized)

  # Build the full path to python
  set(_msys_python "${_msys_prefix_normalized}/bin/python3.exe")

  if(EXISTS "${_msys_python}")
    message(STATUS "Using MSYS Python at: ${_msys_python}")
    set(Python3_EXECUTABLE "${_msys_python}" CACHE FILEPATH "" FORCE)
  else()
    message(WARNING "MSYS Python was expected at ${_msys_python} but not found.")
  endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# USES_TERMINAL runs lit in ninja's console pool with the real terminal on
# stdout — lit only enables its colors and progress bar on a TTY, and a
# piped ninja job never is one.
add_custom_target(check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v
  DEPENDS reussir-opt reussir-rt reussir-translate reussir-llvm-opt
          reussir-syntax rrc rrepl rene
  USES_TERMINAL)

if(REUSSIR_SANITIZED_RUNTIME_TARGETS)
  add_custom_target(check-sanitizer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v --filter "sanitizer/"
    DEPENDS reussir-opt reussir-rt reussir-rt-sanitizers reussir-translate rrc
    USES_TERMINAL)
endif()
