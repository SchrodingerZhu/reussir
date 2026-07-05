
message(STATUS "finding tools in ${LLVM_TOOLS_BINARY_DIR}")

function(find_llvm_tool VAR TOOL)
    find_program(${VAR}
        NAMES "${TOOL}${CMAKE_EXECUTABLE_SUFFIX}"
        HINTS ${LLVM_TOOLS_BINARY_DIR}
        REQUIRED)
    message(STATUS "${TOOL} path: ${${VAR}}")
endfunction()

find_llvm_tool(filecheck_path FileCheck)
find_llvm_tool(not_path not)
find_llvm_tool(opt_path opt)
find_llvm_tool(llc_path llc)
find_llvm_tool(lli_path lli)

if(REUSSIR_RUNTIME_SANITIZERS AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "Sanitizer integration tests require Clang as C compiler because they "
    "compile generated LLVM IR with `%cc -x ir`.")
endif()

if (WIN32)
  set(extra_sys_libs "-lucrt -lntdll -lws2_32 -ladvapi32 -lbcrypt -luserenv")
  set(REUSSIR_LIBRARY_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(REUSSIR_RT_LIBRARY_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/reussir_rt.dll)
  set(rpath_flag "")
elseif(APPLE)
  set(extra_sys_libs "-framework Security -framework CoreFoundation")
  set(REUSSIR_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(REUSSIR_RT_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libreussir_rt.dylib)
  set(rpath_flag "-Wl,-rpath,${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
  set(REUSSIR_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(REUSSIR_RT_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libreussir_rt.so)
  set(rpath_flag "-Wl,-rpath,${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

set(rustc_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rustc${CMAKE_EXECUTABLE_SUFFIX})
set(reussir_rrc_path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rrc${CMAKE_EXECUTABLE_SUFFIX})
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

# `rrepl` carries an order-only dependency on reussir-rt (see
# crates/reussir-repl/CMakeLists.txt), so it is safe to depend on here: it
# builds after the runtime, not concurrently, and enables the repl-rs suite.
add_custom_target(check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v
  DEPENDS reussir-opt reussir-rt reussir-translate reussir-syntax rrc rrepl)

if(REUSSIR_SANITIZED_RUNTIME_TARGETS)
  add_custom_target(check-sanitizer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v --filter "sanitizer/"
    DEPENDS reussir-opt reussir-rt reussir-rt-sanitizers reussir-translate rrc)
endif()
