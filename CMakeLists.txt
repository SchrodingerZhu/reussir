cmake_minimum_required(VERSION 3.28)
project(Reussir LANGUAGES CXX C VERSION 0.1.0)

# Options
option(REUSSIR_ENABLE_TESTS "Build tests for Reussir" ON)
option(REUSSIR_ENABLE_PEDANTIC "Use pedantic compiler flags" OFF)
set(REUSSIR_RUNTIME_SANITIZERS "" CACHE STRING "Semicolon-separated sanitizer runtimes to build (address, leak, memory, thread)")
option(REUSSIR_MSAN_TRACK_ORIGINS "Enable origin tracking in the MSan runtime variant" ON)

if (REUSSIR_ENABLE_PEDANTIC)
  message(STATUS "Pedantic mode enabled")
  # For now, we disable broken warnings in MLIR's own code.
  set(DISABLE_CHECKS
    "-Wno-deprecated-declarations -Wno-ambiguous-reversed-operator -Wno-dangling-assignment-gsl"
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -Werror -pedantic ${DISABLE_CHECKS}")
endif()

# Adjust output paths
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# C++ Standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)

# Find LLVM and MLIR
include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)
if(DEFINED ENV{NIX_STORE})
  find_package(GTest REQUIRED)
  message(STATUS "Detected Nix environment; injecting MLIR/LLVM/GTest include flags to walk around scan-deps issue")
  get_target_property(GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -I${MLIR_INCLUDE_DIRS} -I${LLVM_INCLUDE_DIRS} -I${GTEST_INCLUDE_DIRS}")
elseif(APPLE AND NOT DEFINED ENV{NIX_STORE})
  # On macOS with Homebrew LLVM, the module scanner cannot find system headers.
  # We need to inject the SDK sysroot include path similar to the Nix workaround.
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE XCRUN_RESULT
  )
  if(XCRUN_RESULT EQUAL 0 AND MACOS_SDK_PATH)
    message(STATUS "Detected macOS; injecting SDK and MLIR/LLVM include flags for scan-deps: ${MACOS_SDK_PATH}")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -I${MLIR_INCLUDE_DIRS} -I${LLVM_INCLUDE_DIRS} -isysroot ${MACOS_SDK_PATH}")
  else()
    message(WARNING "Could not determine macOS SDK path via xcrun. Module scanning may fail.")
  endif()
endif()
message(STATUS "include dirs for llvm and mlir: ${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}")
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "Tablegen output directory: ${MLIR_TABLEGEN_OUTPUT_DIR}")

# Include Immer
include(cmake/Immer.cmake)

# Include the mlir-sync `sync` dialect (synchronization primitives). Reussir's
# lock-guarded cells lower onto this dialect.
include(cmake/MlirSync.cmake)

# Detect object format
include(cmake/ObjectFormat.cmake)

# Include TPDE only for ELF object format and LLVM <= 21

if(TARGET_OBJECT_FORMAT STREQUAL "ELF")
  message(STATUS "ELF object format detected, including TPDE support")
  include(cmake/TPDE.cmake)
  set(REUSSIR_HAS_TPDE ON)
else()
  message(STATUS "Non-ELF object format detected, TPDE not supported")
  set(REUSSIR_HAS_TPDE OFF)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLIR_TABLEGEN_OUTPUT_DIR}
    ${immer_SOURCE_DIR}
    ${MLIR_SYNC_INCLUDE_DIRS}
)

include_directories(SYSTEM
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

# Subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)
add_subdirectory(crates/reussir-rt)
add_subdirectory(crates/reussir-syntax)
add_subdirectory(crates/reussir-backend)
add_subdirectory(crates/reussir-jit)
add_subdirectory(crates/reussir-codegen)
add_subdirectory(crates/reussir-compiler)
add_subdirectory(crates/reussir-repl)
add_subdirectory(crates/rene)

if (REUSSIR_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
