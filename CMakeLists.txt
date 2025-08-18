cmake_minimum_required(VERSION 3.31)
project(Reussir LANGUAGES CXX C VERSION 0.1.0)

# Options
option(REUSSIR_ENABLE_TESTS "Build tests for Reussir" ON)
option(REUSSIR_ENABLE_PEDANTIC "Use pedantic compiler flags" OFF)

if (REUSSIR_ENABLE_PEDANTIC)
  message(STATUS "Pedantic mode enabled")
  set(DISABLE_CHECKS "-Wno-deprecated-declarations -Wno-ambiguous-reversed-operator")
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
message(STATUS "include dirs for llvm and mlir: ${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}")
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "Tablegen output directory: ${MLIR_TABLEGEN_OUTPUT_DIR}")
include_directories(include ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${MLIR_TABLEGEN_OUTPUT_DIR})

# Subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)

if (REUSSIR_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
