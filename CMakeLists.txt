cmake_minimum_required(VERSION 4.0)
project(Reussir LANGUAGES CXX C VERSION 0.1.0)

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
add_subdirectory(src)
