# Find GTest
find_package(GTest REQUIRED)
include(GoogleTest)

# Add unit test executable
add_executable(reussir-ut)

file(GLOB_RECURSE module_files "*.cppm")
file(GLOB_RECURSE source_files "*.cpp")
target_sources(
    reussir-ut
    PRIVATE FILE_SET CXX_MODULES FILES ${module_files} 
    PRIVATE ${source_files})

# Link against required libraries
target_link_libraries(reussir-ut
  PRIVATE
    ReussirRustCompiler
    MLIRReussir
    GTest::gtest
    GTest::gtest_main
)

if (LLVM_VERSION VERSION_GREATER_EQUAL 22)
  target_link_libraries(reussir-ut
    PRIVATE
      MLIRRegisterAllDialects
  )
else()
  message(STATUS "LLVM_VERSION is ${LLVM_VERSION}, skipping MLIRRegisterAllDialects")
endif()

# Include necessary directories
target_include_directories(reussir-ut
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

add_dependencies(reussir-ut reussir-rt)

# Register with CTest. The RustCompiler tests invoke rustc and resolve
# `extern crate reussir_rt` against the runtime's deps directory; with no
# toolchain or runtime staged into the build tree anymore, both come in
# through the environment (the same variables the lit suite sets).
gtest_discover_tests(reussir-ut
  PROPERTIES ENVIRONMENT
    "REUSSIR_RUSTC=${REUSSIR_RUSTC_PATH};REUSSIR_RUSTC_DEPS=${REUSSIR_RT_TARGET_DIR}/deps")
