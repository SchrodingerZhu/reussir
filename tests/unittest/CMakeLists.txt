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
    ReussirCAPI
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

# The RustCompiler tests invoke rustc and resolve `extern crate reussir_rt`
# against the runtime's deps directory. With no toolchain or runtime staged
# into the build tree anymore, hand the tests the configure-time paths as
# compile definitions; they prime REUSSIR_RUSTC / REUSSIR_RUSTC_DEPS from
# them when the caller's environment does not already say otherwise.
# (A ctest ENVIRONMENT property cannot carry both variables: list values do
# not survive gtest_discover_tests' property forwarding.)
target_compile_definitions(reussir-ut
  PRIVATE
    REUSSIR_UT_RUSTC="${REUSSIR_RUSTC_PATH}"
    REUSSIR_UT_RUSTC_DEPS="${REUSSIR_RT_TARGET_DIR}/deps")

# Register with CTest
gtest_discover_tests(reussir-ut)
