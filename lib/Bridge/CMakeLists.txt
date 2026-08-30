get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# Find spdlog
find_package(spdlog REQUIRED)

# Conditionally add TPDE to link libraries
set(TPDE_LINK_LIB "")
if(REUSSIR_HAS_TPDE)
  set(TPDE_LINK_LIB tpde::tpde_llvm)
endif()

add_mlir_library(MLIRReussirBridge
  SHARED
  DEPENDS
    MLIRReussir
    MLIRReussirConversion
    MLIRReussirTransformation
  LINK_COMPONENTS
    Core
    ExecutionEngine
    JITLink
    nativecodegen
    OrcJIT
    RuntimeDyld
    Support
    Target
    TargetParser
    MC
  LINK_LIBS PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    MLIRReussir
    MLIRReussirConversion
    MLIRReussirTransformation
    ReussirLLVMAllocationSimplicationPass
    ReussirLLVMLinearRecurrencePass
    spdlog::spdlog
    ${TPDE_LINK_LIB}
)

if (LLVM_VERSION VERSION_GREATER_EQUAL 22)
  target_link_libraries(MLIRReussirBridge
    PRIVATE
      MLIRRegisterAllExtensions
  )
else()
  message(STATUS "LLVM_VERSION is ${LLVM_VERSION}, skipping MLIRRegisterAllExtensions")
endif()

file(GLOB_RECURSE module_files "*.cppm")
target_sources(MLIRReussirBridge
  PUBLIC FILE_SET CXX_MODULES FILES ${module_files} 
)

# Define REUSSIR_HAS_TPDE for the Bridge library
if(REUSSIR_HAS_TPDE)
  target_compile_definitions(MLIRReussirBridge PRIVATE REUSSIR_HAS_TPDE)
endif()

# Nix scan-deps workaround, spdlog/fmt edition (see the NIX_STORE branch in
# the top-level CMakeLists): on darwin, spdlog's and fmt's interface include
# dirs coincide with the nix cc wrapper's implicit dirs, so CMake elides
# their -I flags — the wrapped compiler still finds the headers through its
# environment, but clang-scan-deps (the raw binary) does not and the module
# scan fails with "spdlog/spdlog.h file not found". Raw compile options are
# never elided. Kept out of the top level because find_package(spdlog) there
# would leak SPDLOG_FMT_EXTERNAL into tpde's vendored spdlog subproject.
if(DEFINED ENV{NIX_STORE})
  get_target_property(REUSSIR_SPDLOG_INCS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
  set(REUSSIR_SCANDEP_INCS ${REUSSIR_SPDLOG_INCS})
  if(TARGET fmt::fmt)
    get_target_property(REUSSIR_FMT_INCS fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND REUSSIR_SCANDEP_INCS ${REUSSIR_FMT_INCS})
  endif()
  foreach(inc IN LISTS REUSSIR_SCANDEP_INCS)
    target_compile_options(MLIRReussirBridge PRIVATE "-I${inc}")
  endforeach()
endif()
