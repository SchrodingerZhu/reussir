find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)

# Reussir is locked to the LLVM 22 release line. The Rust backend (mlir-sys /
# melior / tblgen / llvm-sys) is pinned to the 22 ABI, so any other major
# version is rejected outright rather than silently mis-linking.
if(LLVM_PACKAGE_VERSION VERSION_LESS "22.0.0" OR
   NOT LLVM_PACKAGE_VERSION VERSION_LESS "23.0.0")
  message(FATAL_ERROR
    "Reussir requires LLVM 22.x (found ${LLVM_PACKAGE_VERSION})")
endif()

# Normalized install prefix exported for the Rust crates' cargo invocations
# (MLIR_SYS_220_PREFIX / TABLEGEN_220_PREFIX / LLVM_SYS_220_PREFIX). LLVMConfig
# defines LLVM_INSTALL_PREFIX; fall back to the parent of the binary dir.
if(DEFINED LLVM_INSTALL_PREFIX)
  set(REUSSIR_LLVM_PREFIX "${LLVM_INSTALL_PREFIX}")
else()
  get_filename_component(REUSSIR_LLVM_PREFIX "${LLVM_TOOLS_BINARY_DIR}" DIRECTORY)
endif()
set(REUSSIR_LLVM_MAJOR 22)
message(STATUS "Reussir LLVM prefix for Rust crates: ${REUSSIR_LLVM_PREFIX}")

