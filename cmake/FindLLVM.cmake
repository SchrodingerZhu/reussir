find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)

# Assert that LLVM version is 20+
if(LLVM_PACKAGE_VERSION VERSION_LESS "20.0.0")
  message(FATAL_ERROR "LLVM version 20.0.0 or higher is required")
endif()

