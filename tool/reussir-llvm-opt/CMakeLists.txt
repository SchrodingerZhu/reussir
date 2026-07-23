set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Passes
  Support
)

add_llvm_executable(reussir-llvm-opt
  main.cpp
)

target_link_libraries(reussir-llvm-opt
  PRIVATE
    ReussirLLVMAllocationSimplicationPass
    ReussirLLVMLinearRecurrencePass
)
