add_library(ReussirLLVMAllocationSimplicationPass STATIC
  AllocationSimplication.cpp
  MustTailMarking.cpp
  RuntimeFunctionAttributor.cpp
)

set_target_properties(ReussirLLVMAllocationSimplicationPass PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
