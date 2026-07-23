add_library(ReussirLLVMLinearRecurrencePass STATIC
  RecursionLinearization.cpp
  LinearRecurrenceMatExp.cpp
)

set_target_properties(ReussirLLVMLinearRecurrencePass PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
