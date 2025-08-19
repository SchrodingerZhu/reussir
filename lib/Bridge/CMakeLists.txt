add_mlir_library(MLIRReussirBridge
  SHARED
  ReussirBridge.cpp
  DEPENDS
    MLIRReussir
    MLIRReussirBasicOpsLowering
    MLIRReussirSCFOpsLowering
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
)
