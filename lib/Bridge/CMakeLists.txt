get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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
    ${conversion_libs}
    ${extension_libs}
    MLIRReussir
    MLIRReussirBasicOpsLowering
    MLIRReussirSCFOpsLowering
)
