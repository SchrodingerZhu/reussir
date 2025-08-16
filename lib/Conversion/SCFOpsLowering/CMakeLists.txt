add_mlir_conversion_library(MLIRReussirSCFOpsLowering
  SCFOpsLowering.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRReussir
  MLIRMathDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTransforms
)
