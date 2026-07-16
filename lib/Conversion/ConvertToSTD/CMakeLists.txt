add_mlir_conversion_library(MLIRReussirConvertToSTD
  ConvertToSTD.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRReussir
  MLIRMathDialect
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRSync
  MLIRSyncConvertSyncToSTD
)
