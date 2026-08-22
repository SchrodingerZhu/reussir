add_mlir_conversion_library(MLIRReussirInstrumentNonlinearFFI
  InstrumentNonlinearFFI.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransforms
)
