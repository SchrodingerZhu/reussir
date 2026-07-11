add_mlir_library(MLIRReussirClosureBetaReduction
  ClosureBetaReduction.cpp

  DEPENDS
  MLIRReussirTransformationPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRFuncDialect
  MLIRPass
)
