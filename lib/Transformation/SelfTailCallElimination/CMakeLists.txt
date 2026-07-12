add_mlir_library(MLIRReussirSelfTailCallElimination
  SelfTailCallElimination.cpp

  DEPENDS
  MLIRReussirTransformationPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRPass
)
