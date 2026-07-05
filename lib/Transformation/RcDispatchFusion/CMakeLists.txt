add_mlir_library(MLIRReussirRcDispatchFusion
  RcDispatchFusion.cpp

  DEPENDS
  MLIRReussirTransformationPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRFuncDialect
  MLIRPass
)
