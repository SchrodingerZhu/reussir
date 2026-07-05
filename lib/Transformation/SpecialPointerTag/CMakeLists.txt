add_mlir_library(MLIRReussirSpecialPointerTag
  SpecialPointerTag.cpp

  DEPENDS
  MLIRReussirTransformationPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRFuncDialect
  MLIRPass
)
