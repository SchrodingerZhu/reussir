add_mlir_library(MLIRReussirDefaultInliner
  DefaultInliner.cpp

  DEPENDS
  MLIRReussirTransformationPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRTransforms
)
