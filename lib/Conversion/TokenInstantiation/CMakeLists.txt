add_mlir_conversion_library(MLIRReussirTokenInstantiation
  TokenInstantiation.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRSync
  MLIRTransforms
)
