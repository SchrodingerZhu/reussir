add_mlir_conversion_library(MLIRReussirRcDecrementExpansion
  RcDecrementExpansion.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRFuncDialect
  MLIRTransforms
)
