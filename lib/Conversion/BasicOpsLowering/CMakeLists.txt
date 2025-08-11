add_mlir_conversion_library(MLIRReussirBasicOpsLowering
  BasicOpsLowering.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRReussir
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
)
