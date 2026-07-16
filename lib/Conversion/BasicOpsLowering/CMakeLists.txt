add_mlir_conversion_library(MLIRReussirBasicOpsLowering
  BasicOpsLowering.cpp
  CABISignatureConversion.cpp
  DbgInfoConversion.cpp
  VariantDebugInfo.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRConvertToLLVMInterface
  MLIRReussir
  MLIRReussirTypeConverter
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPtrToLLVM
  MLIRTransforms
  MLIRSync
  MLIRSyncConvertSyncToLLVM
  MLIRSyncTypeConverter
)
