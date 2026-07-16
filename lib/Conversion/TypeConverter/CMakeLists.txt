add_mlir_conversion_library(MLIRReussirTypeConverter
  TypeConverter.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRReussir
  MLIRTargetLLVMIRImport
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRLLVMDialect
  MLIRArithAttrToLLVMConversion
  MLIRSync
  MLIRSyncTypeConverter
)
