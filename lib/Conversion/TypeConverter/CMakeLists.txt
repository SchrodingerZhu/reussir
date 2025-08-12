add_mlir_conversion_library(MLIRReussirTypeConverter
  TypeConverter.cpp

  DEPENDS
  MLIRReussirConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRReussir
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRLLVMDialect
  MLIRArithAttrToLLVMConversion
)
