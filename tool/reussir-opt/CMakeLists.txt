get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_tool(reussir-opt
  main.cpp
  DEPENDS
    MLIROptLib
    MLIRReussir
    MLIRReussirConversion
    MLIRReussirTransformation
    MLIRSync
    MLIRSyncConversion
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)

target_link_libraries(reussir-opt
  PRIVATE
    MLIRReussir
    MLIRReussirConversion
    MLIRReussirTransformation
    MLIRSync
    MLIRSyncConversion
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
)

if (LLVM_VERSION VERSION_GREATER_EQUAL 22)
  target_link_libraries(reussir-opt
    PRIVATE
      MLIRRegisterAllDialects
      MLIRRegisterAllExtensions
      MLIRRegisterAllPasses
  )
else()
  message(STATUS "LLVM_VERSION is ${LLVM_VERSION}, skipping All(*) libs")
endif()
