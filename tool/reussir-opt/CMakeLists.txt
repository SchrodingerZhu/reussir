get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_tool(reussir-opt
  main.cpp
  DEPENDS
    MLIROptLib
    MLIRReussir
    MLIRReussirBasicOpsLowering
    MLIRReussirSCFOpsLowering
    MLIRReussirDropExpansion
    MLIRReussirRcDecrementExpansion
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)

target_link_libraries(reussir-opt 
  PRIVATE 
    MLIRReussir
    MLIRReussirBasicOpsLowering
    MLIRReussirSCFOpsLowering
    MLIRReussirDropExpansion
    MLIRReussirRcDecrementExpansion
    ${dialect_libs} 
    ${conversion_libs} 
    ${extension_libs} 
    MLIROptLib
)
