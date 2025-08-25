get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRReussirAliasAnalysis
  AliasAnalysis.cpp
  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRSupport
    ${dialect_libs}
)

