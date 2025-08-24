get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(ReussirAliasAnalysis
  AliasAnalysis.cpp
  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRSupport
    ${dialect_libs}
)

