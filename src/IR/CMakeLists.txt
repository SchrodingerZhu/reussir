get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_library(MLIRReussir
  SHARED
    ReussirDialect.cpp
    ReussirAttrs.cpp
    ReussirEnumAttrs.cpp
  DEPENDS
    ReussirOpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
)
