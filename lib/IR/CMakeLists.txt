get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

file(GLOB source_files "*.cpp")

add_llvm_library(MLIRReussir
    ${source_files}
  DEPENDS
    ReussirOpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
)


