get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_library(MLIRReussir
  SHARED
  DEPENDS
    ReussirOpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
)
file(GLOB MODULE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cppm")

target_sources(MLIRReussir
  PUBLIC
  FILE_SET
    CXX_MODULES
  TYPE
    CXX_MODULES
  FILES
    ${MODULE_FILES}
)


