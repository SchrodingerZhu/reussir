get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

file(GLOB source_files "*.cpp")

add_mlir_dialect_library(MLIRReussir
    ${source_files}
  DEPENDS
    ReussirOpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    # The dialect's LLVMTranslationDialectInterface (ReussirInterfaces.cpp)
    # emits LLVM IR for the ops/attributes the LLVM dialect cannot express.
    MLIRTargetLLVMIRExport
    # Loaded as a dependent dialect (lock-guarded cell layouts are sync-type
    # layouts); listed explicitly rather than via the dialect-libs glob.
    MLIRSync
)
