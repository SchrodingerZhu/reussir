
# Dialect definition
set(LLVM_TARGET_DEFINITIONS ReussirDialect.td)
mlir_tablegen(ReussirDialect.h.inc -gen-dialect-decls -dialect=reussir)
mlir_tablegen(ReussirDialect.cpp.inc -gen-dialect-defs -dialect=reussir)

# Attribute definitions
set(LLVM_TARGET_DEFINITIONS ReussirAttrs.td)
mlir_tablegen(ReussirEnumAttrs.h.inc -gen-enum-decls)
mlir_tablegen(ReussirEnumAttrs.cpp.inc -gen-enum-defs)
mlir_tablegen(ReussirAttrs.h.inc --gen-attrdef-decls)
mlir_tablegen(ReussirAttrs.cpp.inc -gen-attrdef-defs)

# Operation and types
set(LLVM_TARGET_DEFINITIONS ReussirOps.td)
mlir_tablegen(ReussirOps.h.inc -gen-op-decls)
mlir_tablegen(ReussirOps.cpp.inc -gen-op-defs)
mlir_tablegen(ReussirOpsTypes.h.inc --gen-typedef-decls)
mlir_tablegen(ReussirOpsTypes.cpp.inc --gen-typedef-defs)

add_public_tablegen_target(ReussirOpsIncGen)
