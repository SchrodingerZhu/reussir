find_file(filecheck_path FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "FileCheck path: ${filecheck_path}")

find_file(not_path not HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "not path: ${not_path}")

find_file(mlir_translate_path mlir-translate HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "mlir-translate path: ${mlir_translate_path}")

find_file(opt_path opt HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "opt path: ${opt_path}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v
  DEPENDS reussir-opt)
