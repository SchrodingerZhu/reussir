# C API libraries consumed by the Rust backend crates (mlir-sys / melior).
#
# The distro LLVM 22 packages provide the MLIR C API only as static archives
# (libMLIRCAPI*.a) and never as the libMLIR-C.so that mlir-sys links in shared
# mode. We therefore fabricate libMLIR-C.so ourselves by whole-archiving those
# archives against the monolithic system libMLIR.so, and ship the Reussir
# dialect registration in a sibling libReussirCAPI.so. Both reference the single
# system libMLIR.so so the dialect registry and MLIR runtime stay unified.

set(REUSSIR_LLVM_LIBDIR "${REUSSIR_LLVM_PREFIX}/lib")
set(REUSSIR_SYSTEM_MLIR "${REUSSIR_LLVM_LIBDIR}/libMLIR.so")
set(REUSSIR_SYSTEM_LLVM "${REUSSIR_LLVM_LIBDIR}/libLLVM.so")

file(GLOB REUSSIR_MLIR_CAPI_ARCHIVES "${REUSSIR_LLVM_LIBDIR}/libMLIRCAPI*.a")
if(NOT REUSSIR_MLIR_CAPI_ARCHIVES)
  message(FATAL_ERROR
    "No libMLIRCAPI*.a archives found in ${REUSSIR_LLVM_LIBDIR}; "
    "the MLIR 22 development package appears to be missing the C API.")
endif()

# --- libMLIR-C.so: fabricated upstream MLIR C API -------------------------------
add_library(ReussirMLIRC SHARED MlirCApiShim.cpp)
set_target_properties(ReussirMLIRC PROPERTIES
  OUTPUT_NAME MLIR-C
  BUILD_RPATH "${REUSSIR_LLVM_LIBDIR}"
  INSTALL_RPATH "${REUSSIR_LLVM_LIBDIR}")
target_link_options(ReussirMLIRC PRIVATE
  "LINKER:--whole-archive" ${REUSSIR_MLIR_CAPI_ARCHIVES} "LINKER:--no-whole-archive")
target_link_libraries(ReussirMLIRC PRIVATE
  ${REUSSIR_SYSTEM_MLIR} ${REUSSIR_SYSTEM_LLVM})

# --- libReussirCAPI.so: Reussir dialect registration ---------------------------
add_library(ReussirCAPI SHARED Dialects.cpp)
add_dependencies(ReussirCAPI MLIRReussir)
set_target_properties(ReussirCAPI PROPERTIES
  BUILD_RPATH "${REUSSIR_LLVM_LIBDIR}"
  INSTALL_RPATH "${REUSSIR_LLVM_LIBDIR}")
target_link_options(ReussirCAPI PRIVATE
  "LINKER:--whole-archive" "$<TARGET_FILE:MLIRReussir>" "LINKER:--no-whole-archive")
target_link_libraries(ReussirCAPI PRIVATE
  ${REUSSIR_SYSTEM_MLIR} ${REUSSIR_SYSTEM_LLVM})

# Aggregate target the Rust crates depend on.
add_custom_target(reussir-capi DEPENDS ReussirMLIRC ReussirCAPI)
