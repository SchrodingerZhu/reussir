# C API libraries consumed by the Rust backend crates (mlir-sys / melior).
#
# mlir-sys links a shared `MLIR-C` library. Some LLVM 22 distributions ship it
# (libMLIR-C.so / .dylib); others provide the MLIR C API only as static archives
# (libMLIRCAPI*.a). When the shared library is absent we fabricate it by whole-
# archiving those static archives against the monolithic system libMLIR, and ship
# the Reussir dialect registration in a sibling libReussirCAPI. Both reference the
# single system libMLIR so the dialect registry and MLIR runtime stay unified.

set(REUSSIR_LLVM_LIBDIR "${REUSSIR_LLVM_PREFIX}/lib")
set(_shared "${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(REUSSIR_SYSTEM_MLIR "${REUSSIR_LLVM_LIBDIR}/libMLIR${_shared}")
set(REUSSIR_SYSTEM_LLVM "${REUSSIR_LLVM_LIBDIR}/libLLVM${_shared}")

# Emits, into ${out_var}, the linker flags that force every object of the given
# archive files to be included. ELF/GNU uses --whole-archive; mach-o/ld64 uses a
# -force_load per archive.
function(reussir_whole_archive_flags out_var)
  set(_flags "")
  if(APPLE)
    foreach(_archive IN LISTS ARGN)
      list(APPEND _flags "LINKER:-force_load,${_archive}")
    endforeach()
  else()
    list(APPEND _flags "LINKER:--whole-archive" ${ARGN} "LINKER:--no-whole-archive")
  endif()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# --- libMLIR-C: reuse the system library, or fabricate it -----------------------
set(REUSSIR_SYSTEM_MLIRC "${REUSSIR_LLVM_LIBDIR}/libMLIR-C${_shared}")
if(EXISTS "${REUSSIR_SYSTEM_MLIRC}")
  message(STATUS "Using system MLIR C API: ${REUSSIR_SYSTEM_MLIRC}")
  # A placeholder so dependents can still depend on ReussirMLIRC uniformly.
  add_custom_target(ReussirMLIRC)
else()
  file(GLOB REUSSIR_MLIR_CAPI_ARCHIVES "${REUSSIR_LLVM_LIBDIR}/libMLIRCAPI*.a")
  if(NOT REUSSIR_MLIR_CAPI_ARCHIVES)
    message(FATAL_ERROR
      "Neither libMLIR-C${_shared} nor libMLIRCAPI*.a was found in "
      "${REUSSIR_LLVM_LIBDIR}; the MLIR 22 development package is incomplete.")
  endif()
  message(STATUS "Fabricating libMLIR-C${_shared} from static MLIR C API archives")
  reussir_whole_archive_flags(_mlirc_whole_archive ${REUSSIR_MLIR_CAPI_ARCHIVES})

  add_library(ReussirMLIRC SHARED MlirCApiShim.cpp)
  set_target_properties(ReussirMLIRC PROPERTIES
    OUTPUT_NAME MLIR-C
    BUILD_RPATH "${REUSSIR_LLVM_LIBDIR}"
    INSTALL_RPATH "${REUSSIR_LLVM_LIBDIR}")
  target_link_options(ReussirMLIRC PRIVATE ${_mlirc_whole_archive})
  target_link_libraries(ReussirMLIRC PRIVATE
    ${REUSSIR_SYSTEM_MLIR} ${REUSSIR_SYSTEM_LLVM})
endif()

# --- libReussirCAPI: Reussir dialect registration ------------------------------
reussir_whole_archive_flags(_reussir_whole_archive "$<TARGET_FILE:MLIRReussir>")

add_library(ReussirCAPI SHARED Dialects.cpp)
add_dependencies(ReussirCAPI MLIRReussir)
set_target_properties(ReussirCAPI PROPERTIES
  BUILD_RPATH "${REUSSIR_LLVM_LIBDIR}"
  INSTALL_RPATH "${REUSSIR_LLVM_LIBDIR}")
target_link_options(ReussirCAPI PRIVATE ${_reussir_whole_archive})
target_link_libraries(ReussirCAPI PRIVATE
  ${REUSSIR_SYSTEM_MLIR} ${REUSSIR_SYSTEM_LLVM})

# Aggregate target the Rust crates depend on.
add_custom_target(reussir-capi DEPENDS ReussirMLIRC ReussirCAPI)
