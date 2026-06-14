# C API library consumed by the Rust backend crates (mlir-sys / melior).
#
# mlir-sys links MLIR and LLVM statically, so the Reussir dialect registration
# is shipped as a static archive that is linked into the same Rust binary. That
# keeps a single copy of the MLIR runtime and dialect registry: the archive's
# MLIR references resolve against the same static MLIR/LLVM that mlir-sys pulls
# in. Requires the distro's static MLIR/LLVM development packages (including
# libpolly-<ver>-dev for the static Polly archives llvm-config references).
add_library(ReussirCAPI STATIC Dialects.cpp)
add_dependencies(ReussirCAPI MLIRReussir)

# Aggregate target the Rust crates depend on.
add_custom_target(reussir-capi DEPENDS ReussirCAPI)
