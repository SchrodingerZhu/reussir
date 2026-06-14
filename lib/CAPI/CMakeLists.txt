# C API library consumed by the Rust backend crates (mlir-sys / melior).
#
# mlir-sys links MLIR and LLVM statically, so the Reussir dialect registration
# and the lowering-pipeline pass factories are shipped as static archives linked
# into the same Rust binary. That keeps a single copy of the MLIR runtime and
# dialect registry: the archives' MLIR references resolve against the same static
# MLIR/LLVM that mlir-sys pulls in. Requires the distro's static MLIR/LLVM
# development packages (including libpolly-<ver>-dev for the static Polly
# archives llvm-config references).
add_library(ReussirCAPI STATIC Dialects.cpp Passes.cpp Types.cpp)

# Reussir component archives whose symbols ReussirCAPI references: per-pass
# factories plus the standalone helpers (compilePolymorphicFFI via the
# CompilePolymorphicFFI pass + ReussirRustCompiler, gatherCompiledModules via
# MLIRReussir). reussir-backend-sys links these into the final Rust binary; here
# we only ensure they are built first.
set(REUSSIR_BACKEND_ARCHIVES
  MLIRReussir
  MLIRReussirAnalysis
  ReussirRustCompiler
  # conversions
  MLIRReussirTypeConverter
  MLIRReussirBasicOpsLowering
  MLIRReussirSCFOpsLowering
  MLIRReussirRegionPatterns
  MLIRReussirAcquireDropExpansion
  MLIRReussirRcDecrementExpansion
  MLIRReussirTokenInstantiation
  MLIRReussirClosureOutlining
  MLIRReussirCompilePolymorphicFFI
  MLIRReussirAttachNativeTarget
  # transformations
  MLIRReussirRcCreateSink
  MLIRReussirRcCreateFusion
  MLIRReussirInferVariantTag
  MLIRReussirIncDecCancellation
  MLIRReussirTokenReuse
  MLIRReussirInvariantGroupAnalysis
  MLIRReussirUniqueCarryingRecursionAnalysis
  MLIRReussirTRMCRecursionAnalysis)
add_dependencies(ReussirCAPI ${REUSSIR_BACKEND_ARCHIVES})

# Aggregate target the Rust crates depend on.
add_custom_target(reussir-capi DEPENDS ReussirCAPI ${REUSSIR_BACKEND_ARCHIVES})
