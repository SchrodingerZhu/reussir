//===-- Passes.h - Reussir backend pipeline C API --------------*- C -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// C API for assembling the Reussir lowering pipeline from Rust. Each Reussir
// (and required upstream) pass is exposed as an MlirPass factory so the Rust
// backend can build the pipeline step by step with melior's pass manager, plus
// the standalone helpers the pipeline depends on.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_PASSES_H
#define REUSSIR_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Reussir passes
//===----------------------------------------------------------------------===//

MlirPass reussirCreateUniqueCarryingRecursionAnalysisPass(void);
MlirPass reussirCreateTokenInstantiationPass(void);
MlirPass reussirCreateClosureOutliningPass(void);
MlirPass reussirCreateRegionPatternsPass(void);
MlirPass reussirCreateIncDecCancellationPass(void);
MlirPass reussirCreateRcDecrementExpansionPass(void);
MlirPass reussirCreateInferVariantTagPass(void);
MlirPass reussirCreateSCFOpsLoweringPass(void);
MlirPass reussirCreateRcCreateSinkPass(void);

/// Encodes nullary variants of shared rc-boxed enums as tagged pointer
/// immediates (top byte = tag + 1) and stamps the module so the LLVM
/// lowering steers refcount stores away from the dummy boxes. Add only when
/// the scheme is enabled; `archIndependent` selects the encoding: false =
/// `tbi` (top-byte tag, requires hardware top-byte-ignore, aarch64), true =
/// `immortal` (plain dummy address with an immortal refcount, any target).
MlirPass reussirCreateSpecialPointerTagPass(bool archIndependent);
MlirPass reussirCreateRcDispatchFusionPass(void);
MlirPass reussirCreateRcCreateFusionPass(void);
MlirPass reussirCreateTRMCRecursionAnalysisPass(void);
MlirPass reussirCreateCompilePolymorphicFFIPass(bool optimized);
MlirPass reussirCreateInvariantGroupAnalysisPass(void);
MlirPass reussirCreateBasicOpsLoweringPass(void);
MlirPass reussirCreateDebugInfoConversionPass(void);

// Acquire/drop expansion has two phases controlled by these options; the
// pipeline runs it once with both disabled and once with both enabled.
MlirPass reussirCreateAcquireDropExpansionPass(bool expandDecrement,
                                               bool outlineRecord);
// Token reuse can optionally reuse tokens across function calls.
MlirPass reussirCreateTokenReusePass(bool reuseAcrossCall);

//===----------------------------------------------------------------------===//
// Upstream passes used by the pipeline
//===----------------------------------------------------------------------===//

// The default inliner configured with a canonicalizer that does not simplify
// regions, matching the C++ backend pipeline.
MlirPass reussirCreateDefaultInlinerPass(void);
MlirPass reussirCreateCanonicalizerPass(void);
MlirPass reussirCreateCSEPass(void);
MlirPass reussirCreateControlFlowSinkPass(void);
MlirPass reussirCreateSCFToControlFlowPass(void);
MlirPass reussirCreateConvertToLLVMPass(void);
MlirPass reussirCreateReconcileUnrealizedCastsPass(void);

//===----------------------------------------------------------------------===//
// Standalone helpers
//===----------------------------------------------------------------------===//

// Monomorphizes and compiles polymorphic FFI operations in the module. Returns
// true on success.
bool reussirCompilePolymorphicFFI(MlirModule module, bool optimized);

// Gathers the LLVM bitcode modules attached to compiled operations into a
// single LLVM module owned by `context`. Returns NULL on failure; otherwise the
// caller owns the returned module.
LLVMModuleRef reussirGatherCompiledModules(MlirModule module,
                                           LLVMContextRef context,
                                           const char *dataLayout);

// Rewrites the `{ tag, payload-union }` debug type emitted for each enum into a
// real DWARF `DW_TAG_variant_part`, so a debugger shows only the active case.
// Operates in place on the (already debug-info-bearing) LLVM module; a no-op
// when there is no debug info.
void reussirFixupVariantDebugInfo(LLVMModuleRef module);

// Reports whether TPDE support was compiled into the backend.
int reussirHasTPDE(void);

// Stamps `module` with the target's layout facts before the lowering pipeline
// runs: the LLVM data layout string (`llvm.data_layout`), the target triple
// (`llvm.target_triple`), and the translated DLTI spec (`dlti.dl_spec`) that
// MLIR `DataLayout` queries read. Without the spec MLIR falls back to its
// conservative defaults (e.g. `i64` at ABI alignment 4), so every
// size/alignment the pipeline computes — allocation sizes, spill alignments,
// load/store alignment annotations — understates the target. Returns false if
// `dataLayout` does not parse.
bool reussirModuleAttachTargetSpec(MlirModule module, const char *dataLayout,
                                   const char *triple);

// Sets whether compound record members are laid out in packed physical order
// (descending storage alignment) rather than declaration order. On by default.
// This is a whole-compilation layout contract held on the context-loaded
// Reussir dialect, so set it once before the lowering pipeline computes any
// layout. Loads the Reussir dialect into `context` if not already loaded.
void reussirContextSetPackRecordMembers(MlirContext context, bool enable);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_PASSES_H
