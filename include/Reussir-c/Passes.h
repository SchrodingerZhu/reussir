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
MlirPass reussirCreateRcCreateFusionPass(void);
MlirPass reussirCreateTRMCRecursionAnalysisPass(void);
MlirPass reussirCreateCompilePolymorphicFFIPass(bool optimized);
MlirPass reussirCreateInvariantGroupAnalysisPass(void);
MlirPass reussirCreateBasicOpsLoweringPass(void);

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

// Reports whether TPDE support was compiled into the backend.
int reussirHasTPDE(void);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_PASSES_H
