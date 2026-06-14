//===-- Passes.cpp - Reussir backend pipeline C API ----------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Passes.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/Passes.h"
#include "Reussir/Conversion/SCFOpsLowering.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Transformation/Passes.h"

using namespace mlir;

namespace {
MlirPass wrapOwned(std::unique_ptr<mlir::Pass> pass) {
  return wrap(pass.release());
}

// Canonicalizer with region simplification disabled, used as the inliner's
// default pipeline so it matches the C++ backend.
void addCanonicalizerWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addPass(mlir::createCanonicalizerPass(config));
}
} // namespace

//===----------------------------------------------------------------------===//
// Reussir passes
//===----------------------------------------------------------------------===//

MlirPass reussirCreateUniqueCarryingRecursionAnalysisPass(void) {
  return wrapOwned(reussir::createReussirUniqueCarryingRecursionAnalysisPass());
}
MlirPass reussirCreateTokenInstantiationPass(void) {
  return wrapOwned(reussir::createReussirTokenInstantiationPass());
}
MlirPass reussirCreateClosureOutliningPass(void) {
  return wrapOwned(reussir::createReussirClosureOutliningPass());
}
MlirPass reussirCreateRegionPatternsPass(void) {
  return wrapOwned(reussir::createReussirRegionPatternsPass());
}
MlirPass reussirCreateIncDecCancellationPass(void) {
  return wrapOwned(reussir::createReussirIncDecCancellationPass());
}
MlirPass reussirCreateRcDecrementExpansionPass(void) {
  return wrapOwned(reussir::createReussirRcDecrementExpansionPass());
}
MlirPass reussirCreateInferVariantTagPass(void) {
  return wrapOwned(reussir::createReussirInferVariantTagPass());
}
MlirPass reussirCreateSCFOpsLoweringPass(void) {
  return wrapOwned(reussir::createReussirSCFOpsLoweringPass());
}
MlirPass reussirCreateRcCreateSinkPass(void) {
  return wrapOwned(reussir::createReussirRcCreateSinkPass());
}
MlirPass reussirCreateRcCreateFusionPass(void) {
  return wrapOwned(reussir::createReussirRcCreateFusionPass());
}
MlirPass reussirCreateTRMCRecursionAnalysisPass(void) {
  return wrapOwned(reussir::createReussirTRMCRecursionAnalysisPass());
}
MlirPass reussirCreateCompilePolymorphicFFIPass(bool optimized) {
  reussir::ReussirCompilePolymorphicFFIPassOptions options;
  options.optimized = optimized;
  return wrapOwned(reussir::createReussirCompilePolymorphicFFIPass(options));
}
MlirPass reussirCreateInvariantGroupAnalysisPass(void) {
  return wrapOwned(reussir::createReussirInvariantGroupAnalysisPass());
}
MlirPass reussirCreateBasicOpsLoweringPass(void) {
  return wrapOwned(reussir::createReussirBasicOpsLoweringPass());
}
MlirPass reussirCreateAcquireDropExpansionPass(bool expandDecrement,
                                               bool outlineRecord) {
  reussir::ReussirAcquireDropExpansionPassOptions options;
  options.expandDecrement = expandDecrement;
  options.outlineRecord = outlineRecord;
  return wrapOwned(reussir::createReussirAcquireDropExpansionPass(options));
}
MlirPass reussirCreateTokenReusePass(bool reuseAcrossCall) {
  reussir::ReussirTokenReusePassOptions options;
  options.reuseAcrossCall = reuseAcrossCall;
  return wrapOwned(reussir::createReussirTokenReusePass(options));
}

//===----------------------------------------------------------------------===//
// Upstream passes
//===----------------------------------------------------------------------===//

MlirPass reussirCreateDefaultInlinerPass(void) {
  llvm::StringMap<mlir::OpPassManager> pipelines;
  return wrapOwned(mlir::createInlinerPass(
      pipelines, addCanonicalizerWithoutRegionSimplification));
}
MlirPass reussirCreateCanonicalizerPass(void) {
  return wrapOwned(mlir::createCanonicalizerPass());
}
MlirPass reussirCreateCSEPass(void) { return wrapOwned(mlir::createCSEPass()); }
MlirPass reussirCreateControlFlowSinkPass(void) {
  return wrapOwned(mlir::createControlFlowSinkPass());
}
MlirPass reussirCreateSCFToControlFlowPass(void) {
  return wrapOwned(mlir::createSCFToControlFlowPass());
}
MlirPass reussirCreateConvertToLLVMPass(void) {
  return wrapOwned(mlir::createConvertToLLVMPass());
}
MlirPass reussirCreateReconcileUnrealizedCastsPass(void) {
  return wrapOwned(mlir::createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Standalone helpers
//===----------------------------------------------------------------------===//

bool reussirCompilePolymorphicFFI(MlirModule module, bool optimized) {
  return succeeded(reussir::compilePolymorphicFFI(unwrap(module), optimized));
}

LLVMModuleRef reussirGatherCompiledModules(MlirModule module,
                                           LLVMContextRef context,
                                           const char *dataLayout) {
  std::unique_ptr<llvm::Module> result = reussir::gatherCompiledModules(
      unwrap(module), *llvm::unwrap(context), dataLayout);
  return llvm::wrap(result.release());
}

int reussirHasTPDE(void) {
#ifdef REUSSIR_HAS_TPDE
  return 1;
#else
  return 0;
#endif
}
