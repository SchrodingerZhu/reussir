//===-- Passes.cpp - Reussir backend pipeline C API ----------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Passes.h"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/Error.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/Passes.h"
#include "Reussir/Conversion/SCFOpsLowering.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Transformation/Passes.h"
#include "Reussir/Transformation/SpecialPointerTag.h"

using namespace mlir;

namespace {
MlirPass wrapOwned(std::unique_ptr<mlir::Pass> pass) {
  return wrap(pass.release());
}

// Canonicalizer with region simplification disabled, used as the inliner's
// default pipeline so it matches the C++ backend.
void addCanonicalizerWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
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
MlirPass reussirCreateSpecialPointerTagPass(bool archIndependent) {
  reussir::ReussirSpecialPointerTagPassOptions options;
  options.encoding = archIndependent ? reussir::kSpecialPtrTagImmortal.str()
                                     : reussir::kSpecialPtrTagTBI.str();
  return wrapOwned(reussir::createReussirSpecialPointerTagPass(options));
}

MlirPass reussirCreateRcCreateSinkPass(void) {
  return wrapOwned(reussir::createReussirRcCreateSinkPass());
}
MlirPass reussirCreateRcDispatchFusionPass(void) {
  return wrapOwned(reussir::createReussirRcDispatchFusionPass());
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
MlirPass reussirCreateDebugInfoConversionPass(void) {
  return wrapOwned(reussir::createReussirDebugInfoConversionPass());
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
  return wrapOwned(reussir::createReussirConvertToLLVMPass());
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

void reussirFixupVariantDebugInfo(LLVMModuleRef module) {
  reussir::fixupVariantDebugInfo(*llvm::unwrap(module));
}

int reussirHasTPDE(void) {
#ifdef REUSSIR_HAS_TPDE
  return 1;
#else
  return 0;
#endif
}

bool reussirModuleAttachTargetSpec(MlirModule module, const char *dataLayout,
                                   const char *triple) {
  mlir::ModuleOp moduleOp = unwrap(module);
  mlir::MLIRContext *context = moduleOp.getContext();
  // The spec attribute belongs to the DLTI dialect, which the Rust-side
  // context does not otherwise load.
  context->getOrLoadDialect<mlir::DLTIDialect>();
  llvm::Expected<llvm::DataLayout> dl = llvm::DataLayout::parse(dataLayout);
  if (!dl) {
    moduleOp.emitError("invalid data layout string: ")
        << llvm::toString(dl.takeError());
    return false;
  }
  moduleOp->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(context, dataLayout));
  moduleOp->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(context, triple));
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(*dl, context);
  // `translateDataLayout` emits no `index` entry, and without one MLIR's
  // `DataLayout` reports a 64-bit index on every target. Append the entry at
  // the pointer width so index-type queries (and data-layout-aware upstream
  // passes) agree with the target.
  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries(
      dlSpec.getEntries());
  entries.push_back(mlir::DataLayoutEntryAttr::get(
      mlir::IndexType::get(context),
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                             dl->getPointerSizeInBits(0))));
  moduleOp->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                    mlir::DataLayoutSpecAttr::get(context, entries));
  return true;
}
