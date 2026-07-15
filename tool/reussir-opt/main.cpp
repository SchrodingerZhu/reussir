//===-- main.cpp - Reussir optimization driver main -------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/Transformation/Passes.h"

#include "Sync/Conversion/ConvertSyncToLLVM.h"
#include "Sync/Conversion/Passes.h"
#include "Sync/IR/SyncDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<reussir::ReussirDialect>();
  registry.insert<mlir::sync::SyncDialect>();
  reussir::registerReussirBasicOpsLoweringInterface(registry);
  mlir::sync::registerConvertSyncToLLVMInterface(registry);
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::sync::createConvertSyncToSTDPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::sync::createConvertSyncToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirBasicOpsLoweringPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirSCFOpsLoweringPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirConvertToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirClosureBetaReductionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirRcCreateSinkPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirRcCreateFusionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirRcDispatchFusionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirRegionPatternsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirAcquireDropExpansionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirRcDecrementExpansionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirInferVariantTagPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirIncDecCancellationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirTokenInstantiationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirCompilePolymorphicFFIPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirClosureOutliningPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirTokenReusePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirSpecialPointerTagPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirAttachNativeTargetPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirInvariantGroupAnalysisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirUniqueCarryingRecursionAnalysisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return reussir::createReussirTRMCRecursionAnalysisPass();
  });
  return failed(mlir::MlirOptMain(
      argc, argv, "Reussir analysis and optimization driver\n", registry));
}
