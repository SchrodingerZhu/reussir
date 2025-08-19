//===-- ReussirOps.cpp - Reussir backend bridge -----------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the bridge between rust frontend and C++ backend.
//===----------------------------------------------------------------------===//

#include "Reussir/Bridge.h"
#include "Reussir/IR/ReussirDialect.h"

// MLIR
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Import.h>

// LLVM (native target + data layout)
#include <llvm/ADT/Twine.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>

#include <optional>
#include <string>

using namespace mlir;

namespace reussir {
namespace {
void logIfNeeded(void (*logger)(std::string_view, LogLevel), LogLevel lvl,
                 llvm::Twine msg) {
  if (logger)
    logger(msg.str(), lvl);
}
llvm::CodeGenOptLevel toLlvmOptLevel(OptOption opt) {
  switch (opt) {
  case OptOption::None:
    return llvm::CodeGenOptLevel::None;
  case OptOption::Default:
    return llvm::CodeGenOptLevel::Default;
  case OptOption::Aggressive:
    return llvm::CodeGenOptLevel::Aggressive;
  case OptOption::Size:
    return llvm::CodeGenOptLevel::Less;
  }
}
} // namespace

void compileForNativeMachine(std::string_view mlirTextureModule,
                             std::string_view sourceName,
                             std::string_view outputFile,
                             CompileOptions options) {
  (void)outputFile; // not used in this scaffold stage

  // Initialize native target so we can query TargetMachine for layout/triple.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // 1) Build a registry and MLIR context with required dialects.
  DialectRegistry registry;
  registry.insert<reussir::ReussirDialect, arith::ArithDialect, ub::UBDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                  cf::ControlFlowDialect, LLVM::LLVMDialect>();

  MLIRContext context(registry);
  context.allowUnregisteredDialects(); // keep permissive for early scaffolding

  // 2) Parse the incoming MLIR module from string.
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(mlirTextureModule, &context, sourceName);

  if (!module) {
    logIfNeeded(options.backend_log, LogLevel::Error,
                "Failed to parse MLIR module from provided string.");
    return;
  }
  logIfNeeded(options.backend_log, LogLevel::Info,
              "Parsed MLIR module successfully.");

  // 3) Query native target triple, CPU and features via LLVM C API, then
  //    create an LLVM TargetMachine to derive the data layout string.
  std::string triple = llvm::sys::getDefaultTargetTriple();

  llvm::StringRef cpu = llvm::sys::getHostCPUName();
  llvm::StringMap<bool> featuresMap = llvm::sys::getHostCPUFeatures();

  llvm::SubtargetFeatures features;
  for (const auto &[str, enable] : featuresMap)
    features.AddFeature(str, enable);
  std::string featuresStr = features.getString();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    logIfNeeded(options.backend_log, LogLevel::Error,
                llvm::Twine("LLVM target lookup failed: ") + error);
    return;
  }

  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, cpu, featuresStr, targetOptions, std::nullopt, std::nullopt,
      toLlvmOptLevel(options.opt)));

  if (!tm) {
    logIfNeeded(options.backend_log, LogLevel::Error,
                "Failed to create LLVM TargetMachine.");
    return;
  }

  const llvm::DataLayout dl = tm->createDataLayout();
  const std::string dataLayout = dl.getStringRepresentation();
  mlir::DataLayoutSpecInterface dataLayoutSpec =
      mlir::translateDataLayout(dl, &context);

  Operation *op = module->getOperation();
  op->setAttr("dlti.dl_spec", dataLayoutSpec);

  logIfNeeded(options.backend_log, LogLevel::Debug,
              llvm::Twine("Host triple: ") + triple);
  logIfNeeded(options.backend_log, LogLevel::Debug,
              llvm::Twine("CPU: ") + cpu + ", features: " + featuresStr);
  logIfNeeded(options.backend_log, LogLevel::Debug,
              llvm::Twine("Data layout: ") + dataLayout);

  // Remaining lowering/codegen will be added later.
}

} // namespace reussir
