//===-- Bridge.cpp - Reussir backend bridge ---------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the bridge between rust frontend and C++ backend.
//===----------------------------------------------------------------------===//

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/IPO/Inliner.h>

#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Utils.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <string>

#include "Reussir/Bridge.h"
#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/SCFOpsLowering.h"
#include "Reussir/IR/ReussirDialect.h"

using namespace mlir;

namespace reussir {
namespace {
void logIfNeeded(const CompileOptions &options, LogLevel lvl, llvm::Twine msg) {
  if (options.backendLog && lvl <= options.logLevel)
    options.backendLog(msg.str(), lvl);
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
  llvm_unreachable("unknown optimization level");
}
void createLoweringPipeline(mlir::PassManager &pm) {
  pm.addPass(reussir::createReussirSCFOpsLoweringPass());
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(createSCFToControlFlowPass());
#else
  pm.addPass(createConvertSCFToCFPass());
#endif
  pm.addPass(createReussirBasicOpsLoweringPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}
void runNPMOptimization(llvm::Module &llvmModule,
                        const CompileOptions &options) {
  if (options.opt == OptOption::None) {
    return;
  }

  // Initialize PassBuilder without TargetMachine
  llvm::PassBuilder pb;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Register all analysis managers
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Configure optimization level
  llvm::OptimizationLevel optLevel;
  switch (options.opt) {
  case OptOption::None:
    return; // Already handled above
  case OptOption::Default:
    optLevel = llvm::OptimizationLevel::O2;
    break;
  case OptOption::Aggressive:
    optLevel = llvm::OptimizationLevel::O3;
    break;
  case OptOption::Size:
    optLevel = llvm::OptimizationLevel::Os;
    break;
  }

  // Create the default optimization pipeline for the specified level
  llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);

  // Run the optimization
  mpm.run(llvmModule, mam);
  logIfNeeded(options, LogLevel::Info, "Applied NPM optimization passes.");
}
void emitModule(llvm::Module &llvmModule, llvm::TargetMachine &tm,
                std::string_view filename, const CompileOptions &options) {
  std::error_code ec;
  llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    logIfNeeded(options, LogLevel::Error,
                llvm::Twine("Could not open file: ") + ec.message());
    return;
  }
  llvm::legacy::PassManager pass;
  if (tm.addPassesToEmitFile(pass, dest, nullptr,
                             options.target == OutputTarget::ASM
                                 ? llvm::CodeGenFileType::AssemblyFile
                                 : llvm::CodeGenFileType::ObjectFile)) {
    logIfNeeded(options, LogLevel::Error,
                "TargetMachine can't emit this file type");
    return;
  }

  pass.run(llvmModule);
  dest.flush();
}
} // namespace

void compileForNativeMachine(std::string_view mlirTextureModule,
                             std::string_view sourceName,
                             std::string_view outputFile,
                             CompileOptions options) {
  // Initialize native target so we can query TargetMachine for layout/triple.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  logIfNeeded(options, LogLevel::Info, "Initialized native target.");

  // 1) Build a registry and MLIR context with required dialects.
  DialectRegistry registry;
  registry.insert<reussir::ReussirDialect, DLTIDialect, LLVM::LLVMDialect,
                  arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                  ub::UBDialect, func::FuncDialect, cf::ControlFlowDialect>();
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  logIfNeeded(options, LogLevel::Info, "Loaded all available dialects.");

  // 2) Parse the incoming MLIR module from string.
  llvm::SourceMgr sourceMgr;
  auto buffer =
      llvm::MemoryBuffer::getMemBufferCopy(mlirTextureModule, sourceName);
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module) {
    logIfNeeded(options, LogLevel::Error,
                "Failed to parse MLIR module from provided string.");
    return;
  }
  logIfNeeded(options, LogLevel::Info, "Parsed MLIR module successfully.");

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
    logIfNeeded(options, LogLevel::Error,
                llvm::Twine("LLVM target lookup failed: ") + error);
    return;
  }

  llvm::TargetOptions targetOptions;
  auto tm = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, cpu, featuresStr, targetOptions, std::nullopt, std::nullopt,
      toLlvmOptLevel(options.opt)));

  if (!tm) {
    logIfNeeded(options, LogLevel::Error,
                "Failed to create LLVM TargetMachine.");
    return;
  }

  const llvm::DataLayout dl = tm->createDataLayout();

  module->getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(&context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(dl, &context);
  module->getOperation()->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                                  dlSpec);

  logIfNeeded(options, LogLevel::Debug, llvm::Twine("Host triple: ") + triple);
  logIfNeeded(options, LogLevel::Debug,
              llvm::Twine("CPU: ") + cpu + ", features: " + featuresStr);
  logIfNeeded(options, LogLevel::Debug,
              llvm::Twine("Data layout: ") + dl.getStringRepresentation());

  // Remaining lowering/codegen will be added later.
  mlir::PassManager pm(&context);
  createLoweringPipeline(pm);
  if (pm.run(module->getOperation()).failed()) {
    logIfNeeded(options, LogLevel::Error,
                "Failed to lower MLIR module to LLVM dialect.");
    return;
  }
  logIfNeeded(options, LogLevel::Info, "Successfully lowered MLIR module.");

  // 4) Convert the MLIR module to LLVM IR.
  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module->getOperation(), llvmCtx, sourceName);

  if (!llvmModule) {
    logIfNeeded(options, LogLevel::Error,
                "Failed to translate MLIR module to LLVM IR.");
    return;
  }

  // Run NPM optimization passes
  runNPMOptimization(*llvmModule, options);

  if (options.target == OutputTarget::LLVMIR) {
    std::error_code ec;
    llvm::raw_fd_ostream outStream(outputFile, ec, llvm::sys::fs::OF_None);
    if (ec) {
      logIfNeeded(options, LogLevel::Error,
                  llvm::Twine("Failed to open output file: ") + ec.message());
      return;
    }
    llvmModule->print(outStream, nullptr);
    outStream.flush();
    logIfNeeded(options, LogLevel::Info,
                llvm::Twine("Successfully wrote LLVM IR to output file: ") +
                    outputFile);
    return;
  }
  emitModule(*llvmModule, *tm, outputFile, options);
}

} // namespace reussir
