//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the bridge between frontend and backend.
///
//===----------------------------------------------------------------------===//

module;
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/BLAKE3.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
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
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
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
#include <mlir/InitAllExtensions.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/Passes.h>
#include <mutex>
#include <string>
#ifdef REUSSIR_HAS_TPDE
#include <tpde-llvm/LLVMCompiler.hpp>
#endif
#include <vector>

#include <spdlog/spdlog.h>

#include "Reussir/Bridge.h"
#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/Passes.h"
#include "Reussir/Conversion/ConvertToSTD.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/LLVMPass/AllocationSimplication.h"
#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"
#include "Reussir/Transformation/Passes.h"

export module Reussir.Bridge;

using namespace mlir;

namespace reussir {
export namespace bridge {
void setup() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    llvm::EnablePrettyStackTrace();
    llvm::sys::PrintStackTraceOnErrorSignal(/*argv0=*/"");
  });
}
void setSpdlogLevel(ReussirLogLevel level) {
  switch (level) {
  case REUSSIR_LOG_ERROR:
    spdlog::set_level(spdlog::level::err);
    break;
  case REUSSIR_LOG_WARNING:
    spdlog::set_level(spdlog::level::warn);
    break;
  case REUSSIR_LOG_INFO:
    spdlog::set_level(spdlog::level::info);
    break;
  case REUSSIR_LOG_DEBUG:
    spdlog::set_level(spdlog::level::debug);
    break;
  case REUSSIR_LOG_TRACE:
    spdlog::set_level(spdlog::level::trace);
    break;
  }
}
llvm::CodeGenOptLevel toLlvmOptLevel(ReussirOptOption opt) {
  switch (opt) {
  case REUSSIR_OPT_NONE:
    return llvm::CodeGenOptLevel::None;
  case REUSSIR_OPT_DEFAULT:
    return llvm::CodeGenOptLevel::Default;
  case REUSSIR_OPT_AGGRESSIVE:
    return llvm::CodeGenOptLevel::Aggressive;
  case REUSSIR_OPT_SIZE:
    return llvm::CodeGenOptLevel::Less;
  case REUSSIR_OPT_TPDE:
    return llvm::CodeGenOptLevel::Default;
  }
  llvm_unreachable("unknown optimization level");
}

void addCanonicalizerPassWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addPass(mlir::createCanonicalizerPass(config));
}

void createLoweringPipeline(mlir::PassManager &pm,
                            ReussirOptOption opt = REUSSIR_OPT_DEFAULT,
                            bool reuseTokenAcrossCall = false,
                            bool enableInvariantAnalysis = false) {
  if (opt != REUSSIR_OPT_NONE) {
    llvm::StringMap<mlir::OpPassManager> pipelines;
    if (opt == REUSSIR_OPT_AGGRESSIVE)
      pm.addPass(reussir::createReussirUniqueCarryingRecursionAnalysisPass());
    // The default inliner pass adds the canonicalizer pass with the default
    // configuration.
    pm.addPass(mlir::createInlinerPass(
        pipelines, addCanonicalizerPassWithoutRegionSimplification));
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirTokenInstantiationPass());
  pm.addPass(reussir::createReussirClosureOutliningPass());
  pm.addPass(reussir::createReussirRegionPatternsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirIncDecCancellationPass());
  pm.addPass(reussir::createReussirRcDecrementExpansionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirInferVariantTagPass());
  pm.addPass(reussir::createReussirAcquireDropExpansionPass());
  pm.addPass(reussir::createReussirConvertToSTDPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirIncDecCancellationPass());

  reussir::ReussirAcquireDropExpansionPassOptions options;
  options.expandDecrement = true;
  options.outlineRecord = true;
  pm.addPass(reussir::createReussirAcquireDropExpansionPass(options));

  {
    reussir::ReussirTokenReusePassOptions tokenReuseOpts;
    tokenReuseOpts.reuseAcrossCall = reuseTokenAcrossCall;
    pm.addNestedPass<mlir::func::FuncOp>(
        reussir::createReussirTokenReusePass(tokenReuseOpts));
  }
  pm.addPass(reussir::createReussirConvertToSTDPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirRcCreateSinkPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      reussir::createReussirRcCreateFusionPass());
  pm.addPass(reussir::createReussirTRMCRecursionAnalysisPass());
  pm.addPass(reussir::createReussirCompilePolymorphicFFIPass());

  if (enableInvariantAnalysis) {
    pm.addNestedPass<mlir::func::FuncOp>(
        reussir::createReussirInvariantGroupAnalysisPass());
  }
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createControlFlowSinkPass());
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(createSCFToControlFlowPass());
#else
  pm.addPass(createConvertSCFToCFPass());
#endif
  pm.addPass(createReussirBasicOpsLoweringPass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void runNPMOptimization(llvm::Module &llvmModule, ReussirOptOption opt) {
  if (opt == REUSSIR_OPT_NONE || opt == REUSSIR_OPT_TPDE)
    return;

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
  switch (opt) {
  case REUSSIR_OPT_NONE:
    return; // Already handled above
  case REUSSIR_OPT_DEFAULT:
    optLevel = llvm::OptimizationLevel::O2;
    break;
  case REUSSIR_OPT_AGGRESSIVE:
    optLevel = llvm::OptimizationLevel::O3;
    break;
  case REUSSIR_OPT_SIZE:
    optLevel = llvm::OptimizationLevel::Os;
    break;
  case REUSSIR_OPT_TPDE:
    return;
  }

  // Create the default optimization pipeline for the specified level.
  llvm::ModulePassManager mpm;
  mpm.addPass(reussir::llvmpass::RuntimeFunctionAttributorPass());
  mpm.addPass(pb.buildPerModuleDefaultPipeline(optLevel));
  mpm.addPass(reussir::llvmpass::AllocationSimplicationPass());

  // Run the optimization
  mpm.run(llvmModule, mam);
  spdlog::info("Applied NPM optimization passes.");
}
#ifdef REUSSIR_HAS_TPDE
bool writeBufferToFile(const void *data, size_t size, const char *filename) {
  std::error_code ec;
  llvm::raw_fd_ostream outStream(filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    spdlog::error("Failed to open output file: {}", ec.message());
    return false;
  }
  outStream.write(static_cast<const char *>(data), size);
  outStream.flush();
  return true;
}
#endif
void emitModuleEnv(llvm::Module &llvmModule, llvm::TargetMachine &tm,
                   const char *filename, ReussirOutputTarget target) {
  std::error_code ec;
  llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    spdlog::error("Could not open file: {}", ec.message());
    return;
  }
  llvm::legacy::PassManager pass;
  if (tm.addPassesToEmitFile(pass, dest, nullptr,
                             target == REUSSIR_OUTPUT_ASM
                                 ? llvm::CodeGenFileType::AssemblyFile
                                 : llvm::CodeGenFileType::ObjectFile)) {
    spdlog::error("TargetMachine can't emit this file type");
    return;
  }

  pass.run(llvmModule);
  dest.flush();
}
std::optional<llvm::CodeModel::Model> toCodeModel(ReussirCodeModel model) {
  switch (model) {
  case REUSSIR_CODE_MODEL_TINY:
    return llvm::CodeModel::Tiny;
  case REUSSIR_CODE_MODEL_SMALL:
    return llvm::CodeModel::Small;
  case REUSSIR_CODE_MODEL_KERNEL:
    return llvm::CodeModel::Kernel;
  case REUSSIR_CODE_MODEL_MEDIUM:
    return llvm::CodeModel::Medium;
  case REUSSIR_CODE_MODEL_LARGE:
    return llvm::CodeModel::Large;
  case REUSSIR_CODE_MODEL_DEFAULT:
    return std::nullopt;
  }
  llvm_unreachable("unknown code model");
}

std::optional<llvm::Reloc::Model> toRelocModel(ReussirRelocationModel model) {
  switch (model) {
  case REUSSIR_RELOC_MODEL_STATIC:
    return llvm::Reloc::Static;
  case REUSSIR_RELOC_MODEL_PIC:
    return llvm::Reloc::PIC_;
  case REUSSIR_RELOC_MODEL_DYNAMIC:
    return llvm::Reloc::DynamicNoPIC;
  case REUSSIR_RELOC_MODEL_ROPI:
    return llvm::Reloc::ROPI;
  case REUSSIR_RELOC_MODEL_RWPI:
    return llvm::Reloc::RWPI;
  case REUSSIR_RELOC_MODEL_ROPI_RWPI:
    return llvm::Reloc::ROPI_RWPI;
  case REUSSIR_RELOC_MODEL_DEFAULT:
    return std::nullopt;
  }
  llvm_unreachable("unknown relocation model");
}

std::unique_ptr<mlir::MLIRContext> buildMLIRContext() {
  DialectRegistry registry;
  registry.insert<reussir::ReussirDialect, DLTIDialect, LLVM::LLVMDialect,
                  arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                  ub::UBDialect, func::FuncDialect, cf::ControlFlowDialect>();
  reussir::registerReussirBasicOpsLoweringInterface(registry);
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  mlir::registerAllExtensions(registry);
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

std::unique_ptr<mlir::PassManager> buildPassManager(
    mlir::MLIRContext &context, ReussirOptOption opt = REUSSIR_OPT_DEFAULT,
    bool reuseTokenAcrossCall = false, bool enableInvariantAnalysis = false) {
  auto pm = std::make_unique<mlir::PassManager>(&context);
  createLoweringPipeline(*pm, opt, reuseTokenAcrossCall,
                         enableInvariantAnalysis);
  return pm;
}

std::unique_ptr<llvm::Module>
translateToModule(llvm::StringRef texture, llvm::LLVMContext &llvmCtx,
                  mlir::MLIRContext &context, mlir::PassManager &pm,
                  const llvm::DataLayout &dl, llvm::StringRef targetTriple,
                  llvm::StringRef source_name = "", bool optimizeFFI = false) {
#if LLVM_VERSION_MAJOR >= 21
  // Since LLVM 21.1.0, the MLIR parser does not depend on null terminator.
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(texture, &context, source_name);
#else
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(texture, source_name);
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
#endif
  if (!module) {
    spdlog::error("Failed to parse MLIR module from provided string.");
    return nullptr;
  }
  // add data layout to the module
  module->getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(&context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(dl, &context);
  module->getOperation()->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                                  dlSpec);
  module->getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
      mlir::StringAttr::get(&context, targetTriple));
  // first, compile polymorphic FFI
  if (failed(compilePolymorphicFFI(*module, optimizeFFI))) {
    spdlog::error("Failed to compile polymorphic FFI.");
    return nullptr;
  }
  spdlog::info("Successfully compiled polymorphic FFI.");

  // gather compiled modules
  std::unique_ptr<llvm::Module> compiledModules =
      gatherCompiledModules(*module, llvmCtx, dl.getStringRepresentation());
  if (!compiledModules) {
    spdlog::error("Failed to gather compiled modules.");
    return nullptr;
  }

  spdlog::info("Successfully gathered compiled modules.");
  // run the lowering pipeline on the main module
  if (pm.run(module->getOperation()).failed()) {
    spdlog::error("Failed to lower MLIR module to LLVM dialect.");
    return nullptr;
  }
  spdlog::info("Successfully lowered MLIR module.");
  // finalize as LLVM IR
  auto mainModule =
      translateModuleToLLVMIR(module->getOperation(), llvmCtx, source_name);
  if (!mainModule) {
    spdlog::error("Failed to translate MLIR module to LLVM IR.");
    return nullptr;
  }

  // link the compiled modules to the main module
  if (llvm::Linker::linkModules(*mainModule, std::move(compiledModules))) {
    spdlog::error("Failed to link compiled modules to main module.");
    return nullptr;
  }
  spdlog::info("Successfully linked compiled modules to main module.");
  return mainModule;
}
} // namespace bridge

namespace {
using namespace reussir::bridge;
int reussir_bridge_has_tpde() {
#ifdef REUSSIR_HAS_TPDE
  return 1;
#else
  return 0;
#endif
}

char *reussir_bridge_get_default_target_triple() {
  llvm::InitializeNativeTarget();
  std::string triple = llvm::sys::getDefaultTargetTriple();
  return strdup(triple.c_str());
}

char *reussir_bridge_get_default_target_cpu() {
  llvm::InitializeNativeTarget();
  llvm::StringRef cpu = llvm::sys::getHostCPUName();
  return strdup(cpu.str().c_str());
}

char *reussir_bridge_get_default_target_features() {
  llvm::InitializeNativeTarget();
  llvm::StringMap<bool> featuresMap = llvm::sys::getHostCPUFeatures();

  // Build features string from arrays
  llvm::SubtargetFeatures features;
  for (const auto &[str, enable] : featuresMap) {
    features.AddFeature(str, enable);
  }
  std::string featuresStr = features.getString();
  return strdup(featuresStr.c_str());
}

void reussir_bridge_compile_for_target(
    const char *mlir_module, const char *source_name, const char *output_file,
    ReussirOutputTarget target, ReussirOptOption opt, ReussirLogLevel log_level,
    const char *target_triple, const char *target_cpu,
    const char *target_features, ReussirCodeModel code_model,
    ReussirRelocationModel reloc_model, int reuse_token_across_call,
    int enable_invariant_analysis) {
  bridge::setup();
  setSpdlogLevel(log_level);
  // Initialize native target so we can query TargetMachine for layout/triple.
  // llvm::InitializeNativeTarget();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  spdlog::info("Initialized all targets.");

  // Build a registry and MLIR context with required dialects.
  auto context = buildMLIRContext();
  spdlog::info("Built MLIR context.");

  // Query target triple, CPU and features, then
  //    create an LLVM TargetMachine to derive the data layout string.
  std::string triple = target_triple;
  llvm::StringRef cpu = target_cpu;
#if LLVM_VERSION_MAJOR >= 21
  auto targetTriple = llvm::Triple{llvm::StringRef{triple}};
#else
  llvm::StringRef targetTriple = triple;
#endif
  std::string error;
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!llvmTarget) {
    spdlog::error("LLVM target lookup failed: {}", error);
    return;
  }
  spdlog::info("LLVM target lookup succeeded.");
  llvm::TargetOptions targetOptions;
  auto tm =
      std::unique_ptr<llvm::TargetMachine>(llvmTarget->createTargetMachine(
          targetTriple, cpu, target_features, targetOptions,
          toRelocModel(reloc_model), toCodeModel(code_model),
          toLlvmOptLevel(opt)));

  if (!tm) {
    spdlog::error("Failed to create LLVM TargetMachine.");
    return;
  }
  spdlog::info("LLVM TargetMachine created.");
  const llvm::DataLayout dl = tm->createDataLayout();

  spdlog::debug("Target triple: {}", triple);
  spdlog::debug("CPU: {}, features: {}", cpu.str(), target_features);
  spdlog::debug("Data layout: {}", dl.getStringRepresentation());

  auto pm = buildPassManager(*context, opt, reuse_token_across_call,
                             enable_invariant_analysis != 0);

  // 4) Convert the MLIR module to LLVM IR.
  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateToModule(mlir_module, llvmCtx, *context, *pm, dl, triple,
                        source_name, opt == REUSSIR_OPT_AGGRESSIVE);

  if (!llvmModule) {
    spdlog::error("Failed to translate MLIR module to LLVM IR.");
    return;
  }

  // Run NPM optimization passes
  if (opt != REUSSIR_OPT_TPDE) {
    runNPMOptimization(*llvmModule, opt);

    if (target == REUSSIR_OUTPUT_LLVMIR) {
      std::error_code ec;
      llvm::raw_fd_ostream outStream(output_file, ec, llvm::sys::fs::OF_None);
      if (ec) {
        spdlog::error("Failed to open output file: {}", ec.message());
        return;
      }
      llvmModule->print(outStream, nullptr);
      outStream.flush();
      spdlog::info("Successfully wrote LLVM IR to output file: {}",
                   output_file);
      return;
    }
    emitModuleEnv(*llvmModule, *tm, output_file, target);
  } else {
    // TPDE compilation path
#ifdef REUSSIR_HAS_TPDE
    if (target != REUSSIR_OUTPUT_OBJECT) {
      spdlog::error(
          "TPDE compilation requires Object output target (not ASM or LLVMIR)");
      return;
    }

    auto compiler =
        tpde_llvm::LLVMCompiler::create(llvm::Triple{llvm::StringRef{triple}});
    if (!compiler) {
      spdlog::error("Failed to create TPDE compiler (triple unsupported)");
      return;
    }

    spdlog::info("Starting TPDE compilation.");
    std::vector<uint8_t> buf;
    if (compiler->compile_to_elf(*llvmModule, buf)) {
      // Compilation successful, buf contains object file
      if (writeBufferToFile(buf.data(), buf.size(), output_file)) {
        spdlog::info("Successfully compiled with TPDE to: {}", output_file);
      }
    } else {
      spdlog::error("TPDE compilation failed");
    }
#else
    // TPDE not available, fallback to default optimization pipeline
    spdlog::warn("TPDE optimization requested but not available (it requires "
                 "LLVM < 22 and ELF target). "
                 "Falling back to default optimization level.");
    runNPMOptimization(*llvmModule, opt);

    if (target == REUSSIR_OUTPUT_LLVMIR) {
      std::error_code ec;
      llvm::raw_fd_ostream outStream(output_file, ec, llvm::sys::fs::OF_None);
      if (ec) {
        spdlog::error("Failed to open output file: {}", ec.message());
        return;
      }
      llvmModule->print(outStream, nullptr);
      outStream.flush();
      spdlog::info("Successfully wrote LLVM IR to output file: {}",
                   output_file);
      return;
    }
    emitModuleEnv(*llvmModule, *tm, output_file, target);
#endif
  }
}
} // namespace
} // namespace reussir

// C API wrapper
export extern "C" {
  void reussir_bridge_setup() { reussir::bridge::setup(); }

  int reussir_bridge_has_tpde() { return reussir::reussir_bridge_has_tpde(); }

  char *reussir_bridge_get_default_target_triple() {
    return reussir::reussir_bridge_get_default_target_triple();
  }

  char *reussir_bridge_get_default_target_cpu() {
    return reussir::reussir_bridge_get_default_target_cpu();
  }

  char *reussir_bridge_get_default_target_features() {
    return reussir::reussir_bridge_get_default_target_features();
  }

  void reussir_bridge_compile_for_target(
      const char *mlir_module, const char *source_name, const char *output_file,
      ReussirOutputTarget target, ReussirOptOption opt,
      ReussirLogLevel log_level, const char *target_triple,
      const char *target_cpu, const char *target_features,
      ReussirCodeModel code_model, ReussirRelocationModel reloc_model,
      int reuse_token_across_call, int enable_invariant_analysis) {
    reussir::reussir_bridge_compile_for_target(
        mlir_module, source_name, output_file, target, opt, log_level,
        target_triple, target_cpu, target_features, code_model, reloc_model,
        reuse_token_across_call, enable_invariant_analysis);
  }

  void reussir_bridge_hash_bytes(const uint8_t *str, size_t len,
                                 ReussirStringHash *out) {
    *out = std::bit_cast<ReussirStringHash>(llvm::BLAKE3::hash<32>({str, len}));
  }
}
