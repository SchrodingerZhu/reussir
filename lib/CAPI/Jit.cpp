//===-- Jit.cpp - Reussir JIT codegen C API -------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Jit.h"

#include <llvm-c/Core.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/TargetParser/Triple.h>

#include "Reussir/LLVMPass/AllocationSimplication.h"
#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"

#ifdef REUSSIR_HAS_TPDE
#include <cstdint>
#include <vector>

#include <tpde-llvm/LLVMCompiler.hpp>
#endif

void reussirRunBackendLLVMPipeline(LLVMModuleRef module, ReussirJitOptLevel opt) {
  if (opt == ReussirJitOptNone || opt == ReussirJitOptTpde)
    return;

  llvm::Module &m = *llvm::unwrap(module);

  llvm::PassBuilder pb;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel level;
  switch (opt) {
  case ReussirJitOptAggressive:
    level = llvm::OptimizationLevel::O3;
    break;
  case ReussirJitOptSize:
    level = llvm::OptimizationLevel::Os;
    break;
  case ReussirJitOptDefault:
  default:
    level = llvm::OptimizationLevel::O2;
    break;
  }

  llvm::ModulePassManager mpm;
  mpm.addPass(reussir::llvmpass::RuntimeFunctionAttributorPass());
  mpm.addPass(pb.buildPerModuleDefaultPipeline(level));
  mpm.addPass(reussir::llvmpass::AllocationSimplicationPass());
  mpm.run(m, mam);
}

LLVMMemoryBufferRef reussirTpdeCompileToObject(LLVMModuleRef module,
                                               const char *dataLayout,
                                               const char *triple) {
#ifdef REUSSIR_HAS_TPDE
  llvm::Module &m = *llvm::unwrap(module);

  // TPDE compiles directly from the module, so it must carry the host data
  // layout and triple (the freshly translated module may have neither).
  if (dataLayout)
    m.setDataLayout(llvm::StringRef(dataLayout));
  llvm::Triple hostTriple(triple ? triple : "");
  m.setTargetTriple(hostTriple);

  std::unique_ptr<tpde_llvm::LLVMCompiler> compiler =
      tpde_llvm::LLVMCompiler::create(hostTriple);
  if (!compiler)
    return nullptr;

  std::vector<uint8_t> object;
  if (!compiler->compile_to_elf(m, object))
    return nullptr;

  return LLVMCreateMemoryBufferWithMemoryRangeCopy(
      reinterpret_cast<const char *>(object.data()), object.size(),
      "reussir-tpde-object");
#else
  (void)module;
  (void)dataLayout;
  (void)triple;
  return nullptr;
#endif
}
