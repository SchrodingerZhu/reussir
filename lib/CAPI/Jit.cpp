//===-- Jit.cpp - Reussir JIT codegen C API -------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Jit.h"

#include <llvm-c/Core.h>

#include <llvm/IR/IntrinsicInst.h>
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

namespace {
/// TPDE has no lowering for the invariant.group barrier intrinsics the
/// frozen/shared payload loads carry (`llvm.launder.invariant.group`,
/// `llvm.strip.invariant.group`). Both are pure optimizer fences that
/// return their pointer operand, so replacing the call with the operand is
/// always conservative-correct — required for e.g. a REPL wrapper that
/// projects a global binding's value out of its shared box.
void stripInvariantGroupBarriers(llvm::Module &m) {
  llvm::SmallVector<llvm::IntrinsicInst *> barriers;
  for (llvm::Function &f : m)
    for (llvm::BasicBlock &bb : f)
      for (llvm::Instruction &inst : bb)
        if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&inst))
          if (intrinsic->getIntrinsicID() ==
                  llvm::Intrinsic::launder_invariant_group ||
              intrinsic->getIntrinsicID() ==
                  llvm::Intrinsic::strip_invariant_group)
            barriers.push_back(intrinsic);
  for (llvm::IntrinsicInst *barrier : barriers) {
    barrier->replaceAllUsesWith(barrier->getArgOperand(0));
    barrier->eraseFromParent();
  }
}
} // namespace

LLVMMemoryBufferRef reussirTpdeCompileToObject(LLVMModuleRef module,
                                               const char *dataLayout,
                                               const char *triple) {
#ifdef REUSSIR_HAS_TPDE
  llvm::Module &m = *llvm::unwrap(module);
  stripInvariantGroupBarriers(m);

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
