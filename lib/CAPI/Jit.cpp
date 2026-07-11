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
#include <llvm/Transforms/IPO/LowerTypeTests.h>
#include <llvm/Transforms/IPO/WholeProgramDevirt.h>

#include "Reussir/LLVMPass/AllocationSimplication.h"
#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"

#ifdef REUSSIR_HAS_TPDE
#include <cstdint>
#include <vector>

#include <tpde-llvm/LLVMCompiler.hpp>
#endif

void reussirRunBackendLLVMPipeline(LLVMModuleRef module,
                                   ReussirJitOptLevel opt) {
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
  // Whole-program devirtualization of closures, ahead of the module pipeline
  // so devirtualized slot calls can inline. The lowering tagged each closure
  // vtable with its return-type family id and asserted it — on the very
  // vtable-pointer SSA value the slot call hangs off, which is all WPD's
  // def-use call-site discovery needs — at every indirect eval/clone/drop
  // site. `DevirtSpeculatively` makes each single-implementation fold
  // GUARDED: compare the loaded function pointer against the candidate,
  // direct call (which inlines) on the likely arm, the original indirect
  // call as the fallback. That is sound on any world — a vtable this module
  // has never seen simply takes the fallback — and structurally skips branch
  // funnels and virtual const prop, neither of which survives instruction
  // selection outside the LTO pipelines. Only added when the module actually
  // carries type tests; modules lowered without `closure-wpd` keep their
  // pipeline unchanged. This entry point serves both backends: the AOT
  // compiler (`rrc`) and the JIT/REPL run their LLVM leg through here.
  if (llvm::Intrinsic::getDeclarationIfExists(&m, llvm::Intrinsic::type_test))
    mpm.addPass(llvm::WholeProgramDevirtPass(
        /*ExportSummary=*/nullptr, /*ImportSummary=*/nullptr,
        /*DevirtSpeculatively=*/true));
  // Then consume the artifacts (`llvm.type.test` + `assume`): drop the
  // assumes and fold the dead tests away so no `llvm.type.test` ever reaches
  // instruction selection. A no-op for modules without type tests.
  mpm.addPass(llvm::LowerTypeTestsPass(
      /*ExportSummary=*/nullptr, /*ImportSummary=*/nullptr,
      llvm::lowertypetests::DropTestKind::Assume));
  mpm.addPass(pb.buildPerModuleDefaultPipeline(level));
  mpm.addPass(reussir::llvmpass::AllocationSimplicationPass());
  mpm.run(m, mam);
}

#ifdef REUSSIR_HAS_TPDE
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
#endif // REUSSIR_HAS_TPDE

LLVMMemoryBufferRef
reussirTpdeCompileToObject(LLVMModuleRef module, const char *dataLayout,
                           const char *triple,
                           int stripInvariantGroupBarriersFlag) {
#ifdef REUSSIR_HAS_TPDE
  llvm::Module &m = *llvm::unwrap(module);
  // Caller-guarded: the rewrite mutates the (borrowed) module, so callers
  // that may reuse it with the real LLVM backend keep it intact.
  if (stripInvariantGroupBarriersFlag)
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
  (void)stripInvariantGroupBarriersFlag;
  return nullptr;
#endif
}
