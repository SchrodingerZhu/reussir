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
/// This file implements the Reussir JIT code generation C API.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Jit.h"

#include <llvm-c/Core.h>

#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/IPO/LowerTypeTests.h>
#include <llvm/Transforms/IPO/WholeProgramDevirt.h>
#include <llvm/Transforms/Utils/AssignGUID.h>
#include <llvm/Transforms/Utils/NameAnonGlobals.h>

#include "Reussir/LLVMPass/AllocationSimplication.h"
#include "Reussir/LLVMPass/LinearRecurrence.h"
#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"
#include "Reussir/LLVMPass/SizeAttributes.h"

#ifdef REUSSIR_HAS_TPDE
#include <cstdint>
#include <vector>

#include <tpde-llvm/LLVMCompiler.hpp>
#endif

void reussirRunBackendLLVMPipeline(LLVMModuleRef module, ReussirJitOptLevel opt,
                                   LLVMTargetMachineRef machine,
                                   ReussirLtoMode lto) {
  if (opt == ReussirJitOptNone || opt == ReussirJitOptTpde)
    return;

  llvm::Module &m = *llvm::unwrap(module);

  // The cost-driven passes (loop/SLP vectorization) consult the
  // TargetTransformInfo the PassBuilder derives from its TargetMachine; a
  // machine-less PassBuilder falls back to the base model, which has no
  // vector registers, so vectorization would silently never fire. The AOT
  // driver hands its machine through (faithful to --target-cpu/-triple);
  // the JIT passes NULL and gets a host machine with the host's features.
  llvm::TargetMachine *tm = reinterpret_cast<llvm::TargetMachine *>(machine);
  std::unique_ptr<llvm::TargetMachine> ownedTm;
  if (!tm) {
    llvm::Triple triple = m.getTargetTriple();
    if (triple.getTriple().empty())
      triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    std::string lookupError;
    if (const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, lookupError)) {
      llvm::SubtargetFeatures features;
      for (const auto &[name, enabled] : llvm::sys::getHostCPUFeatures())
        features.AddFeature(name, enabled);
      ownedTm.reset(target->createTargetMachine(
          triple, llvm::sys::getHostCPUName(), features.getString(),
          llvm::TargetOptions(), std::nullopt));
      tm = ownedTm.get();
    }
  }

  // Stamp the machine's CPU and feature set onto every function definition
  // (mirroring clang's -march handling): the pipeline below vectorizes
  // against this machine, and the emitted IR must declare the features it
  // uses so a downstream compilation of the IR text (e.g. the sanitizer
  // harnesses driving clang over rrc's .ll) selects for the same ISA.
  if (tm) {
    llvm::StringRef cpu = tm->getTargetCPU();
    llvm::StringRef features = tm->getTargetFeatureString();
    for (llvm::Function &f : m) {
      if (f.isDeclaration())
        continue;
      if (!cpu.empty() && !f.hasFnAttribute("target-cpu"))
        f.addFnAttr("target-cpu", cpu);
      if (!features.empty() && !f.hasFnAttribute("target-features"))
        f.addFnAttr("target-features", features);
    }
  }

  llvm::PassBuilder pb(tm);
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
    // clang's -Os shape: stamp `optsize` on every definition and run the O2
    // pipeline, whose cost models read the attribute.
    reussir::llvmpass::stampSizeAttributes(m, /*minSize=*/false);
    level = llvm::OptimizationLevel::O2;
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
  if (llvm::Intrinsic::getDeclarationIfExists(&m, llvm::Intrinsic::type_test)) {
    // WPD builds a module summary, and since LLVM 23 `GlobalValue::getGUID`
    // reads a GUID that must have been ASSIGNED (`!unique_id` metadata,
    // computed from name, linkage and source file) — it no longer hashes the
    // name on the fly. Upstream schedules `AssignGUIDPass` right before its
    // own speculative WPD (`buildModuleOptimizationPipeline`) and at the end
    // of every LTO pre-link pipeline; a hand-built pipeline that runs WPD
    // early must do the same. Without it, release builds dereference an
    // empty optional: silent garbage under libstdc++ (Linux), SIGTRAP under
    // libc++'s hardened mode (macOS, every `-O aggressive` compile). Anonymous
    // globals are named first, exactly as upstream's pre-link sequence does,
    // so the GUIDs derive from their final names.
    mpm.addPass(llvm::NameAnonGlobalPass());
    mpm.addPass(llvm::AssignGUIDPass());
    mpm.addPass(llvm::WholeProgramDevirtPass(
        /*ExportSummary=*/nullptr, /*ImportSummary=*/nullptr,
        /*DevirtSpeculatively=*/true));
  }
  // Then consume the artifacts (`llvm.type.test` + `assume`): drop the
  // assumes and fold the dead tests away so no `llvm.type.test` ever reaches
  // instruction selection. A no-op for modules without type tests.
  mpm.addPass(
      llvm::DropTypeTestsPass(llvm::lowertypetests::DropTestKind::Assume));
  // Linear-recurrence strength reduction hooks into the default pipeline's
  // extension points: recursion linearization at pipeline start (before the
  // inliner tears the recursive shape apart) and the Kitamasa rewrite at
  // vectorizer start (canonicalized loops, but before SLP can fold wider
  // recurrence windows into vector PHIs the matcher cannot see). Only
  // speed-oriented levels opt in — the exponentiation kernel trades code
  // size for time.
  if (opt == ReussirJitOptDefault || opt == ReussirJitOptAggressive)
    reussir::llvmpass::registerLinearRecurrencePipelines(pb);
  // Under LTO the module is only half-compiled here: the pre-link pipeline
  // canonicalizes and cleans up but leaves the whole-program work (inlining
  // across modules, global DCE, the second vectorization round) to the link
  // step, which sees every module at once. Running the per-module pipeline
  // instead would optimize against a keyhole view and then have the linker
  // redo it.
  switch (lto) {
  case ReussirLtoThin:
    mpm.addPass(pb.buildThinLTOPreLinkDefaultPipeline(level));
    break;
  case ReussirLtoFat:
    mpm.addPass(pb.buildLTOPreLinkDefaultPipeline(level));
    break;
  // `ReussirLtoNone`, and any value outside the enum: compile this module as
  // the whole world, since nothing downstream will optimize it further.
  default:
    mpm.addPass(pb.buildPerModuleDefaultPipeline(level));
    break;
  }
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
