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
/// This file implements the Reussir backend pipeline C API.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Passes.h"

#include <llvm/IR/Comdat.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/Triple.h>

#include <chrono>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Pass/PassInstrumentation.h>

// Complete mlir::DialectVersion before MLIR C API headers reach
// BytecodeWriter.h; MSVC's STL rejects destroying unique_ptr<T> when T is only
// forward declared.
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Transform/Transforms/Passes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/ConvertToSTD.h"
#include "Reussir/Conversion/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Transformation/Passes.h"
#include "Reussir/Transformation/SpecialPointerTag.h"

using namespace mlir;

namespace {
MlirPass wrapOwned(std::unique_ptr<mlir::Pass> pass) {
  return wrap(pass.release());
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
MlirPass reussirCreateConvertToSTDPass(void) {
  return wrapOwned(reussir::createReussirConvertToSTDPass());
}
MlirPass reussirCreateSpecialPointerTagPass(bool archIndependent) {
  reussir::ReussirSpecialPointerTagPassOptions options;
  options.encoding = archIndependent ? reussir::kSpecialPtrTagImmortal.str()
                                     : reussir::kSpecialPtrTagTBI.str();
  return wrapOwned(reussir::createReussirSpecialPointerTagPass(options));
}

MlirPass reussirCreateClosureBetaReductionPass(void) {
  return wrapOwned(reussir::createReussirClosureBetaReductionPass());
}
MlirPass reussirCreateRcCreateSinkPass(void) {
  return wrapOwned(reussir::createReussirRcCreateSinkPass());
}
MlirPass reussirCreateRcDispatchFusionPass(void) {
  return wrapOwned(reussir::createReussirRcDispatchFusionPass());
}
MlirPass reussirCreatePartialMovePass(void) {
  return wrapOwned(reussir::createReussirPartialMovePass());
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
MlirPass reussirCreateBasicOpsLoweringPass(bool closureWpd) {
  reussir::ReussirBasicOpsLoweringPassOptions options;
  options.closureWpd = closureWpd;
  return wrapOwned(reussir::createReussirBasicOpsLoweringPass(options));
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
MlirPass reussirCreateTokenReusePass(bool reuseAcrossCall, bool emitRemarks) {
  reussir::ReussirTokenReusePassOptions options;
  options.reuseAcrossCall = reuseAcrossCall;
  options.emitRemarks = emitRemarks;
  return wrapOwned(reussir::createReussirTokenReusePass(options));
}

//===----------------------------------------------------------------------===//
// Upstream passes
//===----------------------------------------------------------------------===//

MlirPass reussirCreateDefaultInlinerPass(void) {
  // The policy (one iteration, canonicalize-without-region-simplify
  // callable optimizer) lives in the pass itself, built programmatically —
  // no option strings to parse and no pass-registry resolution.
  return wrapOwned(reussir::createReussirDefaultInlinerPass());
}
MlirPass reussirCreateTransformInterpreterPass(MlirStringRef entryPoint) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = unwrap(entryPoint).str();
  return wrapOwned(mlir::transform::createInterpreterPass(options));
}
MlirPass reussirCreateTransformPreloadLibraryPass(MlirStringRef const *paths,
                                                  intptr_t nPaths) {
  mlir::transform::PreloadLibraryPassOptions options;
  options.transformLibraryPaths.reserve(nPaths);
  for (intptr_t i = 0; i < nPaths; ++i)
    options.transformLibraryPaths.push_back(unwrap(paths[i]).str());
  return wrapOwned(mlir::transform::createPreloadLibraryPass(options));
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

bool reussirCompilePolymorphicFFI(MlirModule module, bool optimized,
                                  MlirStringRef rustPath,
                                  const MlirStringRef *libDirs,
                                  intptr_t nLibDirs,
                                  MlirStringRef targetTriple) {
  llvm::SmallVector<llvm::StringRef> dirs;
  dirs.reserve(nLibDirs);
  for (intptr_t i = 0; i < nLibDirs; ++i)
    dirs.push_back(unwrap(libDirs[i]));
  return succeeded(reussir::compilePolymorphicFFI(
      unwrap(module), optimized, unwrap(rustPath), dirs, unwrap(targetTriple)));
}

LLVMModuleRef reussirGatherCompiledModules(MlirModule module,
                                           LLVMContextRef context,
                                           const char *dataLayout,
                                           const char *targetTriple) {
  std::unique_ptr<llvm::Module> result = reussir::gatherCompiledModules(
      unwrap(module), *llvm::unwrap(context), dataLayout, targetTriple);
  return llvm::wrap(result.release());
}

void reussirFixupVariantDebugInfo(LLVMModuleRef module) {
  reussir::fixupVariantDebugInfo(*llvm::unwrap(module));
}

void reussirAttachCoffComdats(LLVMModuleRef module) {
  llvm::Module &m = *llvm::unwrap(module);
  // COFF only: Mach-O rejects comdats outright (hard LLVM error), and ELF
  // weak binding dedups without them.
  if (!m.getTargetTriple().isOSBinFormatCOFF())
    return;
  auto attach = [&m](llvm::GlobalObject &go) {
    // A comdat is only valid on a definition, and an existing one (e.g. from
    // linked-in rustc bitcode) already carries the right selection kind.
    if (go.isDeclaration() || go.hasComdat())
      return;
    switch (go.getLinkage()) {
    case llvm::GlobalValue::WeakODRLinkage:
    case llvm::GlobalValue::LinkOnceODRLinkage:
    case llvm::GlobalValue::WeakAnyLinkage:
    case llvm::GlobalValue::LinkOnceAnyLinkage: {
      // COFF requires the comdat symbol to be the section's own leader, so
      // the comdat must be keyed by the symbol's exact name.
      llvm::Comdat *comdat = m.getOrInsertComdat(go.getName());
      comdat->setSelectionKind(llvm::Comdat::Any);
      go.setComdat(comdat);
      break;
    }
    default:
      break;
    }
  };
  for (llvm::Function &f : m.functions())
    attach(f);
  for (llvm::GlobalVariable &g : m.globals())
    attach(g);
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

void reussirContextSetPackRecordMembers(MlirContext context, bool enable) {
  mlir::MLIRContext *ctx = unwrap(context);
  ctx->getOrLoadDialect<reussir::ReussirDialect>()->setPackRecordMembers(
      enable);
}

namespace {
/// Per-pass progress on stderr, for locating a wedged or pathological pass
/// from nothing but a captured log (a CI timeout kills the process before any
/// debugger can attach). Every pass execution prints a `begin` line before it
/// runs and an `end` line (with wall time) after; the last `begin` without a
/// matching `end` names the pass — and the op it was running on — that never
/// came back. Each line is assembled in a local buffer and written with a
/// single `errs()` call so lines from MLIR's parallel per-op drivers do not
/// tear.
class PassPhaseLogger final : public mlir::PassInstrumentation {
public:
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    starts().push_back({pass, std::chrono::steady_clock::now()});
    llvm::SmallString<128> line;
    llvm::raw_svector_ostream os(line);
    os << "[mlir-pass] begin " << pass->getName() << " on " << label(op)
       << "\n";
    llvm::errs() << line;
  }
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    finish(pass, op, /*failed=*/false);
  }
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    finish(pass, op, /*failed=*/true);
  }

private:
  using Clock = std::chrono::steady_clock;
  // Per-thread begin stack: passes nest within a thread (pipelines inside
  // adaptors) and run in parallel across threads, so matching begin to end
  // thread-locally is both necessary and sufficient.
  static llvm::SmallVector<std::pair<mlir::Pass *, Clock::time_point>> &
  starts() {
    thread_local llvm::SmallVector<std::pair<mlir::Pass *, Clock::time_point>>
        stack;
    return stack;
  }
  static std::string label(mlir::Operation *op) {
    std::string text = op->getName().getStringRef().str();
    if (auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      text += " @";
      text += sym.getValue();
    }
    return text;
  }
  void finish(mlir::Pass *pass, mlir::Operation *op, bool failed) {
    int64_t ms = -1;
    auto &stack = starts();
    for (size_t i = stack.size(); i-- > 0;) {
      if (stack[i].first == pass) {
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 Clock::now() - stack[i].second)
                 .count();
        stack.erase(stack.begin() + i);
        break;
      }
    }
    llvm::SmallString<128> line;
    llvm::raw_svector_ostream os(line);
    os << "[mlir-pass] " << (failed ? "FAILED " : "end ") << pass->getName()
       << " on " << label(op);
    if (ms >= 0)
      os << " (" << ms << " ms)";
    os << "\n";
    llvm::errs() << line;
  }
};
} // namespace

void reussirPassManagerAttachPhaseLogger(MlirPassManager pm) {
  unwrap(pm)->addInstrumentation(std::make_unique<PassPhaseLogger>());
}
