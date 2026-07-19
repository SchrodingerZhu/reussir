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
/// This file implements the Reussir LLVM allocation simplification pass.
///
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/AllocationSimplication.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace reussir::llvmpass {
namespace {

bool isNullPointerArgument(llvm::Value *value) {
  if (auto *constant =
          llvm::dyn_cast<llvm::Constant>(value->stripPointerCasts())) {
    return constant->isNullValue();
  }
  return false;
}

} // namespace

llvm::PreservedAnalyses
AllocationSimplicationPass::run(llvm::Module &module,
                                llvm::ModuleAnalysisManager &) {
  llvm::SmallVector<llvm::CallInst *, 8> callsToErase;
  bool changed = false;

  for (auto &function : module) {
    for (auto &block : function) {
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!call)
          continue;

        auto *callee = call->getCalledFunction();
        if (!callee)
          continue;

        llvm::StringRef calleeName = callee->getName();
        if (calleeName == "__reussir_deallocate") {
          if (call->arg_size() >= 1 &&
              isNullPointerArgument(call->getArgOperand(0))) {
            callsToErase.push_back(call);
            changed = true;
          }
          continue;
        }

        if (calleeName == "__reussir_reallocate" && call->arg_size() == 5 &&
            isNullPointerArgument(call->getArgOperand(0))) {
          llvm::IRBuilder<> builder(call);
          llvm::FunctionType *allocateTy =
              llvm::FunctionType::get(call->getType(),
                                      {call->getArgOperand(3)->getType(),
                                       call->getArgOperand(4)->getType()},
                                      false);
          llvm::FunctionCallee allocateFn =
              module.getOrInsertFunction("__reussir_allocate", allocateTy);

          auto *allocateCall = builder.CreateCall(
              allocateTy, allocateFn.getCallee(),
              {call->getArgOperand(3), call->getArgOperand(4)});
          allocateCall->setDebugLoc(call->getDebugLoc());
          allocateCall->setCallingConv(call->getCallingConv());
          allocateCall->setTailCallKind(call->getTailCallKind());

          call->replaceAllUsesWith(allocateCall);
          callsToErase.push_back(call);
          changed = true;
        }
      }
    }
  }

  for (auto *call : callsToErase) {
    call->eraseFromParent();
  }

  if (!changed) {
    return llvm::PreservedAnalyses::all();
  }
  return llvm::PreservedAnalyses::none();
}

} // namespace reussir::llvmpass
