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
/// Extension-point wiring for the linear-recurrence passes (see the header
/// for the placement rationale).
///
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/LinearRecurrence.h"

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

namespace reussir::llvmpass {

void registerLinearRecurrencePipelines(llvm::PassBuilder &passBuilder) {
  passBuilder.registerPipelineStartEPCallback(
      [](llvm::ModulePassManager &mpm, llvm::OptimizationLevel) {
        // The matcher requires SSA scalars (no allocas/loads); promote
        // memory first in case the frontend emitted stack slots.
        mpm.addPass(
            llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass()));
        mpm.addPass(RecursionLinearizationPass());
      });
  passBuilder.registerVectorizerStartEPCallback(
      [](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel) {
        fpm.addPass(llvm::LoopSimplifyPass());
        fpm.addPass(LinearRecurrenceMatExpPass());
        // The schoolbook polynomial squaring emits commutative duplicate
        // products (b_i*b_j and b_j*b_i); EarlyCSE folds them before the
        // late pipeline takes over.
        fpm.addPass(llvm::EarlyCSEPass());
        fpm.addPass(llvm::InstCombinePass());
      });
}

} // namespace reussir::llvmpass
