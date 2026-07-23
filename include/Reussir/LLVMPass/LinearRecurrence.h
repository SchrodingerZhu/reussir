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
/// Linear-recurrence oriented LLVM passes.
///
/// `RecursionLinearizationPass` rewrites pure, directly self-recursive
/// functions whose recursion argument decreases by constant offsets (the
/// `f(x) = combine(f(x - d_1), ..., f(x - d_k))` shape, with an arbitrary
/// side-effect-free `combine`) into a dynamic-programming loop over a sliding
/// window of the last `k` results, turning exponential call trees into linear
/// loops.
///
/// `LinearRecurrenceMatExpPass` recognizes single-block loops whose
/// loop-carried state evolves as an affine map over Z/2^N (constant
/// coefficients, wrapping integer arithmetic) and replaces the loop with
/// square-and-multiply exponentiation of the companion matrix, turning linear
/// loops into logarithmic-time computations.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/IR/PassManager.h>

namespace reussir::llvmpass {

class RecursionLinearizationPass
    : public llvm::PassInfoMixin<RecursionLinearizationPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &mam);
};

class LinearRecurrenceMatExpPass
    : public llvm::PassInfoMixin<LinearRecurrenceMatExpPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &fam);
};

} // namespace reussir::llvmpass
