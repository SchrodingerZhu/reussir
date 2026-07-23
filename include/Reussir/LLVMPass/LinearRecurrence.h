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

namespace llvm {
class PassBuilder;
} // namespace llvm

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

/// Hooks both passes into the extension points of a PassBuilder-driven
/// default pipeline: recursion linearization (preceded by mem2reg) at
/// pipeline start — before the inliner tears the recursive shape apart —
/// and the Kitamasa rewrite at vectorizer start, after loop canonicalization
/// (rotation, indvar simplification) but *before* the vectorizers.
///
/// The placement inside the pipeline matters twice over: running after the
/// whole pipeline instead would let SLP vectorize wider recurrence windows
/// (k >= 4) into vector PHIs the affine matcher cannot see, and would leave
/// the emitted kernel without the late pipeline's cleanups. At vectorizer
/// start the rewrite sees canonical scalar loops, an EarlyCSE pass right
/// after it folds the schoolbook squaring's commutative duplicate products,
/// and the remaining late pipeline (instcombine, vectorizers, unrolling)
/// then optimizes the emitted kernel like any other code.
///
/// Call before `buildPerModuleDefaultPipeline`.
void registerLinearRecurrencePipelines(llvm::PassBuilder &passBuilder);

} // namespace reussir::llvmpass
