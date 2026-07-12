//===- MustTailMarking.h - Reussir self-tail-call marking pass -*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This pass guarantees constant stack usage for direct self tail recursion by
// duplicating the return into the recursive call's block and marking the call
// `musttail`.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/IR/PassManager.h>

namespace reussir::llvmpass {

class MustTailMarkingPass : public llvm::PassInfoMixin<MustTailMarkingPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &);
};

} // namespace reussir::llvmpass
