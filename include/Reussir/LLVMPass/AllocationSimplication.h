//===- AllocationSimplication.h - Reussir LLVM allocation pass -*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This pass simplifies runtime allocation/deallocation calls when pointer
// operands are compile-time null.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/IR/PassManager.h>

namespace reussir::llvmpass {

class AllocationSimplicationPass
    : public llvm::PassInfoMixin<AllocationSimplicationPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &);
};

} // namespace reussir::llvmpass
