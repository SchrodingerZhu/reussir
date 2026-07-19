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
/// This pass simplifies runtime allocation/deallocation calls when pointer
/// operands are compile-time null.
///
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
