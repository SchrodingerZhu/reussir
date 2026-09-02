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
/// Size-optimization requests follow clang's -Os/-Oz split: every function
/// definition is stamped `optsize` (plus `minsize` for the -Oz flavor), and
/// the caller then builds the plain O2 pipeline, whose cost models read the
/// per-function attributes.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/IR/Module.h>

namespace reussir::llvmpass {

inline void stampSizeAttributes(llvm::Module &module, bool minSize) {
  for (llvm::Function &function : module) {
    // optnone wins over optsize/minsize, same as clang.
    if (function.isDeclaration() ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone))
      continue;
    function.addFnAttr(llvm::Attribute::OptimizeForSize);
    if (minSize)
      function.addFnAttr(llvm::Attribute::MinSize);
  }
}

} // namespace reussir::llvmpass
