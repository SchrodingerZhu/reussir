//===-- SpecialPointerTag.h - Nullary variants as immediates ----*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 With LLVM Exceptions OR MIT
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_TRANSFORMATION_SPECIALPOINTERTAG_H
#define REUSSIR_TRANSFORMATION_SPECIALPOINTERTAG_H

#include "llvm/ADT/StringRef.h"

namespace reussir {

/// Module unit attribute set when the special-pointer-tag scheme is enabled:
/// nullary variants of shared rc-boxed enums are encoded as immediates whose
/// top byte is `tag + 1` (low 56 bits zero). The LLVM lowering patterns read
/// it to guard refcount and tag-word accesses on taggable types.
constexpr llvm::StringLiteral kSpecialPtrTagAttr = "reussir.special_ptr_tag";

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_SPECIALPOINTERTAG_H
