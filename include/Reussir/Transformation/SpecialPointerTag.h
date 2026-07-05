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

/// Module string attribute set when the special-pointer-tag scheme is
/// enabled: nullary variants of shared rc-boxed enums are encoded as
/// unboxed immediates pointing at per-tag dummy boxes. Its value selects
/// the encoding (`kSpecialPtrTagTBI` or `kSpecialPtrTagImmortal`); the LLVM
/// lowering patterns read it to pick the guard strategy on taggable types.
constexpr llvm::StringLiteral kSpecialPtrTagAttr = "reussir.special_ptr_tag";

/// TBI encoding (aarch64): the immediate's top byte is `tag + 1` and its low
/// bits hold the dummy box address; hardware top-byte-ignore makes accesses
/// through the value land in the dummy box. Guards test the top byte.
constexpr llvm::StringLiteral kSpecialPtrTagTBI = "tbi";

/// Arch-independent encoding: the immediate *is* the dummy box address (no
/// pointer bit tricks). The dummy refcount is initialized to an immortal
/// value (2^62); guards test refcount magnitude instead of pointer bits.
constexpr llvm::StringLiteral kSpecialPtrTagImmortal = "immortal";

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_SPECIALPOINTERTAG_H
