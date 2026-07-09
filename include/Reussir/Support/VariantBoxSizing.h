//===-- VariantBoxSizing.h - per-constructor variant box sizing -*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 With LLVM Exceptions OR MIT
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_SUPPORT_VARIANTBOXSIZING_H
#define REUSSIR_SUPPORT_VARIANTBOXSIZING_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace reussir {

/// Module unit attribute opting the compilation into *per-constructor* box
/// sizing (#325): a boxed variant's heap cell is sized for the constructed
/// arm (`header + arm[k]`) instead of the uniform max-arm width. Field
/// offsets never move — the payload offset is derived from the variant-wide
/// alignment, so per-arm sizing only drops the trailing padding up to the
/// max arm. The measured motivation is nbe-closure, where the numerous
/// leaf arms (Var/VVar-shaped: header + one word = 16 bytes) doubled to the
/// 32-byte max-arm bin dominate the cache footprint (padding Koka to
/// uniform 32B reproduced our LLC misses and runtime almost exactly).
///
/// The invariant every consumer must preserve: for any boxed variant,
///   allocated size == token<> size == freed size == header + arm[k]
/// where k is the box's own (immutable) tag. Producers/consumers of the
/// size are staged behind this attribute (see
/// docs/design/per-constructor-box-sizing.md for the landing plan); until
/// the full stack lands the attribute is experimental and must not be
/// stamped by default.
constexpr llvm::StringLiteral kPerConstructorBoxSizingAttr =
    "reussir.per_constructor_box_sizing";

/// Whether the module enclosing `op` opted into per-constructor box sizing.
inline bool perConstructorBoxSizing(mlir::Operation *op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  return module && module->hasAttr(kPerConstructorBoxSizingAttr);
}

} // namespace reussir

#endif // REUSSIR_SUPPORT_VARIANTBOXSIZING_H
