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
///
/// THE SCHEME'S CORE INVARIANT (the one non-obvious rule everything else
/// follows from): guards classify a value as immediate-vs-box WITHOUT a
/// per-value fallback path, so the classifier's "this is a real box" answer
/// must never be wrong for an immediate. Under `tbi` the classifier is the
/// pointer's top byte alone (a register-only test — no load): ZERO MEANS
/// "REAL BOX, STORES MAY PROCEED". That is sound only if EVERY immediate
/// carries a nonzero top byte, i.e. `tag + 1` fits one byte. Hence an arm
/// whose tag cannot be encoded must never become an immediate — it stays a
/// real, allocated, token-producing box — and eligibility is enforced
/// all-or-nothing at the type level (`RcType::mayCarrySpecialPointerTag`:
/// every nullary arm must fit) rather than per arm, so downstream logic
/// (the rewrite pattern, token typing) can reason simply "nullary arm of a
/// taggable type = immediate = never a token" with no tag arithmetic.
/// A hybrid "immortal fallback inside a tbi module" (zero-top-byte
/// immediates recognized by refcount magnitude) would decouple eligibility
/// from the tag range. It was rejected not for guard cost (at the guarded
/// sites the count is already in a register: `rc.set` tests its own operand,
/// `rc.inc` just loaded it) but for what it erodes: two classifiers to keep
/// sound instead of one, immediates whose tags are no longer decodable from
/// the pointer (the tbi encoding's FFI point), and allocation/token
/// behavior that would differ between encodings for the same source —
/// all to serve variants with more than 255 arms.
constexpr llvm::StringLiteral kSpecialPtrTagAttr = "reussir.special_ptr_tag";

/// TBI encoding (aarch64): the immediate's top byte is `tag + 1` and its low
/// bits hold the dummy box address; hardware top-byte-ignore makes accesses
/// through the value land in the dummy box. Guards test the top byte.
constexpr llvm::StringLiteral kSpecialPtrTagTBI = "tbi";

/// Arch-independent encoding: the immediate *is* the dummy box address (no
/// pointer bit tricks). The dummy refcount is initialized to an immortal
/// value (2^62); guards test refcount magnitude instead of pointer bits.
constexpr llvm::StringLiteral kSpecialPtrTagImmortal = "immortal";

/// Low-bit encoding (any 64-bit target): the immortal encoding with the tag
/// *duplicated* into the pointer's low alignment bits — the immediate is
/// `dummy address | (tag + 1)` when `tag + 1 < 8` (every rc box is 8-aligned)
/// and the plain dummy address otherwise. Tag classification of an
/// encodable nullary arm is then pure pointer arithmetic (`ptr & 7`),
/// like TBI but without a hardware dependency; the price is an align-down
/// mask at every access through a possibly-tagged value. The dummy box
/// still answers any access after the mask, so unencodable arms and
/// generic paths fall back to the immortal behavior uniformly.
constexpr llvm::StringLiteral kSpecialPtrTagLsb = "lsb";

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_SPECIALPOINTERTAG_H
