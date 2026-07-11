//===-- ClosureWpd.h - closure whole-program devirt type ids ---*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Type identifiers for whole-program devirtualization (WPD) of closures.
//
// Closure vtables form a suffix hierarchy: `closure.apply` never changes the
// vtable — only the static type (one fewer leading parameter) and the cursor —
// so a vtable created for signature `(a, b) -> c` may back a value of static
// type `(b) -> c` or `() -> c`. Every slot's ABI depends on nothing beyond the
// return type (`evaluate` is `fn(rc<box>) -> c` reading all args from the
// payload; `drop`/`clone` are signature-independent), which is what makes the
// substitution sound. In C++/WPD terms, `apply` is an upcast: the vtable of a
// closure is tagged with one type id per *suffix* of its original signature,
// and an indirect call site tests the id of its operand's static type.
//
// Ids follow the project's v0 mangling scheme (the subset documented in
// `crates/reussir-core/src/full/mangle.rs`), using its closure-type
// production with one content-addressed segment per type. The id of a suffix
// `(tᵢ, …, tₙ) -> c` is
//
//   _R F K 17ReussirClosureWpd {C<digest(tⱼ)>}* E (C<digest(c)> | u)
//
// where `<digest(t)>` is the length-prefixed v0 identifier holding
// `b62(blake3(print(t)))` — the `Blake3Symbol.h` convention, shared with the
// Rust frontend — and a closure returning nothing uses the v0 unit code `u`.
// Ids are built from the types' canonical textual forms, so identical
// signatures produce identical ids in any module. The most-derived tier is
// unique per vtable: the path `<vtable>::wpd`, i.e. `_RNv<vtable path>3wpd`,
// nested under the vtable's own v0 symbol.
//
// See docs/design/closure-wpd.md for the full design, including the
// closed-world conditions under which the type tests are sound.
//
//===----------------------------------------------------------------------===//
#ifndef REUSSIR_CONVERSION_CLOSUREWPD_H
#define REUSSIR_CONVERSION_CLOSUREWPD_H

#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

/// The module-level attribute carrying `vtable symbol -> [type ids]`, set by
/// the basic-ops lowering when `closure-wpd` is enabled. Its presence is the
/// switch the conversion patterns key their WPD emission off.
inline constexpr llvm::StringRef kClosureWpdVtableMapAttr =
    "reussir.wpd.vtables";

/// The per-global attribute (an array of id strings) the vtable conversion
/// pattern stamps onto each closure vtable `llvm.mlir.global`. The dialect's
/// LLVM translation interface turns it into `!type` metadata (offset 0 — the
/// vtable has a single address point) plus translation-unit
/// `!vcall_visibility` on the translated global.
inline constexpr llvm::StringRef kClosureWpdTypeIdsAttr =
    "reussir.wpd.type_ids";

/// Appends `body` as a v0 identifier: `<length> [_] <body>`, with the `_`
/// separator exactly when the body leads with a digit or `_` (mirroring
/// `mangle.rs`; the length excludes the separator).
inline void appendClosureWpdV0Ident(std::string &out, llvm::StringRef body) {
  out += std::to_string(body.size());
  if (!body.empty() && (llvm::isDigit(body.front()) || body.front() == '_'))
    out += '_';
  out += body;
}

/// Appends the id segment of one type: a crate-root nominal (`C<ident>`)
/// whose identifier is the base-62 BLAKE3 digest of the type's canonical
/// textual form.
inline void appendClosureWpdTypeSegment(std::string &out, mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  llvm::BLAKE3 hasher;
  hasher.update(text);
  out += 'C';
  appendClosureWpdV0Ident(out, base62Digits(blake3Words(hasher)));
}

/// The type id of `closureType`'s exact static signature — the id an indirect
/// call site tests for an operand of this static type.
inline std::string closureWpdTypeId(ClosureType closureType) {
  std::string id = "_RFK";
  appendClosureWpdV0Ident(id, "ReussirClosureWpd");
  for (mlir::Type input : closureType.getInputTypes())
    appendClosureWpdTypeSegment(id, input);
  id += 'E';
  if (mlir::Type output = closureType.getOutputType())
    appendClosureWpdTypeSegment(id, output);
  else
    id += 'u';
  return id;
}

/// The most-derived, per-vtable tier: the path `<vtable>::wpd` nested under
/// the vtable's own v0 symbol.
inline std::string closureWpdUidTypeId(llvm::StringRef vtableSymbol) {
  std::string id = "_RNv";
  if (vtableSymbol.consume_front("_R")) {
    id += vtableSymbol;
  } else {
    // Not a v0 symbol (hand-written IR): treat it as a crate-root segment.
    id += 'C';
    appendClosureWpdV0Ident(id, vtableSymbol);
  }
  id += "3wpd";
  return id;
}

/// All type ids a vtable for `closureType` (the closure's ORIGINAL, unapplied
/// signature) carries: one per signature suffix, from the family root
/// `() -> c` up to the full signature, plus the per-vtable uid tier.
inline llvm::SmallVector<std::string>
closureWpdVtableTypeIds(ClosureType closureType, llvm::StringRef vtableSymbol) {
  llvm::SmallVector<std::string> ids;
  auto inputs = closureType.getInputTypes();
  mlir::Type output = closureType.getOutputType();
  // Suffixes from empty (family root) to the full signature.
  for (size_t kept = 0; kept <= inputs.size(); ++kept) {
    ClosureType suffix = ClosureType::get(closureType.getContext(),
                                          inputs.take_back(kept), output);
    ids.push_back(closureWpdTypeId(suffix));
  }
  ids.push_back(closureWpdUidTypeId(vtableSymbol));
  return ids;
}

} // namespace reussir

#endif // REUSSIR_CONVERSION_CLOSUREWPD_H
