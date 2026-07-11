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
// `closure.apply`/`closure.uniqify`/`closure.clone` never change a closure's
// vtable — only the static type (one fewer leading parameter) and the cursor
// — so the RETURN type is the one component of the static type that every
// value backed by the same vtable agrees on. It is also the only thing the
// slot ABIs depend on (`evaluate` is `fn(rc<box>) -> c` reading all arguments
// from the payload; `drop`/`clone` are signature-free). Each closure vtable
// therefore carries exactly one type id, keyed by its return type, and each
// indirect slot call site asserts the id of its operand's return type: the
// minimal scheme under which every vtable observable at a tested site carries
// the tested id, which is all `WholeProgramDevirt` requires.
//
// The id embeds the base-62 BLAKE3 digest of the return type's canonical
// textual form (the `Blake3Symbol.h` convention), so identical return types
// produce identical ids in any module. Ids are metadata strings, never
// symbols, so no mangling scheme applies.
//
// See docs/design/closure-wpd.md for the full design.
//
//===----------------------------------------------------------------------===//
#ifndef REUSSIR_CONVERSION_CLOSUREWPD_H
#define REUSSIR_CONVERSION_CLOSUREWPD_H

#include <string>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

/// The module-level unit attribute set by the basic-ops lowering when
/// `closure-wpd` is enabled. Its presence is the switch the conversion
/// patterns key their call-site WPD emission off.
inline constexpr llvm::StringRef kClosureWpdModuleAttr = "reussir.wpd";

/// The string attribute holding a vtable's WPD type id. The basic-ops pass
/// prologue stamps it on each outlined `reussir.closure.vtable` op; the
/// vtable conversion pattern carries it onto the `llvm.mlir.global`, and the
/// dialect's LLVM translation interface turns it into `!type` metadata
/// (offset 0 — the vtable has a single address point) plus translation-unit
/// `!vcall_visibility` on the translated global.
inline constexpr llvm::StringRef kClosureWpdTypeIdAttr = "reussir.wpd.type_id";

/// The WPD type id of the return-type family `closureType` belongs to — the
/// id its vtable carries and the strongest id any call site can assert
/// against a value the vtable may back.
inline std::string closureWpdTypeId(ClosureType closureType) {
  std::string id = "reussir.closure.wpd.";
  if (mlir::Type output = closureType.getOutputType()) {
    std::string text;
    llvm::raw_string_ostream os(text);
    output.print(os);
    llvm::BLAKE3 hasher;
    hasher.update(text);
    id += base62Digits(blake3Words(hasher));
  } else {
    id += "unit";
  }
  return id;
}

} // namespace reussir

#endif // REUSSIR_CONVERSION_CLOSUREWPD_H
