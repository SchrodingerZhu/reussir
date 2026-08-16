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
/// This file declares the Reussir equivalence analysis.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_ANALYSIS_EQUIVALENCEANALYSIS_H
#define REUSSIR_ANALYSIS_EQUIVALENCEANALYSIS_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/IntEqClasses.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>

#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

/// A structural *value equivalence* analysis: are two SSA values known to
/// hold identical runtime values (for pointer-typed values, the same
/// address)?
///
/// This is deliberately NOT a may-alias analysis. It answers "same bits",
/// not "overlapping locations": it never proves disjointness, and its rules
/// compare producers structurally instead of reasoning about memory
/// locations. Two refs it declares equivalent do point at the same address,
/// but two refs it cannot relate may still overlap.
///
/// Key rules:
/// 1. Identity: a value is equivalent to itself.
/// 2. Type compatibility: only values of the same type kind (MLIR type ID)
///    are related; everything else is "unknown".
/// 3. Reference projection: refs projected at the same index from
///    equivalent parents are equivalent.
/// 4. RC borrowing: refs borrowed from equivalent RC pointers are
///    equivalent.
/// 5. Nullable coercion: values coerced (or dispatched) from the same
///    nullable are equivalent.
/// 6. Record coercion: refs coerced (or dispatched) from equivalent variant
///    refs are equivalent.
/// 7. Loads: RC/nullable values loaded from equivalent refs are equivalent.
///    This rule leans on Reussir's immutability of shared box content —
///    the referenced slot cannot be overwritten between the two loads —
///    and additionally recognizes reads of the same field of the same
///    by-value record through *distinct* spill slots (equal contents in
///    different storage: value equivalence where address aliasing would be
///    plain wrong).
///
/// Results are cached in a union-find keyed on the SSA value, so queries
/// against a fixed function body are cheap. The cache does not observe IR
/// mutation: erase-and-recreate churn between queries can, in principle,
/// recycle a `Value`'s storage onto a stale entry. Lookups are guarded by
/// the value's type to catch the common shape of that hazard, but clients
/// that mutate the IR mid-query-stream should treat the analysis as
/// invalidated and rebuild it.
class EquivalenceAnalysis {
public:
  /// Whether `lhs` and `rhs` are known to hold identical runtime values.
  /// `false` means "unknown", never "distinct".
  bool areEquivalent(mlir::Value lhs, mlir::Value rhs);

private:
  llvm::IntEqClasses equivalenceClasses{};
  llvm::DenseMap<mlir::Value, std::pair<unsigned, mlir::Type>> valueToIndex{};
  unsigned nextIndex = 0;

  unsigned getIndex(mlir::Value value);
  bool decideEquivalent(mlir::TypedValue<RefType> lhs,
                        mlir::TypedValue<RefType> rhs);
  bool decideEquivalent(mlir::TypedValue<RcType> lhs,
                        mlir::TypedValue<RcType> rhs);
  bool decideEquivalent(mlir::TypedValue<NullableType> lhs,
                        mlir::TypedValue<NullableType> rhs);
  mlir::TypedValue<RefType> getProducerAsRef(mlir::TypedValue<RefType> value);
  mlir::TypedValue<NullableType> getProducerAsNullable(mlir::Value value);
  bool loadedFromSameStorage(mlir::Value lhs, mlir::Value rhs);
};

/// Register the Reussir equivalence analysis on the given aggregate, adapted
/// to the `mlir::AliasAnalysis` vocabulary: equivalent values report
/// MustAlias, everything else falls through as MayAlias so other registered
/// implementations (notably `LocalAliasAnalysis`, which can prove NoAlias
/// from effect annotations) get their say.
void registerAliasAnalysisImplementations(mlir::AliasAnalysis &aliasAnalysis);

} // namespace reussir

#endif // REUSSIR_ANALYSIS_EQUIVALENCEANALYSIS_H
