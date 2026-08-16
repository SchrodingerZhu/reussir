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
/// This file implements the Reussir equivalence analysis.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Analysis/EquivalenceAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <cstddef>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <utility>

using namespace mlir;

namespace reussir {

bool EquivalenceAnalysis::areEquivalent(Value lhs, Value rhs) {
  // Identity law.
  if (lhs == rhs)
    return true;

  // Values of different type kinds are never related.
  if (lhs.getType().getTypeID() != rhs.getType().getTypeID())
    return false;

  // Check the union-find cache first.
  unsigned lhsIndex = getIndex(lhs);
  unsigned rhsIndex = getIndex(rhs);
  if (equivalenceClasses.findLeader(lhsIndex) ==
      equivalenceClasses.findLeader(rhsIndex))
    return true;

  bool equivalent =
      llvm::TypeSwitch<Value, bool>(lhs)
          .Case<TypedValue<RefType>>([&](TypedValue<RefType> typedLhs) {
            return decideEquivalent(typedLhs,
                                    llvm::cast<TypedValue<RefType>>(rhs));
          })
          .Case<TypedValue<RcType>>([&](TypedValue<RcType> typedLhs) {
            return decideEquivalent(typedLhs,
                                    llvm::cast<TypedValue<RcType>>(rhs));
          })
          .Case<TypedValue<NullableType>>(
              [&](TypedValue<NullableType> typedLhs) {
                return decideEquivalent(
                    typedLhs, llvm::cast<TypedValue<NullableType>>(rhs));
              })
          .Default([](Value) { return false; });

  if (equivalent)
    equivalenceClasses.join(lhsIndex, rhsIndex);
  return equivalent;
}

unsigned EquivalenceAnalysis::getIndex(Value value) {
  auto [it, inserted] =
      valueToIndex.try_emplace(value, nextIndex, value.getType());
  // A hit whose recorded type disagrees with the value's current type is a
  // recycled `Value` from erase-and-recreate churn — retire the stale entry
  // instead of resurrecting its equivalence class. (Same-type recycling
  // remains undetectable; see the class comment.)
  if (!inserted && it->second.second != value.getType()) {
    it->second = {nextIndex, value.getType()};
    inserted = true;
  }
  if (inserted)
    equivalenceClasses.grow(++nextIndex);
  return it->second.first;
}

bool EquivalenceAnalysis::decideEquivalent(TypedValue<RefType> lhs,
                                           TypedValue<RefType> rhs) {
  // Case 1: refs projected at the same index from equivalent parents.
  auto lhsProjOp =
      llvm::dyn_cast_if_present<ReussirRefProjectOp>(lhs.getDefiningOp());
  auto rhsProjOp =
      llvm::dyn_cast_if_present<ReussirRefProjectOp>(rhs.getDefiningOp());
  if (lhsProjOp && rhsProjOp && lhsProjOp.getIndex() == rhsProjOp.getIndex())
    return areEquivalent(lhsProjOp.getRef(), rhsProjOp.getRef());

  // Case 2: refs borrowed from equivalent RC pointers.
  auto lhsRcOp =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(lhs.getDefiningOp());
  auto rhsRcOp =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(rhs.getDefiningOp());
  if (lhsRcOp && rhsRcOp)
    return areEquivalent(lhsRcOp.getRcPtr(), rhsRcOp.getRcPtr());

  // Case 3: values produced by a nullable coercion or dispatch.
  auto lhsNullable = getProducerAsNullable(lhs);
  auto rhsNullable = getProducerAsNullable(rhs);
  if (lhsNullable && rhsNullable)
    return areEquivalent(lhsNullable, rhsNullable);

  // Case 4: variant payload refs from a record coercion or dispatch.
  auto lhsRef = getProducerAsRef(lhs);
  auto rhsRef = getProducerAsRef(rhs);
  if (lhsRef && rhsRef)
    return areEquivalent(lhsRef, rhsRef);

  return false;
}

/// Strip matching projection chains off both refs: both sides must project
/// at the same index to the same type at every step. On success, `lhs` and
/// `rhs` are rewritten to the (non-projection) roots.
static bool stripMatchingProjections(Value &lhs, Value &rhs) {
  while (true) {
    auto lhsProj =
        llvm::dyn_cast_if_present<ReussirRefProjectOp>(lhs.getDefiningOp());
    auto rhsProj =
        llvm::dyn_cast_if_present<ReussirRefProjectOp>(rhs.getDefiningOp());
    if (lhsProj && rhsProj) {
      if (lhsProj.getIndex() != rhsProj.getIndex())
        return false;
      if (lhsProj.getProjected().getType() != rhsProj.getProjected().getType())
        return false;
      lhs = lhsProj.getRef();
      rhs = rhsProj.getRef();
      continue;
    }
    return !lhsProj && !rhsProj;
  }
}

bool EquivalenceAnalysis::loadedFromSameStorage(Value lhs, Value rhs) {
  if (!stripMatchingProjections(lhs, rhs))
    return false;

  auto lhsBorrow =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(lhs.getDefiningOp());
  auto rhsBorrow =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(rhs.getDefiningOp());
  if (lhsBorrow && rhsBorrow)
    return areEquivalent(lhsBorrow.getRcPtr(), rhsBorrow.getRcPtr());

  // Two spills of the same SSA value are *distinct* stack slots holding
  // identical bytes: loads through them agree even though the refs never
  // alias. This is the rule that makes the analysis value equivalence
  // rather than address aliasing.
  auto lhsSpill =
      llvm::dyn_cast_if_present<ReussirRefSpilledOp>(lhs.getDefiningOp());
  auto rhsSpill =
      llvm::dyn_cast_if_present<ReussirRefSpilledOp>(rhs.getDefiningOp());
  if (lhsSpill && rhsSpill)
    return lhsSpill.getValue() == rhsSpill.getValue();

  return false;
}

TypedValue<NullableType>
EquivalenceAnalysis::getProducerAsNullable(Value value) {
  // Case 1: coerced from nullable.
  auto coerceOp =
      llvm::dyn_cast_if_present<ReussirNullableCoerceOp>(value.getDefiningOp());
  if (coerceOp)
    return coerceOp.getNullable();
  // Case 2: the nonnull region argument of a nullable dispatch.
  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  if (blockArg) {
    auto blockNullableDispatchOp =
        llvm::dyn_cast_if_present<ReussirNullableDispatchOp>(
            blockArg.getOwner()->getParentOp());
    if (blockNullableDispatchOp)
      return blockNullableDispatchOp.getNullable();
  }
  return nullptr;
}

TypedValue<RefType>
EquivalenceAnalysis::getProducerAsRef(TypedValue<RefType> value) {
  // Case 1: coerced from a variant reference.
  auto coerceOp =
      llvm::dyn_cast_if_present<ReussirRecordCoerceOp>(value.getDefiningOp());
  if (coerceOp)
    return coerceOp.getVariant();
  // Case 2: an arm argument of a record dispatch.
  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  if (blockArg) {
    auto blockRecordDispatchOp =
        llvm::dyn_cast_if_present<ReussirRecordDispatchOp>(
            blockArg.getOwner()->getParentOp());
    if (blockRecordDispatchOp)
      return blockRecordDispatchOp.getVariant();
  }
  return nullptr;
}

// A by-value record's field read as an SSA extract (`record.extract %s[i]`)
// and read back through the record's spill
// (`ref.load(ref.project(ref.spilled %s, [i]))`) produce the same stored
// pointer. The pair appears wherever a by-value struct result is consumed:
// the member is extracted for further use while the struct itself is
// spilled so its release can be decomposed field by field. The two reads
// must compare equivalent for the member's retain (issued on the extract)
// to cancel against the release's decrement (issued on the spilled load) —
// otherwise the surviving decrement observes a shared count at runtime,
// never yields a reuse token, and poisons token-donor selection with a
// statically-plausible but always-null candidate.
static std::optional<std::pair<Value, uint64_t>> valueRecordField(Value value) {
  if (auto extractOp = llvm::dyn_cast_if_present<ReussirRecordExtractOp>(
          value.getDefiningOp()))
    return std::make_pair(mlir::Value(extractOp.getRecord()),
                          extractOp.getIndex().getZExtValue());
  auto loadOp =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!loadOp)
    return std::nullopt;
  auto projectOp = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      loadOp.getRef().getDefiningOp());
  if (!projectOp)
    return std::nullopt;
  auto spilledOp = llvm::dyn_cast_if_present<ReussirRefSpilledOp>(
      projectOp.getRef().getDefiningOp());
  if (!spilledOp)
    return std::nullopt;
  return std::make_pair(spilledOp.getValue(),
                        projectOp.getIndex().getZExtValue());
}

bool EquivalenceAnalysis::decideEquivalent(TypedValue<RcType> lhs,
                                           TypedValue<RcType> rhs) {
  // Case 1: both loaded from the same storage (equivalent refs, or distinct
  // spill slots of the same value). Loads through equivalent refs agree
  // because shared box content is immutable between the two reads.
  auto lhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(lhs.getDefiningOp());
  auto rhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(rhs.getDefiningOp());
  if (lhsOp && rhsOp) {
    if (loadedFromSameStorage(lhsOp.getRef(), rhsOp.getRef()))
      return true;
    return areEquivalent(lhsOp.getRef(), rhsOp.getRef());
  }
  // Case 2: produced by a nullable coercion or dispatch.
  auto lhsNullable = getProducerAsNullable(lhs);
  auto rhsNullable = getProducerAsNullable(rhs);
  if (lhsNullable && rhsNullable)
    return areEquivalent(lhsNullable, rhsNullable);
  // Case 3: the same field of the same by-value record, read directly by
  // extract on one side and through the record's spill on the other.
  auto lhsField = valueRecordField(lhs);
  auto rhsField = valueRecordField(rhs);
  return lhsField && rhsField && lhsField->first == rhsField->first &&
         lhsField->second == rhsField->second;
}

bool EquivalenceAnalysis::decideEquivalent(TypedValue<NullableType> lhs,
                                           TypedValue<NullableType> rhs) {
  // Like RC pointers, but only the loaded-from-equivalent-refs rule applies.
  auto lhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(lhs.getDefiningOp());
  auto rhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(rhs.getDefiningOp());
  if (lhsOp && rhsOp)
    return areEquivalent(lhsOp.getRef(), rhsOp.getRef());
  return false;
}

namespace {
/// Adapts `EquivalenceAnalysis` to the `mlir::AliasAnalysis` aggregate:
/// equivalence maps to MustAlias, "unknown" falls through as MayAlias so
/// other implementations may still refine the query. It contributes no
/// mod-ref precision.
class EquivalenceAliasAnalysisAdapter {
public:
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs) {
    return equivalence.areEquivalent(lhs, rhs) ? AliasResult::MustAlias
                                               : AliasResult::MayAlias;
  }
  mlir::ModRefResult getModRef(mlir::Operation *, mlir::Value) {
    return ModRefResult::getModAndRef();
  }

private:
  EquivalenceAnalysis equivalence;
};
} // namespace

void registerAliasAnalysisImplementations(AliasAnalysis &aa) {
  aa.addAnalysisImplementation(EquivalenceAliasAnalysisAdapter{});
}
} // namespace reussir
