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
/// This file implements the Reussir alias analysis.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <cstddef>
#include <llvm/ADT/IntEqClasses.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <utility>

using namespace mlir;

namespace reussir {
namespace {
/// A simple AliasAnalysis implementation for Reussir that conservatively
/// returns MayAlias/ModRef for all queries. This is intended to be registered
/// into an instance of mlir::AliasAnalysis via addAnalysisImplementation.
class ReussirAliasAnalysisImpl {
  llvm::IntEqClasses equivalenceClasses{};
  llvm::DenseMap<Value, unsigned> valueToIndex{};
  unsigned nextIndex = 0;

public:
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  unsigned getIndex(Value value);
  AliasResult decideAlias(TypedValue<RefType> lhs, TypedValue<RefType> rhs);
  AliasResult decideAlias(TypedValue<RcType> lhs, TypedValue<RcType> rhs);
  AliasResult decideAlias(TypedValue<NullableType> lhs,
                          TypedValue<NullableType> rhs);
  TypedValue<RefType> getProducerAsRef(TypedValue<RefType> value);
  TypedValue<NullableType> getProducerAsNullable(Value value);
  bool sameProjectionPath(Value lhs, Value rhs);
};

AliasResult ReussirAliasAnalysisImpl::alias(Value lhs, Value rhs) {
  // Identity law
  if (lhs == rhs)
    return AliasResult::MustAlias;

  Type lhsType = lhs.getType();
  Type rhsType = rhs.getType();

  // We do not proceed with the analysis if the types are different in ID
  if (lhsType.getTypeID() != rhsType.getTypeID())
    return AliasResult::MayAlias;

  // Check if values are already in the same equivalence class
  unsigned lhsIndex = getIndex(lhs);
  unsigned rhsIndex = getIndex(rhs);

  if (equivalenceClasses.findLeader(lhsIndex) ==
      equivalenceClasses.findLeader(rhsIndex))
    return AliasResult::MustAlias;

  auto res =
      llvm::TypeSwitch<Value, AliasResult>(lhs)
          .Case<TypedValue<RefType>>([&](TypedValue<RefType> typedLhs) {
            TypedValue<RefType> typedRhs = llvm::cast<TypedValue<RefType>>(rhs);
            return decideAlias(typedLhs, typedRhs);
          })
          .Case<TypedValue<RcType>>([&](TypedValue<RcType> typedLhs) {
            TypedValue<RcType> typedRhs = llvm::cast<TypedValue<RcType>>(rhs);
            return decideAlias(typedLhs, typedRhs);
          })
          .Case<TypedValue<NullableType>>(
              [&](TypedValue<NullableType> typedLhs) {
                TypedValue<NullableType> typedRhs =
                    llvm::cast<TypedValue<NullableType>>(rhs);
                return decideAlias(typedLhs, typedRhs);
              })
          .Default([](Value) { return AliasResult::MayAlias; });

  if (res == AliasResult::MustAlias)
    equivalenceClasses.join(lhsIndex, rhsIndex);
  return res;
}

ModRefResult ReussirAliasAnalysisImpl::getModRef(Operation *, Value) {
  return ModRefResult::getModAndRef();
}

bool ReussirAliasAnalysisImpl::sameProjectionPath(Value lhs, Value rhs) {
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
    if (lhsProj || rhsProj)
      return false;
    break;
  }

  auto lhsBorrow =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(lhs.getDefiningOp());
  auto rhsBorrow =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(rhs.getDefiningOp());
  if (lhsBorrow && rhsBorrow)
    return alias(lhsBorrow.getRcPtr(), rhsBorrow.getRcPtr()) ==
           AliasResult::MustAlias;

  auto lhsSpill =
      llvm::dyn_cast_if_present<ReussirRefSpilledOp>(lhs.getDefiningOp());
  auto rhsSpill =
      llvm::dyn_cast_if_present<ReussirRefSpilledOp>(rhs.getDefiningOp());
  if (lhsSpill && rhsSpill)
    return lhsSpill.getValue() == rhsSpill.getValue();

  return false;
}

unsigned ReussirAliasAnalysisImpl::getIndex(Value value) {
  auto it = valueToIndex.find(value);
  if (it == valueToIndex.end()) {
    unsigned index = nextIndex++;
    valueToIndex[value] = index;
    equivalenceClasses.grow(nextIndex);
    return index;
  }
  return it->second;
}

AliasResult ReussirAliasAnalysisImpl::decideAlias(TypedValue<RefType> lhs,
                                                  TypedValue<RefType> rhs) {
  // Case 1: check if two references are produced by two projection. If so,
  // check if they are projected at the same index inside an aliased parent.
  auto lhsProjOp =
      llvm::dyn_cast_if_present<ReussirRefProjectOp>(lhs.getDefiningOp());
  auto rhsProjOp =
      llvm::dyn_cast_if_present<ReussirRefProjectOp>(rhs.getDefiningOp());
  if (lhsProjOp && rhsProjOp && lhsProjOp.getIndex() == rhsProjOp.getIndex())
    return alias(lhsProjOp.getRef(), rhsProjOp.getRef());

  // Case 2, check if they are borrowed from a aliased RC pointer.
  auto lhsRcOp =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(lhs.getDefiningOp());
  auto rhsRcOp =
      llvm::dyn_cast_if_present<ReussirRcBorrowOp>(rhs.getDefiningOp());
  if (lhsRcOp && rhsRcOp)
    return alias(lhsRcOp.getRcPtr(), rhsRcOp.getRcPtr());

  // Case 3, check if they are produced by a nullable coercion.
  auto lhsNullable = getProducerAsNullable(lhs);
  auto rhsNullable = getProducerAsNullable(rhs);
  if (lhsNullable && rhsNullable)
    return alias(lhsNullable, rhsNullable);

  // Case 4, record coerce operation.
  auto lhsRef = getProducerAsRef(lhs);
  auto rhsRef = getProducerAsRef(rhs);
  if (lhsRef && rhsRef)
    return alias(lhsRef, rhsRef);

  return AliasResult::MayAlias;
}

TypedValue<NullableType>
ReussirAliasAnalysisImpl::getProducerAsNullable(Value value) {
  // Case 1: coerced from nullable
  auto coerceOp =
      llvm::dyn_cast_if_present<ReussirNullableCoerceOp>(value.getDefiningOp());
  if (coerceOp)
    return coerceOp.getNullable();
  // Case 2: block argument
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
ReussirAliasAnalysisImpl::getProducerAsRef(TypedValue<RefType> value) {
  // Case 1: coerced from references
  auto coerceOp =
      llvm::dyn_cast_if_present<ReussirRecordCoerceOp>(value.getDefiningOp());
  if (coerceOp)
    return coerceOp.getVariant();
  // Case 2: block argument
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
// must compare MustAlias for the member's retain (issued on the extract)
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

AliasResult ReussirAliasAnalysisImpl::decideAlias(TypedValue<RcType> lhs,
                                                  TypedValue<RcType> rhs) {
  // If both side stems from a load operation, check if the reference being
  // loaded is aliased.
  auto lhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(lhs.getDefiningOp());
  auto rhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(rhs.getDefiningOp());
  if (lhsOp && rhsOp) {
    if (sameProjectionPath(lhsOp.getRef(), rhsOp.getRef()))
      return AliasResult::MustAlias;
    return alias(lhsOp.getRef(), rhsOp.getRef());
  }
  // Case 2: check if they are produced by a nullable coercion.
  auto lhsNullable = getProducerAsNullable(lhs);
  auto rhsNullable = getProducerAsNullable(rhs);
  if (lhsNullable && rhsNullable)
    return alias(lhsNullable, rhsNullable);
  // Case 3: the same field of the same by-value record, read directly by
  // extract on one side and through the record's spill on the other.
  auto lhsField = valueRecordField(lhs);
  auto rhsField = valueRecordField(rhs);
  if (lhsField && rhsField && lhsField->first == rhsField->first &&
      lhsField->second == rhsField->second)
    return AliasResult::MustAlias;
  return AliasResult::MayAlias;
}

AliasResult
ReussirAliasAnalysisImpl::decideAlias(TypedValue<NullableType> lhs,
                                      TypedValue<NullableType> rhs) {
  // Similar to Rc, but only check if they are produced aliased pointer loading.
  auto lhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(lhs.getDefiningOp());
  auto rhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(rhs.getDefiningOp());
  if (lhsOp && rhsOp)
    return alias(lhsOp.getRef(), rhsOp.getRef());
  return AliasResult::MayAlias;
}
} // namespace

void registerAliasAnalysisImplementations(AliasAnalysis &aa) {
  aa.addAnalysisImplementation(ReussirAliasAnalysisImpl{});
}
} // namespace reussir
