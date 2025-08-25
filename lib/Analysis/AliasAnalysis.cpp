#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <cstddef>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>

using namespace mlir;

namespace reussir {
namespace {
/// A simple AliasAnalysis implementation for Reussir that conservatively
/// returns MayAlias/ModRef for all queries. This is intended to be registered
/// into an instance of mlir::AliasAnalysis via addAnalysisImplementation.
class ReussirAliasAnalysisImpl {
  struct UnionFindNode {
    size_t parent;
    size_t rank;
  };
  std::vector<UnionFindNode> nodes{};
  llvm::DenseMap<Value, size_t> valueToNode{};

public:
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  size_t getNode(Value value);
  size_t findRoot(size_t nodeId);
  void unionByRank(size_t x, size_t y);
  AliasResult decideAlias(TypedValue<RefType> lhs, TypedValue<RefType> rhs);
  AliasResult decideAlias(TypedValue<RcType> lhs, TypedValue<RcType> rhs);
  AliasResult decideAlias(TypedValue<NullableType> lhs,
                          TypedValue<NullableType> rhs);
  TypedValue<RefType> getProducerAsRef(TypedValue<RefType> value);
  TypedValue<NullableType> getProducerAsNullable(Value value);
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

  size_t lhsNode = getNode(lhs);
  size_t rhsNode = getNode(rhs);

  if (findRoot(lhsNode) == findRoot(rhsNode))
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
          .Default([](Value) { return AliasResult::MayAlias; });

  if (res == AliasResult::MustAlias)
    unionByRank(lhsNode, rhsNode);
  return res;
}

ModRefResult ReussirAliasAnalysisImpl::getModRef(Operation *, Value) {
  return ModRefResult::getModAndRef();
}

size_t ReussirAliasAnalysisImpl::getNode(Value value) {
  auto it = valueToNode.find(value);
  if (it == valueToNode.end()) {
    nodes.emplace_back(nodes.size(), 0);
    valueToNode[value] = nodes.size() - 1;
    return nodes.size() - 1;
  }
  return it->second;
}

size_t ReussirAliasAnalysisImpl::findRoot(size_t nodeId) {
  size_t root = nodeId;
  while (nodes[root].parent != root)
    root = nodes[root].parent;

  while (nodes[nodeId].parent != root) {
    size_t parent = nodes[nodeId].parent;
    nodes[nodeId].parent = root;
    nodeId = parent;
  }
  return root;
}

void ReussirAliasAnalysisImpl::unionByRank(size_t x, size_t y) {
  x = findRoot(x);
  y = findRoot(y);
  if (x == y)
    return;
  if (nodes[x].rank < nodes[y].rank)
    std::swap(x, y);
  nodes[y].parent = x;
  if (nodes[x].rank == nodes[y].rank)
    nodes[x].rank++;
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
  auto blockNullableDispatchOp =
      llvm::dyn_cast_if_present<ReussirNullableDispatchOp>(
          blockArg.getOwner()->getParentOp());
  if (blockNullableDispatchOp)
    return blockNullableDispatchOp.getNullable();
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

AliasResult ReussirAliasAnalysisImpl::decideAlias(TypedValue<RcType> lhs,
                                                  TypedValue<RcType> rhs) {
  // If both side stems from a load operation, check if the reference being
  // loaded is aliased.
  auto lhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(lhs.getDefiningOp());
  auto rhsOp = llvm::dyn_cast_if_present<ReussirRefLoadOp>(rhs.getDefiningOp());
  if (lhsOp && rhsOp)
    return alias(lhsOp.getRef(), rhsOp.getRef());
  // Case 2: check if they are produced by a nullable coercion.
  auto lhsNullable = getProducerAsNullable(lhs);
  auto rhsNullable = getProducerAsNullable(rhs);
  if (lhsNullable && rhsNullable)
    return alias(lhsNullable, rhsNullable);
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
