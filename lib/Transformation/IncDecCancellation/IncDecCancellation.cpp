//===-- IncDecCancellation.cpp ----------------------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/IncDecCancellation.h"
#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRINCDECCANCELLATIONPASS
#include "Reussir/Transformation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// IncDecCancellationPass
//===----------------------------------------------------------------------===//

namespace {
struct IncDecCancellationPass
    : public impl::ReussirIncDecCancellationPassBase<IncDecCancellationPass> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(runIncDecCancellation(getOperation())))
      signalPassFailure();
  }
};

ReussirRcDecOp findPostDominantAliasedRcDec(
    mlir::TypedValue<RcType> rcPtr, mlir::AliasAnalysis &aliasAnalysis,
    mlir::PostDominanceInfo &postDominanceInfo, mlir::Region &dropRegion) {
  ReussirRcDecOp fusionTarget{};
  dropRegion.walk([&](ReussirRcDecOp decOp) {
    auto aliasResult = aliasAnalysis.alias(decOp.getRcPtr(), rcPtr);
    if (postDominanceInfo.postDominates(decOp->getBlock(),
                                        &dropRegion.front()) &&
        aliasResult == mlir::AliasResult::MustAlias) {
      fusionTarget = decOp;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return fusionTarget;
}

void eraseOrReplaceDecOp(ReussirRcDecOp decOp) {
  if (decOp.use_empty())
    decOp.erase();
  else {
    mlir::OpBuilder builder(decOp);
    auto replacement = ReussirNullableCreateOp::create(
        builder, decOp.getLoc(), decOp.getResultTypes(), nullptr);
    decOp.replaceAllUsesWith(replacement->getResults());
    decOp.erase();
  }
}

// Whether `op` is a structured branch whose regions execute at most once
// with exactly one region taken, and whose results are all trivial (their
// validity cannot depend on the released box). Loops are excluded, as is an
// `scf.if` without an else (its implicit empty path never releases).
bool isSingleShotTrivialBranch(mlir::Operation *op) {
  if (llvm::isa<mlir::LoopLikeOpInterface>(op))
    return false;
  if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
    if (ifOp.getElseRegion().empty())
      return false;
  } else if (!llvm::isa<mlir::scf::IndexSwitchOp, ReussirRecordDispatchOp,
                        ReussirNullableDispatchOp>(op)) {
    return false;
  }
  return llvm::all_of(op->getResultTypes(), [](mlir::Type type) {
    return isTriviallyCopyable(type);
  });
}

// Collects the release of `rcPtr` on *every* path through `region`: either a
// direct release at the region's top level, or a nested single-shot branch
// all of whose paths release. Everything before the release must leave the
// box's count alone (borrows are fine; anything else touching the pointer,
// or an opaque op, disqualifies the path).
bool armReleasesOnAllPaths(mlir::Region &region,
                           mlir::TypedValue<RcType> rcPtr,
                           mlir::AliasAnalysis &aliasAnalysis,
                           llvm::SmallVectorImpl<ReussirRcDecOp> &releases) {
  if (!region.hasOneBlock())
    return false;
  for (mlir::Operation &op : region.front()) {
    if (auto dec = llvm::dyn_cast<ReussirRcDecOp>(op)) {
      if (aliasAnalysis.alias(dec.getRcPtr(), rcPtr) ==
          mlir::AliasResult::MustAlias) {
        releases.push_back(dec);
        return true;
      }
      if (aliasAnalysis.alias(dec.getRcPtr(), rcPtr) !=
          mlir::AliasResult::NoAlias)
        return false;
      continue;
    }
    if (isSingleShotTrivialBranch(&op)) {
      llvm::SmallVector<ReussirRcDecOp> nested;
      bool all = op.getNumRegions() > 0;
      for (mlir::Region &inner : op.getRegions())
        if (!armReleasesOnAllPaths(inner, rcPtr, aliasAnalysis, nested)) {
          all = false;
          break;
        }
      if (all) {
        releases.append(nested.begin(), nested.end());
        return true;
      }
      return false;
    }
    if (llvm::is_contained(op.getOperands(), mlir::Value(rcPtr)) &&
        !llvm::isa<ReussirRcBorrowOp>(op))
      return false;
    if (op.getNumRegions() > 0 || llvm::isa<mlir::CallOpInterface>(op))
      return false;
  }
  return false;
}

// `inc %v` followed by a dispatch over `%v` whose *every* arm releases the
// box is borrow semantics in disguise: the increment and the arm releases
// cancel. A destructuring release (see `reussir-rc-dispatch-fusion`)
// additionally fused away the arm's bound-member retains — on the now
// certainly-shared path the arm still needs its own references, so they are
// rematerialized at the release's position. This is what reduces an inlined
// read-only predicate (rbtree's `is_red`) to pure tag loads.
bool cancelIntoDispatch(ReussirRcIncOp incOp,
                        ReussirRecordDispatchOp dispatch,
                        mlir::AliasAnalysis &aliasAnalysis) {
  auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
      dispatch.getVariant().getDefiningOp());
  if (!borrow || aliasAnalysis.alias(borrow.getRcPtr(), incOp.getRcPtr()) !=
                     mlir::AliasResult::MustAlias)
    return false;

  llvm::SmallVector<ReussirRcDecOp> releases;
  for (mlir::Region &region : dispatch.getRegions())
    if (!armReleasesOnAllPaths(region, incOp.getRcPtr(), aliasAnalysis,
                               releases))
      return false;

  for (ReussirRcDecOp dec : releases) {
    if (dec.isDestructuring()) {
      RcType rcType = dec.getRcPtr().getType();
      auto recordType = llvm::cast<RecordType>(rcType.getElementType());
      int64_t tag = dec.getDestructureTagAttr().getInt();
      auto payload = llvm::cast<RecordType>(recordType.getMembers()[tag]);
      mlir::OpBuilder builder(dec);
      if (!dec.getBoundMembersAttr().empty()) {
        auto refType = builder.getType<RefType>(recordType,
                                                Capability::unspecified,
                                                rcType.getAtomicKind());
        auto payloadRefType = builder.getType<RefType>(
            payload, Capability::unspecified, rcType.getAtomicKind());
        mlir::Value ref = ReussirRcBorrowOp::create(builder, dec.getLoc(),
                                                    refType, dec.getRcPtr());
        mlir::Value coerced = ReussirRecordCoerceOp::create(
            builder, dec.getLoc(), payloadRefType, builder.getIndexAttr(tag),
            ref);
        for (int64_t idx : dec.getBoundMembersAttr().asArrayRef()) {
          auto projectedTy = getProjectedType(
              payload.getMembers()[idx], payload.getMemberIsField()[idx],
              Capability::unspecified);
          auto projectedRefTy = builder.getType<RefType>(
              projectedTy, Capability::unspecified, rcType.getAtomicKind());
          mlir::Value slot = ReussirRefProjectOp::create(
              builder, dec.getLoc(), projectedRefTy, coerced,
              builder.getIndexAttr(idx));
          mlir::Value member = ReussirRefLoadOp::create(
              builder, dec.getLoc(), projectedTy, slot);
          ReussirRcIncOp::create(builder, dec.getLoc(), member);
        }
      }
    }
    eraseOrReplaceDecOp(dec);
  }
  incOp.erase();
  return true;
}

} // namespace

// Whether a value of `container` may transitively hold a reference of
// `needle`'s type: walks record members, rc/nullable payloads, and treats
// closures as universal containers (their captures are type-erased).
// Conservative `true` when unsure; memoized against nominal recursion.
static bool mayTransitivelyContain(mlir::Type container, mlir::Type needle,
                                   llvm::SmallPtrSetImpl<const void *> &seen) {
  if (container == needle)
    return true;
  if (!seen.insert(container.getAsOpaquePointer()).second)
    return false;
  if (llvm::isa<ClosureType>(container))
    return true;
  if (auto rc = llvm::dyn_cast<RcType>(container))
    return mayTransitivelyContain(rc.getElementType(), needle, seen);
  if (auto nullable = llvm::dyn_cast<NullableType>(container))
    return mayTransitivelyContain(nullable.getPtrTy(), needle, seen);
  if (auto record = llvm::dyn_cast<RecordType>(container)) {
    if (record.getMembers().empty())
      return true; // incomplete view: assume the worst
    for (mlir::Type member : record.getMembers())
      if (member && mayTransitivelyContain(member, needle, seen))
        return true;
    return false;
  }
  // Scalars and tokens cannot hold references.
  return !llvm::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(
      container);
}

// A `func.call` is transparent to an inc/dec cancellation of `rcPtr` when it
// provably neither consumes nor releases that value: every argument position
// either merely lends its value (a declared borrowed parameter of the
// callee) or is an owned value whose type cannot transitively contain
// `rcPtr`'s type (so consuming it cannot release `rcPtr`).
static bool callOnlyBorrows(mlir::func::CallOp call, mlir::Value rcPtr,
                            mlir::SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee)
    return false;
  auto borrowed =
      callee->getAttrOfType<mlir::DenseI64ArrayAttr>("reussir.borrowed_params");
  if (!borrowed)
    return false;
  for (auto [i, operand] : llvm::enumerate(call.getArgOperands())) {
    if (llvm::is_contained(borrowed.asArrayRef(), static_cast<int64_t>(i)))
      continue;
    if (operand == rcPtr)
      return false;
    llvm::SmallPtrSet<const void *, 16> seen;
    if (mayTransitivelyContain(operand.getType(), rcPtr.getType(), seen))
      return false;
  }
  return true;
}

llvm::LogicalResult runIncDecCancellation(mlir::func::FuncOp func) {
  mlir::SymbolTableCollection symbolTables;
  mlir::AliasAnalysis aliasAnalysis(func);
  registerAliasAnalysisImplementations(aliasAnalysis);
  mlir::PostDominanceInfo postDominanceInfo(func);
  llvm::SmallVector<ReussirRcIncOp> incOps;
  func->walk([&](ReussirRcIncOp op) { incOps.push_back(op); });
  for (auto op : incOps) {
    // iterate all succeeding operations in the same block
    mlir::Operation *next = op->getNextNode();
    while (next) {
      if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(next)) {
        // This is a similar situation as other aborting cases.
        if (!ifOp->hasAttr(kExpandedDecrementAttr))
          break;
        ReussirRcDecOp decOp = findPostDominantAliasedRcDec(
            op.getRcPtr(), aliasAnalysis, postDominanceInfo,
            ifOp.getThenRegion());
        if (decOp) {
          op->moveBefore(ifOp.elseYield());
          eraseOrReplaceDecOp(decOp);
          break;
        }
      }

      if (auto decOp = llvm::dyn_cast<ReussirRcDecOp>(next)) {
        if (aliasAnalysis.alias(op.getRcPtr(), decOp.getRcPtr()) ==
            mlir::AliasResult::MustAlias) {
          op.erase();
          eraseOrReplaceDecOp(decOp);
        }
        // avoid nested decrement operation that may mess up the cancellation
        // maybe this should be handled more gracefully to continue the
        // cancellation if the decrement is known to be disjoint from the
        // increment.
        break;
      }
      // A call *site* (`mlir::CallOpInterface`, e.g. `func.call`) is opaque: it
      // may consume the incremented value (a closure application lowers to a
      // `func.call` that takes ownership of its closure), so cancelling across
      // it could drop a live reference. Guard on the call-site interface —
      // *not* `CallableOpInterface`, which is the call *target* (`func.func`)
      // and never appears mid-block. A control-flow op with regions is likewise
      // opaque. Exception: a `func.call` that provably only borrows relative
      // to the incremented value (declared borrowed parameters; owned
      // arguments type-disjoint from it) neither consumes nor releases it,
      // so the walk continues across it.
      if (llvm::isa<mlir::CallOpInterface>(next)) {
        auto funcCall = llvm::dyn_cast<mlir::func::CallOp>(next);
        if (!funcCall || !callOnlyBorrows(funcCall, op.getRcPtr(), symbolTables))
          break;
        next = next->getNextNode();
        continue;
      }
      if (auto dispatch = llvm::dyn_cast<ReussirRecordDispatchOp>(next)) {
        cancelIntoDispatch(op, dispatch, aliasAnalysis);
        break;
      }
      if (llvm::isa<mlir::RegionBranchOpInterface>(next))
        break;
      // `reussir.ref.drop` releases every rc reachable *through* the referenced
      // value (the first cancellation run precedes acquire/drop expansion, so
      // it appears here unexpanded). The incremented rc may be stored inside
      // the dropped value — e.g. a spilled `[value]` enum whose payload was
      // just loaded — and since the reference operand never aliases the rc
      // pointer itself, no operand alias query can prove disjointness.
      // Cancelling across it lets the drop take the count to zero while the
      // payload is still live. A drop of a trivially-copyable element expands
      // to nothing and stays transparent. `reussir.region.cleanup` likewise
      // releases the region's objects wholesale.
      if (auto dropOp = llvm::dyn_cast<ReussirRefDropOp>(next);
          dropOp &&
          !isTriviallyCopyable(dropOp.getRef().getType().getElementType()))
        break;
      if (llvm::isa<ReussirRegionCleanupOp>(next))
        break;
      // `closure.apply`/`closure.eval` consume/mutate only their own closure
      // operand, so they are risky only when that operand may alias the
      // incremented pointer; an application of an unrelated closure is harmless
      // and does not block the cancellation.
      if (auto applyOp = llvm::dyn_cast<ReussirClosureApplyOp>(next);
          applyOp && aliasAnalysis.alias(op.getRcPtr(), applyOp.getClosure()) !=
                         mlir::AliasResult::NoAlias)
        break;
      if (auto evalOp = llvm::dyn_cast<ReussirClosureEvalOp>(next);
          evalOp && aliasAnalysis.alias(op.getRcPtr(), evalOp.getClosure()) !=
                        mlir::AliasResult::NoAlias)
        break;
      next = next->getNextNode();
    }
  }
  return mlir::success();
}

} // namespace reussir
