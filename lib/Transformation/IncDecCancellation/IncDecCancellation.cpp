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
/// This file implements the Reussir increment/decrement cancellation pass.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/IncDecCancellation.h"
#include "Reussir/Analysis/EquivalenceAnalysis.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
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
  // A destructuring release does more than drop the box's count: it also
  // transfers a retain to each bound member (`reussir-rc-dispatch-fusion`
  // folded the member's own `rc.inc` into the release). Cancelling keeps
  // the box alive, but the members still owe their references — reissue
  // them at the release's position before it disappears.
  if (decOp.isDestructuring()) {
    mlir::OpBuilder builder(decOp);
    decOp.rematerializeBoundRetains(builder);
  }
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
bool armReleasesOnAllPaths(mlir::Region &region, mlir::TypedValue<RcType> rcPtr,
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
bool cancelIntoDispatch(ReussirRcIncOp incOp, ReussirRecordDispatchOp dispatch,
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

  for (ReussirRcDecOp dec : releases)
    eraseOrReplaceDecOp(dec);
  incOp.erase();
  return true;
}

} // namespace

llvm::LogicalResult runIncDecCancellation(mlir::func::FuncOp func) {
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
      // opaque.
      if (llvm::isa<mlir::CallOpInterface>(next))
        break;
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
      // `reussir.cell.set` implicitly releases the old cell element. Like
      // `ref.drop`, the released rc is reachable only *through* the cell, so
      // no operand alias query can prove it disjoint from the incremented
      // pointer; cancelling across it could let the release hit zero while
      // the value is still live. A set of a trivially-copyable element
      // expands to a plain store and stays transparent.
      if (auto setOp = llvm::dyn_cast<ReussirCellSetOp>(next);
          setOp &&
          !isTriviallyCopyable(setOp.getCell().getType().getElementType()))
        break;
      // A `reussir.cell.rmw` or `reussir.cell.rdlock` body is arbitrary
      // nested code (it may drop the moved-out or cloned element or anything
      // else) that this same-block scan never visits — treat the whole op as
      // opaque.
      if (llvm::isa<ReussirCellRmwOp, ReussirCellRdlockOp>(next))
        break;
      // Count-*observing* ops: `array.with_unique_view`, `closure.uniqify`
      // and a standalone `rc.is_unique` read `count == 1` to decide between
      // in-place mutation and a defensive clone. The inc/dec pair around them
      // is not redundant — it is exactly what makes the count observation see
      // the value as shared — so cancelling across them would turn a
      // copy-on-write into an in-place mutation of a live shared value.
      if (auto viewOp = llvm::dyn_cast<ReussirArrayWithUniqueViewOp>(next);
          viewOp && aliasAnalysis.alias(op.getRcPtr(), viewOp.getArray()) !=
                        mlir::AliasResult::NoAlias)
        break;
      if (auto uniqifyOp = llvm::dyn_cast<ReussirClosureUniqifyOp>(next);
          uniqifyOp &&
          aliasAnalysis.alias(op.getRcPtr(), uniqifyOp.getClosure()) !=
              mlir::AliasResult::NoAlias)
        break;
      if (auto isUniqueOp = llvm::dyn_cast<ReussirRcIsUniqueOp>(next);
          isUniqueOp &&
          aliasAnalysis.alias(op.getRcPtr(), isUniqueOp.getRcPtr()) !=
              mlir::AliasResult::NoAlias)
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
