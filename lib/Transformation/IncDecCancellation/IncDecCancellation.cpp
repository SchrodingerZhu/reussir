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
      if (llvm::isa<mlir::CallOpInterface>(next) ||
          llvm::isa<mlir::RegionBranchOpInterface>(next))
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
