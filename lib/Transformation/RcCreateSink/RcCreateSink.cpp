//===-- RcCreateSink.cpp - Reussir rc.create sinking -----------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRRCCREATESINKPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

struct SinkRcCreateIntoExpandedEnsurePattern
    : public mlir::OpRewritePattern<ReussirRcCreateOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isMatchCandidate(ReussirRcCreateOp create, mlir::scf::IfOp ifOp) {
    if (!create.getToken() || !ifOp ||
        !ifOp->hasAttr(REUSSIR_EXPANDED_ENSURE_ATTR))
      return false;
    if (ifOp.getNumResults() != 1 || !ifOp->hasOneUse())
      return false;
    if (create.getToken() != ifOp.getResult(0))
      return false;
    if (ifOp->getBlock() != create->getBlock())
      return false;
    if (ifOp->getNextNode() != create.getOperation())
      return false;
    return true;
  }

  static ReussirRcCreateOp
  cloneCreateIntoBranch(mlir::PatternRewriter &rewriter,
                        ReussirRcCreateOp create, mlir::Value branchToken,
                        mlir::scf::YieldOp yieldOp, bool skipRc) {
    rewriter.setInsertionPoint(yieldOp);
    auto *clonedOp = rewriter.clone(*create.getOperation());
    auto clonedCreate = llvm::cast<ReussirRcCreateOp>(clonedOp);
    clonedCreate->setOperand(1, branchToken);
    if (skipRc)
      clonedCreate.setSkipRcAttr(rewriter.getUnitAttr());
    return clonedCreate;
  }

  static void rewriteRegionYield(mlir::PatternRewriter &rewriter,
                                 ReussirRcCreateOp create, mlir::Block &block,
                                 bool skipRc) {
    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(block.getTerminator());
    auto sunkCreate = cloneCreateIntoBranch(
        rewriter, create, yieldOp.getOperand(0), yieldOp, skipRc);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yieldOp,
                                                    sunkCreate.getRcPtr());
  }

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp create,
                  mlir::PatternRewriter &rewriter) const override {
    auto ifOp = llvm::dyn_cast_or_null<mlir::scf::IfOp>(
        create.getToken().getDefiningOp());
    if (!isMatchCandidate(create, ifOp))
      return mlir::failure();

    auto newIf = mlir::scf::IfOp::create(
        rewriter, ifOp.getLoc(), create.getRcPtr().getType(),
        ifOp.getCondition(),
        /*addThenRegion=*/true, /*addElseRegion=*/true);
    newIf->setAttrs(ifOp->getAttrs());
    newIf->removeAttr(REUSSIR_EXPANDED_ENSURE_ATTR);

    rewriter.inlineBlockBefore(&ifOp.getThenRegion().front(),
                               &newIf.getThenRegion().front(),
                               newIf.getThenRegion().front().begin());
    rewriter.inlineBlockBefore(&ifOp.getElseRegion().front(),
                               &newIf.getElseRegion().front(),
                               newIf.getElseRegion().front().begin());

    rewriteRegionYield(rewriter, create, newIf.getThenRegion().front(),
                       /*skipRc=*/true);
    rewriteRegionYield(rewriter, create, newIf.getElseRegion().front(),
                       /*skipRc=*/false);

    rewriter.replaceOp(create, newIf.getResults());
    rewriter.eraseOp(ifOp);
    return mlir::success();
  }
};

struct RcCreateSinkPass
    : public impl::ReussirRcCreateSinkPassBase<RcCreateSinkPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SinkRcCreateIntoExpandedEnsurePattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace reussir
