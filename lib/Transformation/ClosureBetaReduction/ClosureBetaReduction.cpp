//===-- ClosureBetaReduction.cpp - Reussir closure beta reduction -*-C++-*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Beta-reduce closure application chains. Runs right after the inliner (so
// cross-function create→eval pairs are visible) and before token
// instantiation / closure outlining (so `closure.create` bodies are still
// inline regions and carry no materialized token yet).
//
// Three one-step local patterns under the greedy driver; their fixpoint
// walks each eval's chain without any hand-driven recursion:
//
//  * FOLD: `eval(apply(a, x), pack)` with a single-use apply becomes
//    `eval(x, [a] ++ pack)` — the argument prepends, so the pack stays in
//    application order. Dominance is free (def-use edge), and ownership is
//    untouched: the fused eval consumes closure and pack exactly as the
//    apply chain did.
//
//  * STRIP: a single-use `closure.uniqify` under a fused eval is erased
//    when it is a provable runtime no-op — its operand is a single-use
//    `closure.apply` or `closure.create`, i.e. a value that is unique by
//    construction (a fresh box, or an in-place application to a value the
//    chain's genuine guard already made unique). The bottom-most uniqify,
//    whose operand is anything else, never matches: it stays in the IR as
//    the fused eval's operand — the fused form carries `apply`'s contract
//    (uniqueness is the producer's obligation) and never re-establishes it.
//
//  * INLINE: `eval(create{body}, pack)` over a single-use inlined create is
//    a visible beta redex: the body block is spliced in front of the eval
//    with the pack substituted for its parameters (legal anywhere — the
//    region is IsolatedFromAbove), and the chain, the create, and its dead
//    `token.alloc` are erased. Evals spliced out of the body are re-queued
//    by the driver, which is what makes the reduction recursive.
//
// Fused evals that bottom out anywhere else survive the pass; the SCF-ops
// lowering re-expands them into unchecked applies + plain eval, so the net
// effect of FOLD+STRIP on a chain the inliner merged is the removal of its
// intermediate uniqueness checks.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRCLOSUREBETAREDUCTIONPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

// FOLD: eval(apply(a, x), pack) → eval(x, [a] ++ pack).
struct FoldApplyIntoEvalPattern
    : public mlir::OpRewritePattern<ReussirClosureEvalOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp eval,
                  mlir::PatternRewriter &rewriter) const override {
    auto apply = eval.getClosure().getDefiningOp<ReussirClosureApplyOp>();
    if (!apply || !apply.getApplied().hasOneUse())
      return mlir::failure();
    llvm::SmallVector<mlir::Value> args;
    args.push_back(apply.getArg());
    llvm::append_range(args, eval.getArgs());
    auto fused = ReussirClosureEvalOp::create(
        rewriter, eval.getLoc(),
        eval.getNumResults() ? eval.getResult().getType() : mlir::Type(),
        apply.getClosure(), args);
    rewriter.replaceOp(eval, fused->getResults());
    rewriter.eraseOp(apply);
    return mlir::success();
  }
};

// STRIP: eval(uniqify(x), pack) → eval(x, pack) when the uniqify is a
// provable runtime no-op (x is a single-use apply/create link, unique by
// construction). Only fires under a FUSED eval: a plain eval paired with
// its frontend-emitted uniqify is left byte-identical.
struct StripNoopUniqifyPattern
    : public mlir::OpRewritePattern<ReussirClosureEvalOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp eval,
                  mlir::PatternRewriter &rewriter) const override {
    if (eval.getArgs().empty())
      return mlir::failure();
    auto uniqify = eval.getClosure().getDefiningOp<ReussirClosureUniqifyOp>();
    if (!uniqify || !uniqify.getUniqified().hasOneUse())
      return mlir::failure();
    mlir::Value below = uniqify.getClosure();
    mlir::Operation *belowDef = below.getDefiningOp();
    if (!below.hasOneUse() || !belowDef ||
        !llvm::isa<ReussirClosureApplyOp, ReussirClosureCreateOp>(belowDef))
      return mlir::failure();
    rewriter.replaceOp(uniqify, below);
    return mlir::success();
  }
};

// INLINE: eval(create{body}, pack) → the body, pack substituted for its
// parameters.
struct InlineCreateIntoEvalPattern
    : public mlir::OpRewritePattern<ReussirClosureEvalOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp eval,
                  mlir::PatternRewriter &rewriter) const override {
    auto create = eval.getClosure().getDefiningOp<ReussirClosureCreateOp>();
    if (!create || !create.getClosure().hasOneUse() || !create.isInlined())
      return mlir::failure();
    // A token producer other than a plain `token.alloc` (impossible today —
    // token reuse runs later — but cheap to guard) would leak if orphaned.
    mlir::Value token = create.getToken();
    if (token && !token.getDefiningOp<ReussirTokenAllocOp>())
      return mlir::failure();

    mlir::Block &body = create.getBody().front();
    assert(body.getNumArguments() == eval.getArgs().size() &&
           "eval pack must cover the create's full signature");
    auto yield = llvm::cast<ReussirClosureYieldOp>(body.getTerminator());
    mlir::Value yielded = yield.getValue();
    rewriter.inlineBlockBefore(&body, eval, eval.getArgs());
    rewriter.eraseOp(yield);
    if (eval.getNumResults())
      rewriter.replaceOp(eval, yielded);
    else
      rewriter.eraseOp(eval);
    rewriter.eraseOp(create);
    if (token && token.use_empty())
      rewriter.eraseOp(token.getDefiningOp());
    return mlir::success();
  }
};

struct ClosureBetaReductionPass
    : public impl::ReussirClosureBetaReductionPassBase<
          ClosureBetaReductionPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldApplyIntoEvalPattern, StripNoopUniqifyPattern,
                 InlineCreateIntoEvalPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace reussir
