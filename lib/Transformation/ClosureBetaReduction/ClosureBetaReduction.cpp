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
//  * LOOPINLINE: the loop-carried redex, in its general frontend shape —
//
//      %c = create {body}; (uniqify; apply cap)*     // capture binding
//      loop* {                                       // any LoopLike nest
//        rc.inc %c; (uniqify; apply x_i)*; eval(...) // per-iteration chain
//      }
//      rc.dec %c
//
//    (`scf.for` and `affine.for` alike). The body is spliced into the loop
//    at the eval and the closure box vanishes: its create / per-iteration
//    inc+chain / trailing dec traffic is self-contained (a fresh box
//    nothing else can observe), so erasing all of it is count-neutral. The
//    per-iteration uniqify is a *genuine* guard — the carried box is
//    shared, so it clones every iteration — but the clone is itself a fresh
//    box the chain consumes whole: cloning inc'd the stored captures out
//    and the eval consumed them, a wash the fused form reproduces. Capture
//    counts are preserved, not reasoned about: each rc-typed capture bound
//    *outside* the loop gains a per-iteration `rc.inc` at the splice point
//    (the inlined body consumes its capture parameters each iteration) and
//    one `rc.dec` where the box's dec stood (the box's drop used to release
//    the ref the apply consumed). Captures bound *inside* the loop were
//    consumed once per iteration before and still are — they just take
//    their seat in the substitution pack.
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
#include <mlir/Interfaces/LoopLikeInterface.h>
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
    rewriter.inlineBlockBefore(&body, eval, eval.getArgs());
    // Read the operand after inlining, once block arguments have been replaced
    // with the eval pack and before the moved terminator is erased.
    mlir::Value yielded = yield.getValue();
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

// LOOPINLINE: a beta redex carried through a loop nest. See the file
// comment for the ownership accounting.
struct InlineLoopCarriedCreatePattern
    : public mlir::OpRewritePattern<ReussirClosureEvalOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp eval,
                  mlir::PatternRewriter &rewriter) const override {
    // Peel the per-iteration links: single-use `uniqify | apply` ops in the
    // eval's own block, down to the value carried across iterations. The
    // uniqify here is a *genuine* guard (the carried box is shared, so it
    // clones every iteration) — but the clone is a fresh box the fused
    // chain consumes whole, so erasing it is count-neutral; cloning used to
    // inc the stored captures out and the eval consumed them, a wash the
    // fused form reproduces exactly. Per-iteration applied args were
    // consumed once per iteration before and still are: they need no
    // adjustment, only a seat in the substitution pack.
    llvm::SmallVector<mlir::Operation *> innerChain;
    llvm::SmallVector<mlir::Value> innerCaps; // collected top-down
    mlir::Value carried = eval.getClosure();
    while (true) {
      mlir::Operation *def = carried.getDefiningOp();
      if (!def)
        return mlir::failure();
      if (def->getBlock() != eval->getBlock())
        break; // left the loop body: `carried` is the loop-carried value
      if (!carried.hasOneUse())
        return mlir::failure();
      if (auto apply = llvm::dyn_cast<ReussirClosureApplyOp>(def)) {
        innerChain.push_back(def);
        innerCaps.push_back(apply.getArg());
        carried = apply.getClosure();
      } else if (auto uniqify = llvm::dyn_cast<ReussirClosureUniqifyOp>(def)) {
        innerChain.push_back(def);
        carried = uniqify.getClosure();
      } else {
        return mlir::failure();
      }
    }
    mlir::Operation *top = carried.getDefiningOp();
    mlir::Block *defBlock = top->getBlock();

    // Every region between the carried value and the eval must be a loop
    // (any dialect implementing `LoopLikeOpInterface`): the eval then runs
    // once per iteration of a nest the chain dominates from outside.
    mlir::Operation *walk = eval;
    while (walk->getBlock() != defBlock) {
      mlir::Operation *parent = walk->getParentOp();
      if (!parent || !llvm::isa<mlir::LoopLikeOpInterface>(parent))
        return mlir::failure();
      walk = parent;
    }
    mlir::Operation *outermostLoop = walk;

    // The chain below the loop-carried value: linear (every inner link
    // single-use), one block, bottoming at an inlined create. Every uniqify
    // over such a chain is a runtime no-op — the value is unique by
    // construction, fresh from the create with no use between the links.
    llvm::SmallVector<mlir::Operation *> outerChain;
    llvm::SmallVector<mlir::Value> outerCaps; // collected top-down
    ReussirClosureCreateOp create;
    mlir::Value link = carried;
    while (true) {
      mlir::Operation *def = link.getDefiningOp();
      if (!def || def->getBlock() != defBlock)
        return mlir::failure();
      if (link != carried && !link.hasOneUse())
        return mlir::failure();
      if (auto apply = llvm::dyn_cast<ReussirClosureApplyOp>(def)) {
        outerChain.push_back(def);
        outerCaps.push_back(apply.getArg());
        link = apply.getClosure();
      } else if (auto uniqify = llvm::dyn_cast<ReussirClosureUniqifyOp>(def)) {
        outerChain.push_back(def);
        link = uniqify.getClosure();
      } else if ((create = llvm::dyn_cast<ReussirClosureCreateOp>(def))) {
        outerChain.push_back(def);
        break;
      } else {
        return mlir::failure();
      }
    }
    if (!create.isInlined())
      return mlir::failure();
    mlir::Value token = create.getToken();
    if (token && !token.getDefiningOp<ReussirTokenAllocOp>())
      return mlir::failure();
    // Walking top-down visits the last application first; the body's
    // parameters bind in application order — outer applies first, then the
    // per-iteration ones, then the eval's pack.
    std::reverse(outerCaps.begin(), outerCaps.end());
    std::reverse(innerCaps.begin(), innerCaps.end());

    mlir::Block &body = create.getBody().front();
    if (body.getNumArguments() !=
        outerCaps.size() + innerCaps.size() + eval.getArgs().size())
      return mlir::failure();

    // The carried value's remaining uses: exactly one per-iteration inc
    // beside the eval, and exactly one plain dec after the nest at the
    // chain's level.
    mlir::Operation *firstLink =
        innerChain.empty() ? eval.getOperation() : innerChain.back();
    llvm::SmallVector<ReussirRcIncOp> incs;
    ReussirRcDecOp dec;
    for (mlir::Operation *user : carried.getUsers()) {
      if (user == firstLink)
        continue;
      if (auto inc = llvm::dyn_cast<ReussirRcIncOp>(user)) {
        if (inc->getBlock() != eval->getBlock() || !inc->isBeforeInBlock(eval))
          return mlir::failure();
        incs.push_back(inc);
        continue;
      }
      if (auto d = llvm::dyn_cast<ReussirRcDecOp>(user)) {
        if (dec || d->getBlock() != defBlock ||
            !outermostLoop->isBeforeInBlock(d))
          return mlir::failure();
        dec = d;
        continue;
      }
      return mlir::failure();
    }
    if (incs.size() != 1 || !dec || dec.getNumResults() != 0)
      return mlir::failure();

    // Per-iteration ownership for the rc-typed captures the box stored:
    // the inlined body consumes its capture parameters each iteration (see
    // the file comment).
    rewriter.setInsertionPoint(eval);
    for (mlir::Value cap : outerCaps)
      if (llvm::isa<RcType>(cap.getType()))
        ReussirRcIncOp::create(rewriter, eval.getLoc(), cap);

    auto yield = llvm::cast<ReussirClosureYieldOp>(body.getTerminator());
    llvm::SmallVector<mlir::Value> args(outerCaps);
    llvm::append_range(args, innerCaps);
    llvm::append_range(args, eval.getArgs());
    rewriter.inlineBlockBefore(&body, eval, args);
    // The yield may refer directly to a body argument; fetch it only after
    // inlining has replaced that argument with its value from `args`.
    mlir::Value yielded = yield.getValue();
    rewriter.eraseOp(yield);
    if (eval.getNumResults())
      rewriter.replaceOp(eval, yielded);
    else
      rewriter.eraseOp(eval);

    // The box's dec used to release the captures the outer applies
    // consumed; release them directly where it stood.
    rewriter.setInsertionPoint(dec);
    for (mlir::Value cap : outerCaps)
      if (llvm::isa<RcType>(cap.getType()))
        ReussirRcDecOp::create(rewriter, dec.getLoc(),
                               /*nullableToken=*/NullableType{}, cap,
                               /*destructureTag=*/mlir::IntegerAttr{},
                               /*boundMembers=*/mlir::DenseI64ArrayAttr{});
    for (mlir::Operation *op : innerChain)
      rewriter.eraseOp(op);
    for (ReussirRcIncOp inc : incs)
      rewriter.eraseOp(inc);
    rewriter.eraseOp(dec);
    for (mlir::Operation *op : outerChain)
      rewriter.eraseOp(op);
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
                 InlineCreateIntoEvalPattern, InlineLoopCarriedCreatePattern>(
        &getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace reussir
