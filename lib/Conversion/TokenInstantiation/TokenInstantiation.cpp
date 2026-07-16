//===-- TokenInstantiation.cpp - Token instantiation pass ---------*- C++
//-*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the TokenInstantiation pass which inserts token
// allocation operations for TokenAcceptor operations that do not have a token
// assigned.
//
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirInterfaces.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Sync/IR/SyncDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRTOKENINSTANTIATIONPASS
#include "Reussir/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// TokenInstantiation Pattern
//===----------------------------------------------------------------------===//

struct TokenInstantiationPattern : public mlir::RewritePattern {
  TokenInstantiationPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check if this operation implements TokenAcceptor interface
    auto tokenAcceptor = dyn_cast<TokenAcceptor>(op);
    if (!tokenAcceptor)
      return mlir::failure();

    // Skip if already has a token
    if (tokenAcceptor.hasToken())
      return mlir::failure();

    // Get the expected token type
    TokenType tokenType = tokenAcceptor.getTokenType();

    // Create the token allocation operation before the current operation
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto allocOp =
        ReussirTokenAllocOp::create(rewriter, op->getLoc(), tokenType);

    // Assign the token to the operation
    tokenAcceptor.assignToken(allocOp.getToken());

    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// TokenProduction Pattern
//===----------------------------------------------------------------------===//

struct TokenProductionPattern : public mlir::RewritePattern {
  TokenProductionPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check if this operation implements TokenProducer interface
    auto tokenProducer = dyn_cast<TokenProducer>(op);
    if (!tokenProducer)
      return mlir::failure();

    // Skip if operation should not produce a token
    if (!tokenProducer.shouldProduceToken())
      return mlir::failure();

    // Skip if already produced a value
    if (tokenProducer.getProducedValue())
      return mlir::failure();

    // Use the interface method to replace the operation with one that produces
    return tokenProducer.replaceWithProduced(rewriter);
  }
};

//===----------------------------------------------------------------------===//
// TokenInstantiation Pass
//===----------------------------------------------------------------------===//

class ReussirTokenInstantiationPass
    : public impl::ReussirTokenInstantiationPassBase<
          ReussirTokenInstantiationPass> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();

    // Set up the pattern rewrite infrastructure
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TokenInstantiationPattern, TokenProductionPattern>(
        &getContext());

    // Apply the patterns using greedy rewrite
    if (mlir::failed(
            mlir::applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace reussir
