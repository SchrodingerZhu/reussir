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
/// This file implements the TokenInstantiation pass which inserts token
/// allocation operations for TokenAcceptor operations that do not have a token
/// assigned.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirInterfaces.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Sync/IR/SyncDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
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

    // A dynamically sized token means a dynamic-extent array box: its byte
    // size is `header + product(sizes) * elemsize`, assembled here from the
    // construction's extent operands (canonical layout — offset 0, suffix
    // strides — so the payload is exactly the element product).
    mlir::Value dynamicSize;
    if (tokenType.isDynamicSize()) {
      auto rcCreate = dyn_cast<ReussirRcCreateOp>(op);
      if (!rcCreate)
        return op->emitOpError("cannot instantiate a dynamically sized token "
                               "for this acceptor"),
               mlir::failure();
      auto arrayType = llvm::cast<ArrayType>(
          rcCreate.getRcPtr().getType().getElementType());
      auto boxType = rcCreate.getRcPtr().getType().getInnerBoxType();
      mlir::DataLayout dataLayout = mlir::DataLayout::closest(op);
      uint64_t payloadOffset = boxType.getDynamicPayloadOffset(dataLayout);
      uint64_t elementSize =
          dataLayout.getTypeSize(arrayType.getElementType());
      mlir::Location loc = op->getLoc();
      mlir::Value count;
      size_t nextExtent = 0;
      for (int64_t dim : arrayType.getShape()) {
        mlir::Value factor =
            mlir::ShapedType::isDynamic(dim)
                ? rcCreate.getExtents()[nextExtent++]
                : mlir::arith::ConstantIndexOp::create(rewriter, loc, dim)
                      .getResult();
        count = count ? mlir::arith::MulIOp::create(rewriter, loc, count,
                                                    factor)
                            .getResult()
                      : factor;
      }
      mlir::Value elemBytes = mlir::arith::MulIOp::create(
          rewriter, loc, count,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, elementSize));
      dynamicSize = mlir::arith::AddIOp::create(
          rewriter, loc, elemBytes,
          mlir::arith::ConstantIndexOp::create(rewriter, loc, payloadOffset));
    }

    auto allocOp = ReussirTokenAllocOp::create(rewriter, op->getLoc(),
                                               tokenType, dynamicSize);

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
  using Base::Base;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();

    // Set up the pattern rewrite infrastructure
    mlir::RewritePatternSet patterns(&getContext());
    if (acceptors)
      patterns.add<TokenInstantiationPattern>(&getContext());
    if (producers)
      patterns.add<TokenProductionPattern>(&getContext());

    // Apply the patterns using greedy rewrite
    if (mlir::failed(
            mlir::applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace reussir
