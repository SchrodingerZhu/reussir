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
/// This file implements Reussir region-related lowering patterns.
///
//===----------------------------------------------------------------------===//

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Reussir/Conversion/RegionPatterns.h"
#include "Reussir/IR/ReussirDialect.h.inc"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRREGIONPATTERNSPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct AttachVTablePattern : public mlir::OpRewritePattern<ReussirRcCreateOp> {
private:
  ReussirRegionVTableOp
  createVTableIfNotExists(mlir::ModuleOp moduleOp, RecordType type,
                          mlir::func::FuncOp dropOp,
                          mlir::OpBuilder &builder) const {
    mlir::SymbolTable symbolTable(moduleOp);
    llvm::Twine vtablePrefix = "_RINvNvC4core8regional6vtable";
    llvm::Twine vtableSuffix = "E";
    std::string vtableName =
        (vtablePrefix + type.getName().strref().ltrim("_R") + vtableSuffix)
            .str();
    if (auto existingVTable = symbolTable.lookup<ReussirRegionVTableOp>(
            builder.getStringAttr(vtableName)))
      return existingVTable;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    // A trivially-copyable record has no destructor: the vtable is still needed
    // (for the object's size/alignment and scan program), but its drop pointer
    // is left null, which the runtime skips.
    mlir::FlatSymbolRefAttr dropOpRef =
        dropOp ? mlir::SymbolRefAttr::get(dropOp) : mlir::FlatSymbolRefAttr{};
    auto vtableOp = ReussirRegionVTableOp::create(builder, moduleOp.getLoc(),
                                                  vtableName, type, dropOpRef);
    return vtableOp;
  }

public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Nothing to do if vtable is already attached or no region
    if (!op.needsVTable() || op.getVtable())
      return mlir::failure();
    mlir::Type elementType = op.getRcPtr().getType().getElementType();
    RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    // A trivially-copyable record needs no destructor, so leave the vtable's
    // drop pointer null rather than synthesizing a no-op dtor.
    mlir::func::FuncOp dropFunc =
        isTriviallyCopyable(recordType)
            ? mlir::func::FuncOp{}
            : createDtorIfNotExists(moduleOp, recordType, rewriter);
    ReussirRegionVTableOp vtableOp =
        createVTableIfNotExists(moduleOp, recordType, dropFunc, rewriter);
    mlir::FlatSymbolRefAttr vtableRef = mlir::SymbolRefAttr::get(vtableOp);
    op.setVtableAttr(vtableRef);
    return mlir::success();
  }
};

struct RegionInlinePattern : public mlir::OpRewritePattern<ReussirRegionRunOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRegionRunOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto region = ReussirRegionCreateOp::create(
        rewriter, op.getLoc(), RegionType::get(op->getContext()));
    auto yieldOp =
        llvm::cast<ReussirRegionYieldOp>(op.getBody().front().getTerminator());
    rewriter.inlineBlockBefore(&op.getBody().front(), op, region->getResults());
    auto yieldValue = yieldOp.getValue();
    auto rcType =
        yieldValue ? llvm::dyn_cast<RcType>(yieldValue.getType()) : nullptr;

    // Remove the yield op
    rewriter.eraseOp(yieldOp);

    // Create the final value based on the yield value
    mlir::Value finalValue;
    if (rcType && rcType.getCapability() == Capability::flex) {
      auto freezeOp = ReussirRcFreezeOp::create(
          rewriter, op.getLoc(), op->getResult(0).getType(), yieldValue);
      finalValue = freezeOp.getResult();
    } else {
      finalValue = yieldValue;
    }

    // Create cleanup operation
    rewriter.setInsertionPoint(op);
    ReussirRegionCleanupOp::create(rewriter, op.getLoc(), region);

    // Replace the whole RegionRunOp with the final value
    if (finalValue)
      rewriter.replaceOp(op, finalValue);
    else
      rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RegionPatternsPass
//===----------------------------------------------------------------------===//

namespace {
struct RegionPatternsPass
    : public impl::ReussirRegionPatternsPassBase<RegionPatternsPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateRegionPatternsConversionPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateRegionPatternsConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<AttachVTablePattern, RegionInlinePattern>(patterns.getContext());
}

} // namespace reussir
