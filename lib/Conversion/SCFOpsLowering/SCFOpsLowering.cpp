#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

#include "Reussir/Conversion/SCFOpsLowering.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRSCFOPSLOWERINGPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct ReussirNullableDispatchOpRewritePattern
    : public mlir::OpRewritePattern<ReussirNullableDispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirNullableDispatchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // First, create a check operation to get the null flag from the input.
    auto flag = rewriter.create<reussir::ReussirNullableCheckOp>(
        op.getLoc(), op.getNullable());
    auto scfIfOp = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), op->getResultTypes(), flag, /*addThenRegion=*/true,
        /*addElseRegion=*/true);
    // first, do the easy part, for else region, we can just inline the
    // operation
    rewriter.inlineBlockBefore(&*op.getNullRegion().begin(),
                               &*scfIfOp.getElseRegion().begin(),
                               scfIfOp.getElseRegion().begin()->begin());

    // Now, for the then region, we first create the coerced value
    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    auto coerced = rewriter.create<reussir::ReussirNullableCoerceOp>(
        op.getLoc(), op.getNullable().getType().getPtrTy(), op.getNullable());
    // Then we inline the region, supplying coerced value as the argument
    rewriter.inlineBlockBefore(
        &*op.getNonNullRegion().begin(), &*scfIfOp.getThenRegion().begin(),
        scfIfOp.getThenRegion().begin()->end(), mlir::ValueRange{coerced});
    rewriter.replaceOp(op, scfIfOp);
    return mlir::success();
  }
};

struct ReussirRecordDispatchOpRewritePattern
    : public mlir::OpRewritePattern<ReussirRecordDispatchOp> {
  using OpRewritePattern::OpRewritePattern;

private:
  static mlir::DenseI64ArrayAttr
  getAllTagsAsSingletons(ReussirRecordDispatchOp op) {
    llvm::SmallVector<int64_t> allTags;
    for (auto tagSet : op.getTagSets()) {
      mlir::DenseI64ArrayAttr tagArray =
          llvm::cast<mlir::DenseI64ArrayAttr>(tagSet);
      if (tagArray.size() != 1)
        return {};
      allTags.push_back(tagArray[0]);
    }
    return mlir::DenseI64ArrayAttr::get(op.getContext(), allTags);
  }
  static mlir::Value buildPreDispatcher(ReussirRecordTagOp tag,
                                        ReussirRecordDispatchOp op,
                                        mlir::PatternRewriter &rewriter) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    llvm::DenseMap<int64_t, int64_t> tagToRegionIdx;
    llvm::SmallVector<int64_t> allTags;
    for (auto [idx, tagSet] : llvm::enumerate(op.getTagSets())) {
      mlir::DenseI64ArrayAttr tagArray =
          llvm::cast<mlir::DenseI64ArrayAttr>(tagSet);
      for (auto tag : tagArray.asArrayRef()) {
        allTags.push_back(tag);
        tagToRegionIdx[tag] = idx;
      }
    };
    auto indexSwitchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
        op.getLoc(), rewriter.getIndexType(), tag.getResult(), allTags,
        allTags.size());

    for (auto [tag, region] :
         llvm::zip(allTags, indexSwitchOp.getCaseRegions())) {
      mlir::Block *block = rewriter.createBlock(&region, region.begin());
      rewriter.setInsertionPointToStart(block);
      auto constantIdx = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), tagToRegionIdx[tag]);
      rewriter.create<mlir::scf::YieldOp>(op.getLoc(),
                                          constantIdx->getResults());
    }
    {
      mlir::Block *block =
          rewriter.createBlock(&indexSwitchOp.getDefaultRegion(),
                               indexSwitchOp.getDefaultRegion().begin());
      rewriter.setInsertionPointToStart(block);
      auto poison = rewriter.create<mlir::ub::PoisonOp>(
          op.getLoc(), rewriter.getIndexType());
      rewriter.create<mlir::scf::YieldOp>(op.getLoc(), poison->getResults());
    }
    return indexSwitchOp.getResult(0);
  }

public:
  mlir::LogicalResult
  matchAndRewrite(ReussirRecordDispatchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // First, create a RecordTagOps operation to get the tag from the input.
    auto tag = rewriter.create<reussir::ReussirRecordTagOp>(op.getLoc(),
                                                            op.getVariant());

    mlir::Value outerSwitchValue;
    mlir::DenseI64ArrayAttr outerSwitchCases;

    if (auto allTags = getAllTagsAsSingletons(op)) {
      outerSwitchValue = tag.getResult();
      outerSwitchCases = allTags;
    } else {
      outerSwitchValue = buildPreDispatcher(tag, op, rewriter);
      outerSwitchCases = mlir::DenseI64ArrayAttr::get(
          op.getContext(),
          llvm::to_vector(llvm::seq<int64_t>(0, op.getTagSets().size())));
    }

    auto indexSwitchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
        op.getLoc(), op->getResultTypes(), outerSwitchValue, outerSwitchCases,
        outerSwitchCases.size());
    // mark default region as unreachable
    {
      mlir::Block *block =
          rewriter.createBlock(&indexSwitchOp.getDefaultRegion(),
                               indexSwitchOp.getDefaultRegion().begin());
      rewriter.setInsertionPointToStart(block);
      llvm::SmallVector<mlir::Value, 1> poisonValues;
      if (op.getValue()) {
        auto poison = rewriter.create<mlir::ub::PoisonOp>(
            op.getLoc(), op.getValue().getType());
        poisonValues.push_back(poison);
      }
      rewriter.create<mlir::scf::YieldOp>(op.getLoc(), poisonValues);
    }
    for (auto [idx, zipped] : llvm::enumerate(
             llvm::zip(op.getTagSets(), indexSwitchOp.getCaseRegions()))) {
      auto [tagSet, region] = zipped;
      mlir::DenseI64ArrayAttr tagArray =
          llvm::cast<mlir::DenseI64ArrayAttr>(tagSet);
      llvm::SmallVector<mlir::Value, 1> args;
      mlir::Block *block = rewriter.createBlock(&region, region.begin());
      rewriter.setInsertionPointToStart(block);
      // if we know exact variant, we need to coerce the variant to the exact
      // type
      if (tagArray.size() == 1) {
        RefType variantRef = op.getVariant().getType();
        RecordType recordType =
            llvm::cast<RecordType>(variantRef.getElementType());
        mlir::Type targetVariantType = recordType.getMembers()[tagArray[0]];
        RefType coercedType =
            RefType::get(rewriter.getContext(), targetVariantType,
                         variantRef.getCapability());
        auto coerced = rewriter.create<reussir::ReussirRecordCoerceOp>(
            op.getLoc(), coercedType, rewriter.getIndexAttr(tagArray[0]),
            op.getVariant());
        args.push_back(coerced);
      }
      // inline the block, supplying coerced value as the argument
      rewriter.inlineBlockBefore(&op->getRegion(idx).front(), block,
                                 block->end(), args);
    }
    rewriter.replaceOp(op, indexSwitchOp);
    return mlir::success();
  }
};

struct ReussirScfYieldOpRewritePattern
    : public mlir::OpRewritePattern<ReussirScfYieldOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirScfYieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op->getOperands());
    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// SCFOpsLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct SCFOpsLoweringPass
    : public impl::ReussirSCFOpsLoweringPassBase<SCFOpsLoweringPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateSCFOpsLoweringConversionPatterns(patterns);

    // Configure target legality
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect,
                           mlir::ub::UBDialect, reussir::ReussirDialect>();

    // Illegal operations
    target.addIllegalOp<ReussirNullableDispatchOp, ReussirRecordDispatchOp,
                        ReussirScfYieldOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateSCFOpsLoweringConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // Add conversion patterns for Reussir SCF operations
  patterns.add<ReussirNullableDispatchOpRewritePattern,
               ReussirRecordDispatchOpRewritePattern,
               ReussirScfYieldOpRewritePattern>(patterns.getContext());
}

} // namespace reussir
