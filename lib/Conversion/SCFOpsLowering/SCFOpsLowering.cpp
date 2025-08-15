#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

struct ReussirNullableDispatchOpConversionPattern
    : public mlir::OpRewritePattern<ReussirNullableDispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirNullableDispatchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // First, create a check operation to get the null flag from the input.
    auto nullFlag = rewriter.create<reussir::ReussirNullableCheckOp>(
        op.getLoc(), op.getNullable());
    auto scfIfOp = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), op->getResultTypes(), nullFlag, /*addThenRegion=*/true,
        /*addElseRegion=*/true);
    (void)scfIfOp;
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
                           reussir::ReussirDialect>();

    // Illegal operations
    target.addIllegalOp<ReussirNullableDispatchOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateSCFOpsLoweringConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // Add conversion patterns for Reussir SCF operations
  patterns.add<ReussirNullableDispatchOpConversionPattern>(
      patterns.getContext());
}

} // namespace reussir
