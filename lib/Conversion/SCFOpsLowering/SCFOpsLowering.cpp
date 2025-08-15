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

// TODO: Add conversion patterns here
// Example pattern structure:
// struct ReussirSomeOpConversionPattern
//     : public mlir::OpRewritePattern<ReussirSomeOp> {
//   using OpRewritePattern::OpRewritePattern;
//   mlir::LogicalResult
//   matchAndRewrite(ReussirSomeOp op,
//                   mlir::PatternRewriter &rewriter) const override {
//     // Implementation will go here
//     return mlir::success();
//   }
// };

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
                           mlir::math::MathDialect, mlir::func::FuncDialect>();
    target.addLegalDialect<ReussirDialect>();

    // TODO: Add illegal operations here
    // target.addIllegalOp<ReussirSomeOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateSCFOpsLoweringConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // TODO: Add patterns here
  // patterns.add<ReussirSomeOpConversionPattern>(patterns.getContext());
}

} // namespace reussir
