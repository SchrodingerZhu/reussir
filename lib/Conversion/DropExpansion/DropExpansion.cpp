#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

#include "Reussir/Conversion/DropExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRDROPEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

// TODO: Add conversion patterns for drop operations
// Examples:
// - ReussirDropOpRewritePattern
// - ReussirOutlinedDropOpRewritePattern
// - ReussirNormalDropOpRewritePattern

} // namespace

//===----------------------------------------------------------------------===//
// DropExpansionPass
//===----------------------------------------------------------------------===//

namespace {
struct DropExpansionPass
    : public impl::ReussirDropExpansionPassBase<DropExpansionPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateDropExpansionConversionPatterns(patterns);

    // Configure target legality
    target.addLegalDialect<mlir::func::FuncDialect, reussir::ReussirDialect>();

    // Illegal operations - TODO: Add specific drop operations that need
    // expansion target.addIllegalOp<ReussirDropOp, ReussirOutlinedDropOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateDropExpansionConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // Add conversion patterns for Reussir drop operations
  // TODO: Add patterns here
  // patterns.add<ReussirDropOpRewritePattern,
  //              ReussirOutlinedDropOpRewritePattern,
  //              ReussirNormalDropOpRewritePattern>(patterns.getContext());
}

} // namespace reussir
