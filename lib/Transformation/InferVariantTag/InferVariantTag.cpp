//===-- InferVariantTag.cpp - Reussir variant tag inference impl -*- C++
//-*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/InferVariantTag.h"
#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRINFERVARIANTTAGPASS
#include "Reussir/Transformation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// InferVariantTagPass
//===----------------------------------------------------------------------===//

namespace {
struct InferVariantTagPass
    : public impl::ReussirInferVariantTagPassBase<InferVariantTagPass> {
  using Base::Base;
  void runOnOperation() override { runTagInference(getOperation()); }
};
} // namespace

void runTagInference(mlir::func::FuncOp func) {
  // Initialize analyses
  mlir::AliasAnalysis aliasAnalysis(func);
  mlir::DominanceInfo dominanceInfo(func);
  registerAliasAnalysisImplementations(aliasAnalysis);

  // Collect all coercion operations in the function
  llvm::SmallVector<ReussirRecordCoerceOp> coercionOps;
  func.walk([&](ReussirRecordCoerceOp op) { coercionOps.push_back(op); });

  // Process each block
  func.walk([&](mlir::Block *block) {
    if (block->empty())
      return;

    // Find coercion operations that dominate this block
    llvm::DenseMap<mlir::TypedValue<RefType>, int64_t> referenceTags;

    for (ReussirRecordCoerceOp coercionOp : coercionOps)
      if (dominanceInfo.dominates(coercionOp.getOperation(),
                                  &block->getOperations().front()))
        referenceTags[coercionOp.getVariant()] =
            coercionOp.getTag().getZExtValue();

    mlir::Region *region = block->getParent();
    while (region && region->getParentOp() != func) {
      ReussirRecordDispatchOp dispatchOp =
          llvm::dyn_cast<ReussirRecordDispatchOp>(region->getParentOp());
      if (dispatchOp) {
        size_t regionIdx = region->getRegionNumber();
        mlir::DenseI64ArrayAttr regionTags =
            llvm::cast<mlir::DenseI64ArrayAttr>(
                dispatchOp.getTagSets()[regionIdx]);
        if (regionTags.size() == 1)
          referenceTags[dispatchOp.getVariant()] = regionTags[0];
      }
      region = region->getParentRegion();
    }

    for (mlir::Operation &op : *block) {
      if (auto dropOp = llvm::dyn_cast<ReussirRefDropOp>(&op)) {
        mlir::TypedValue<RefType> refToDrop = dropOp.getRef();
        for (const auto &[ref, tag] : referenceTags)
          if (aliasAnalysis.alias(refToDrop, ref) ==
              mlir::AliasResult::MustAlias) {
            dropOp.setVariantAttr(mlir::IntegerAttr::get(
                mlir::IndexType::get(dropOp.getContext()), tag));
            break;
          }
      }
      if (auto acquireOp = llvm::dyn_cast<ReussirRefAcquireOp>(&op)) {
        mlir::TypedValue<RefType> refToAcquire = acquireOp.getRef();
        for (const auto &[ref, tag] : referenceTags)
          if (aliasAnalysis.alias(refToAcquire, ref) ==
              mlir::AliasResult::MustAlias) {
            acquireOp.setVariantAttr(mlir::IntegerAttr::get(
                mlir::IndexType::get(acquireOp.getContext()), tag));
            break;
          }
      }
    }
  });
}

} // namespace reussir
