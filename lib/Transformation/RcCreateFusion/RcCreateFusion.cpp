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
/// This file implements fusion of Reussir rc.create operations.
///
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRRCCREATEFUSIONPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

void eraseDeadRecordMaterialization(mlir::PatternRewriter &rewriter,
                                    mlir::Operation *op) {
  while (op && op->use_empty()) {
    mlir::Operation *next = nullptr;
    if (auto variant = llvm::dyn_cast<ReussirRecordVariantOp>(op))
      next = variant.getValue().getDefiningOp();
    rewriter.eraseOp(op);
    op = llvm::dyn_cast_if_present<ReussirRecordCompoundOp>(next) ? next
                                                                  : nullptr;
  }
}

mlir::TypedValue<RcType> getReusedRcFromToken(mlir::Value token) {
  auto launder =
      llvm::dyn_cast_if_present<ReussirTokenLaunderOp>(token.getDefiningOp());
  if (!launder)
    return nullptr;

  auto reinterpret = llvm::dyn_cast_if_present<ReussirRcReinterpretOp>(
      launder.getToken().getDefiningOp());
  if (!reinterpret)
    return nullptr;
  return reinterpret.getRcPtr();
}

bool structurallySameType(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs)
    return true;

  auto lhsRecord = llvm::dyn_cast<RecordType>(lhs);
  auto rhsRecord = llvm::dyn_cast<RecordType>(rhs);
  if (lhsRecord || rhsRecord) {
    if (!lhsRecord || !rhsRecord)
      return false;
    if (lhsRecord.isVariant() != rhsRecord.isVariant())
      return false;
    if (lhsRecord.isCompound() != rhsRecord.isCompound())
      return false;
    if (lhsRecord.getComplete() != rhsRecord.getComplete())
      return false;
    if (lhsRecord.getMembers().size() != rhsRecord.getMembers().size())
      return false;
    for (auto [lhsMember, rhsMember, lhsField, rhsField] : llvm::zip(
             lhsRecord.getMembers(), rhsRecord.getMembers(),
             lhsRecord.getMemberIsField(), rhsRecord.getMemberIsField())) {
      if (lhsField != rhsField)
        return false;
      if (!structurallySameType(lhsMember, rhsMember))
        return false;
    }
    return true;
  }

  return false;
}

bool isLoadFromCompoundField(mlir::Value value,
                             mlir::TypedValue<RcType> sourceRc,
                             int64_t fieldIndex) {
  auto load =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!load)
    return false;

  auto project = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      load.getRef().getDefiningOp());
  if (!project || project.getIndex().getSExtValue() != fieldIndex)
    return false;

  auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
      project.getRef().getDefiningOp());
  return borrow && borrow.getRcPtr() == sourceRc;
}

bool hasCompatibleFieldPrefix(RecordType sourcePayloadType,
                              RecordType targetPayloadType,
                              int64_t fieldIndex) {
  if (!sourcePayloadType || !targetPayloadType)
    return false;
  if (!sourcePayloadType.isCompound() || !targetPayloadType.isCompound())
    return false;
  if (fieldIndex < 0)
    return false;
  if (static_cast<size_t>(fieldIndex) >=
          sourcePayloadType.getMembers().size() ||
      static_cast<size_t>(fieldIndex) >= targetPayloadType.getMembers().size())
    return false;

  for (int64_t index = 0; index <= fieldIndex; ++index) {
    if (sourcePayloadType.getMemberIsField()[index] !=
        targetPayloadType.getMemberIsField()[index])
      return false;
    if (!structurallySameType(sourcePayloadType.getMembers()[index],
                              targetPayloadType.getMembers()[index]))
      return false;
  }
  return true;
}

bool isLoadFromVariantField(mlir::Value value,
                            mlir::TypedValue<RcType> sourceRc,
                            mlir::Type targetPayloadType, int64_t fieldIndex) {
  auto load =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!load)
    return false;

  auto project = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      load.getRef().getDefiningOp());
  if (!project || project.getIndex().getSExtValue() != fieldIndex)
    return false;

  auto coerce = llvm::dyn_cast_if_present<ReussirRecordCoerceOp>(
      project.getRef().getDefiningOp());
  if (!coerce)
    return false;

  auto sourcePayloadType = llvm::dyn_cast<RecordType>(
      coerce.getCoerced().getType().getElementType());
  auto targetPayloadRecord = llvm::dyn_cast<RecordType>(targetPayloadType);
  if (!hasCompatibleFieldPrefix(sourcePayloadType, targetPayloadRecord,
                                fieldIndex))
    return false;

  auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
      coerce.getVariant().getDefiningOp());
  return borrow && borrow.getRcPtr() == sourceRc;
}

void markCompoundAvoidedCopies(ReussirRcCreateCompoundOp op) {
  auto sourceRc = op.getToken() ? getReusedRcFromToken(op.getToken())
                                : mlir::TypedValue<RcType>{};
  if (!sourceRc)
    return;

  llvm::SmallVector<int64_t> skippedFields;
  for (auto [index, field] : llvm::enumerate(op.getFields()))
    if (isLoadFromCompoundField(field, sourceRc, index))
      skippedFields.push_back(static_cast<int64_t>(index));

  if (!skippedFields.empty())
    op->setAttr("skipFields",
                mlir::DenseI64ArrayAttr::get(op.getContext(), skippedFields));
}

void markVariantAvoidedCopies(ReussirRcCreateVariantOp op) {
  if (op.getValue())
    return;

  auto sourceRc = op.getToken() ? getReusedRcFromToken(op.getToken())
                                : mlir::TypedValue<RcType>{};
  if (!sourceRc)
    return;

  auto payloadType =
      op.getRecordType().getMembers()[op.getTag().getZExtValue()];
  llvm::SmallVector<int64_t> skippedFields;
  for (auto [index, field] : llvm::enumerate(op.getFields()))
    if (isLoadFromVariantField(field, sourceRc, payloadType, index))
      skippedFields.push_back(static_cast<int64_t>(index));

  if (!skippedFields.empty())
    op->setAttr("skipFields",
                mlir::DenseI64ArrayAttr::get(op.getContext(), skippedFields));
}

struct FuseRcCreatePattern : public mlir::OpRewritePattern<ReussirRcCreateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp create,
                  mlir::PatternRewriter &rewriter) const override {
    auto variant = llvm::dyn_cast_if_present<ReussirRecordVariantOp>(
        create.getValue().getDefiningOp());
    if (variant) {
      auto compound = llvm::dyn_cast_if_present<ReussirRecordCompoundOp>(
          variant.getValue().getDefiningOp());
      auto fused = ReussirRcCreateVariantOp::create(
          rewriter, create.getLoc(),
          mlir::TypeRange{create.getRcPtr().getType()}, variant.getTagAttr(),
          compound ? mlir::Value{} : variant.getValue(),
          compound ? compound.getFields() : mlir::ValueRange{},
          create.getToken(), create.getRegion(), create.getVtableAttr(),
          create.getSkipRcAttr(), mlir::DenseI64ArrayAttr{},
          mlir::DenseI64ArrayAttr{});
      rewriter.replaceOp(create, fused.getRcPtr());
      eraseDeadRecordMaterialization(rewriter, variant.getOperation());
      return mlir::success();
    }

    auto compound = llvm::dyn_cast_if_present<ReussirRecordCompoundOp>(
        create.getValue().getDefiningOp());
    if (!compound)
      return mlir::failure();

    auto fused = ReussirRcCreateCompoundOp::create(
        rewriter, create.getLoc(), mlir::TypeRange{create.getRcPtr().getType()},
        compound.getFields(), create.getToken(), create.getRegion(),
        create.getVtableAttr(), create.getSkipRcAttr(),
        mlir::DenseI64ArrayAttr{}, mlir::DenseI64ArrayAttr{});
    rewriter.replaceOp(create, fused.getRcPtr());
    eraseDeadRecordMaterialization(rewriter, compound.getOperation());
    return mlir::success();
  }
};

struct RcCreateFusionPass
    : public impl::ReussirRcCreateFusionPassBase<RcCreateFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuseRcCreatePattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
    getOperation().walk(
        [](ReussirRcCreateCompoundOp op) { markCompoundAvoidedCopies(op); });
    getOperation().walk(
        [](ReussirRcCreateVariantOp op) { markVariantAvoidedCopies(op); });
  }
};

} // namespace
} // namespace reussir
