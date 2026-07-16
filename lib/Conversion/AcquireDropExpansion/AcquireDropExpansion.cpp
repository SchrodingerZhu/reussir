//===-- AcquireDropExpansion.cpp - Reussir acquire/drop expansion -*- C++
//-*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Reussir/Conversion/AcquireDropExpansion.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Sync/IR/SyncOps.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRACQUIREDROPEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Drop expansion pattern
//===----------------------------------------------------------------------===//

namespace {
class DropExpansionPattern : public mlir::OpRewritePattern<ReussirRefDropOp> {
private:
  mlir::LogicalResult rewriteDropCell(CellType cellType, ReussirRefDropOp op,
                                      mlir::PatternRewriter &rewriter) const {
    RefType refType = op.getRef().getType();
    RefType slotType =
        RefType::get(rewriter.getContext(), cellType.getElementType(),
                     Capability::field, refType.getAtomicKind());
    // A mutex cell's payload lives behind the lock header, so its drop glue
    // reaches it the same way every other access does: a critical section
    // whose body releases the managed payload (for an RC element, a load and
    // rc.dec). The box count is already zero here, so the section is
    // uncontended by construction; taking it keeps payload addressing uniform
    // and pairs the drop with the lock words' release/acquire chain.
    if (cellType.getMutex()) {
      auto mutexType = mlir::sync::MutexType::get(rewriter.getContext(),
                                                  cellType.getElementType());
      mlir::Value mutexView = ReussirRefToMemrefOp::create(
          rewriter, op.getLoc(), mlir::MemRefType::get({}, mutexType),
          op.getRef());
      auto critical = mlir::sync::SyncMutexCriticalSectionOp::create(
          rewriter, op.getLoc(), mlir::TypeRange{}, mutexView);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      mlir::Block *body = rewriter.createBlock(
          &critical.getBody(), critical.getBody().begin(),
          {mlir::MemRefType::get({}, cellType.getElementType())},
          {op.getLoc()});
      rewriter.setInsertionPointToStart(body);
      mlir::Value slot = ReussirRefFromMemrefOp::create(
          rewriter, op.getLoc(), slotType, body->getArgument(0));
      ReussirRefDropOp::create(rewriter, op.getLoc(), slot);
      mlir::sync::SyncYieldOp::create(rewriter, op.getLoc());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    mlir::Value slot = ReussirRefProjectOp::create(
        rewriter, op.getLoc(), slotType, op.getRef(), rewriter.getIndexAttr(0));
    ReussirRefDropOp::create(rewriter, op.getLoc(), slot);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult rewriteDropArray(ArrayType arrayType, ReussirRefDropOp op,
                                       mlir::PatternRewriter &rewriter) const {
    mlir::Value view = ReussirArrayViewOp::create(
                           rewriter, op.getLoc(),
                           mlir::MemRefType::get(arrayType.getShape(),
                                                 arrayType.getElementType()),
                           op.getRef())
                           .getView();
    auto recurse = [&](auto &&self, mlir::Value currentView,
                       ArrayType currentType) -> mlir::LogicalResult {
      for (int64_t index :
           llvm::seq<int64_t>(0, currentType.getShape().front())) {
        auto idx =
            mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
        if (currentType.getRank() == 1) {
          RefType projectedRefType = rewriter.getType<RefType>(
              currentType.getElementType(), Capability::unspecified);
          auto projected = ReussirArrayProjectOp::create(
              rewriter, op.getLoc(), projectedRefType, currentView,
              idx.getResult());
          if (!isTriviallyCopyable(currentType.getElementType()))
            ReussirRefDropOp::create(rewriter, op.getLoc(),
                                     projected.getProjected());
          continue;
        }
        auto droppedType = currentType.dropFront();
        auto projected = ReussirArrayProjectOp::create(
            rewriter, op.getLoc(),
            mlir::MemRefType::get(droppedType.getShape(),
                                  droppedType.getElementType()),
            currentView, idx.getResult());
        if (mlir::failed(
                self(self, projected.getProjected(), currentType.dropFront())))
          return mlir::failure();
      }
      return mlir::success();
    };

    if (mlir::failed(recurse(recurse, view, arrayType)))
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult rewriteDropRc(RcType rcType, ReussirRefDropOp op,
                                    mlir::PatternRewriter &rewriter) const {
    // Replace drop of ref rc with load then dec
    NullableType nullableType = nullptr;
    if (rcType.getCapability() != Capability::rigid &&
        !llvm::isa<ClosureType>(rcType.getElementType())) {
      auto layout = mlir::DataLayout::closest(op.getOperation());
      RcBoxType rcBoxType = rcType.getInnerBoxType();
      size_t size = layout.getTypeSize(rcBoxType).getFixedValue();
      size_t align = layout.getTypeABIAlignment(rcBoxType);
      TokenType tokenType = TokenType::get(op.getContext(), align, size);
      nullableType = NullableType::get(op.getContext(), tokenType);
    }
    mlir::Value loaded =
        ReussirRefLoadOp::create(rewriter, op.getLoc(), rcType, op.getRef());
    auto dec =
        ReussirRcDecOp::create(rewriter, op.getLoc(), nullableType, loaded,
                               /*destructureTag=*/mlir::IntegerAttr{},
                               /*boundMembers=*/mlir::DenseI64ArrayAttr{});
    // This leaf member-decrement is created *after* token instantiation, so
    // the max-arm token computed above never saw `getTokenType()` — the
    // single source of truth for the box's per-constructor layout.
    // An unpinned non-uniform variant needs the dynamic (runtime-sized)
    // token, else every arm would free at the max-arm width (e.g. a 16-byte
    // leaf freed at 32). Fix the result type to the authoritative one; the
    // result has no users yet, so retyping in place is safe.
    if (nullableType) {
      TokenType correct = dec.getTokenType();
      if (correct != llvm::cast<TokenType>(nullableType.getPtrTy()))
        dec->getResult(0).setType(NullableType::get(op.getContext(), correct));
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  rewriteDropCompound(RecordType recordType, Capability refCap,
                      ReussirRefDropOp op,
                      mlir::PatternRewriter &rewriter) const {
    assert(recordType.isCompound());
    for (auto [idx, memberTy, memberIsField] : llvm::enumerate(
             recordType.getMembers(), recordType.getMemberIsField())) {
      if (memberIsField)
        continue;
      auto projectedTy = getProjectedType(memberTy, false, refCap);
      if (isTriviallyCopyable(projectedTy))
        continue;
      RefType projectedRefTy =
          RefType::get(op.getContext(), projectedTy, refCap);
      mlir::IntegerAttr index = rewriter.getIndexAttr(idx);
      mlir::Value projectedVal = ReussirRefProjectOp::create(
          rewriter, op.getLoc(), projectedRefTy, op.getRef(), index);
      ReussirRefDropOp::create(rewriter, op.getLoc(), projectedVal);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  rewriteDropVariant(RecordType recordType, Capability refCap,
                     ReussirRefDropOp op,
                     mlir::PatternRewriter &rewriter) const {
    assert(recordType.isVariant());
    llvm::SmallVector<mlir::Attribute> tagSets;
    for (auto idx : llvm::seq<int64_t>(0, recordType.getMembers().size()))
      tagSets.push_back(rewriter.getDenseI64ArrayAttr({idx}));
    auto tagSetsAttr = rewriter.getArrayAttr(tagSets);
    auto dispatcher = ReussirRecordDispatchOp::create(
        rewriter, op.getLoc(), mlir::Type{}, op.getRef(), tagSetsAttr,
        tagSets.size());
    for (auto [idx, memberTy, memberIsField] : llvm::enumerate(
             recordType.getMembers(), recordType.getMemberIsField())) {
      auto projectedTy = getProjectedType(memberTy, memberIsField, refCap);
      RefType projectedRefTy =
          RefType::get(op.getContext(), projectedTy, refCap);
      mlir::Block *block = rewriter.createBlock(
          &dispatcher.getRegions()[idx], dispatcher.getRegions()[idx].begin(),
          {projectedRefTy}, {op.getLoc()});
      rewriter.setInsertionPointToStart(block);
      if (!memberIsField && !isTriviallyCopyable(projectedTy))
        ReussirRefDropOp::create(rewriter, op.getLoc(), block->getArgument(0),
                                 true, nullptr);

      ReussirScfYieldOp::create(rewriter, op.getLoc(), nullptr);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  rewriteDropVariant(RecordType recordType, size_t tag, Capability refCap,
                     ReussirRefDropOp op,
                     mlir::PatternRewriter &rewriter) const {
    assert(recordType.isVariant());
    auto targetType = recordType.getMembers()[tag];
    auto targetRefType = rewriter.getType<RefType>(targetType, refCap);
    auto targetRef =
        ReussirRecordCoerceOp::create(rewriter, op.getLoc(), targetRefType,
                                      rewriter.getIndexAttr(tag), op.getRef());
    ReussirRefDropOp::create(rewriter, op.getLoc(), targetRef);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::LogicalResult
  rewriteDropNullable(NullableType nullableType, ReussirRefDropOp op,
                      mlir::PatternRewriter &rewriter) const {
    if (auto rcType = llvm::dyn_cast<RcType>(nullableType.getPtrTy())) {
      mlir::Value loaded = ReussirRefLoadOp::create(rewriter, op.getLoc(),
                                                    nullableType, op.getRef());
      auto dispatcher = ReussirNullableDispatchOp::create(rewriter, op.getLoc(),
                                                          mlir::Type{}, loaded);
      // do nothing if null
      mlir::Block *nullBlock = rewriter.createBlock(
          &dispatcher.getNullRegion(), dispatcher.getNullRegion().begin());
      rewriter.setInsertionPointToStart(nullBlock);
      ReussirScfYieldOp::create(rewriter, op.getLoc(), nullptr);

      // drop inner if not null
      mlir::Block *nonNullBlock = rewriter.createBlock(
          &dispatcher.getNonNullRegion(), dispatcher.getNonNullRegion().begin(),
          {nullableType.getPtrTy()}, {op.getLoc()});
      rewriter.setInsertionPointToStart(nonNullBlock);
      NullableType retNullableTy = nullptr;
      if (rcType.getCapability() != Capability::rigid) {
        auto layout = mlir::DataLayout::closest(op.getOperation());
        RcBoxType rcBoxType = rcType.getInnerBoxType();
        size_t size = layout.getTypeSize(rcBoxType).getFixedValue();
        size_t align = layout.getTypeABIAlignment(rcBoxType);
        TokenType tokenType = TokenType::get(op.getContext(), align, size);
        retNullableTy = NullableType::get(op.getContext(), tokenType);
      }
      auto dec = ReussirRcDecOp::create(
          rewriter, op.getLoc(), retNullableTy, nonNullBlock->getArgument(0),
          /*destructureTag=*/mlir::IntegerAttr{},
          /*boundMembers=*/mlir::DenseI64ArrayAttr{});
      // Route through `getTokenType()` (the per-constructor source of
      // truth); the max-arm token above would free non-uniform arms at the
      // wrong size. Result has no users yet — retyping in place is safe.
      if (retNullableTy) {
        TokenType correct = dec.getTokenType();
        if (correct != llvm::cast<TokenType>(retNullableTy.getPtrTy()))
          dec->getResult(0).setType(
              NullableType::get(op.getContext(), correct));
      }
      ReussirScfYieldOp::create(rewriter, op.getLoc(), nullptr);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

  bool outlineRecord;

  bool shouldOutline(ReussirRefDropOp op, RecordType type) const {
    if (!outlineRecord)
      return false;

    if (isTriviallyCopyable(type))
      return false;

    if (op.getInlined())
      return false;

    if (type.isVariant())
      return !op.getVariant();

    return type.getName() != nullptr;
  }

public:
  DropExpansionPattern(mlir::MLIRContext *context, bool outlineRecord)
      : mlir::OpRewritePattern<ReussirRefDropOp>(context),
        outlineRecord(outlineRecord) {}

  mlir::LogicalResult
  matchAndRewrite(ReussirRefDropOp op,
                  mlir::PatternRewriter &rewriter) const override {
    RefType refType = op.getRef().getType();
    Capability refCap = refType.getCapability();
    if (isTriviallyCopyable(refType.getElementType())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::Type elementType = refType.getElementType();

    return llvm::TypeSwitch<mlir::Type, llvm::LogicalResult>(elementType)
        .Case<RcType>(
            [&](RcType rcType) { return rewriteDropRc(rcType, op, rewriter); })
        .Case<ArrayType>([&](ArrayType arrayType) {
          return rewriteDropArray(arrayType, op, rewriter);
        })
        .Case<CellType>([&](CellType cellType) {
          return rewriteDropCell(cellType, op, rewriter);
        })
        .Case<RecordType>([&](RecordType recordType) {
          if (shouldOutline(op, recordType)) {
            mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
            mlir::func::FuncOp dtor =
                createDtorIfNotExists(moduleOp, recordType, rewriter);
            mlir::func::CallOp::create(rewriter, op.getLoc(), dtor,
                                       op.getRef());
            rewriter.eraseOp(op);
            return llvm::success();
          }
          if (recordType.isCompound())
            return rewriteDropCompound(recordType, refCap, op, rewriter);
          if (op.getVariant())
            return rewriteDropVariant(recordType,
                                      op.getVariant()->getZExtValue(), refCap,
                                      op, rewriter);
          return rewriteDropVariant(recordType, refCap, op, rewriter);
        })
        .Case<NullableType>([&](NullableType nullableType) {
          return rewriteDropNullable(nullableType, op, rewriter);
        })
        .Default([&](mlir::Type) { return mlir::failure(); });
  }
};

//===----------------------------------------------------------------------===//
// Acquire expansion pattern
//===----------------------------------------------------------------------===//

class AcquireExpansionPattern
    : public mlir::OpRewritePattern<ReussirRefAcquireOp> {
private:
  bool outlineRecord;

  bool shouldOutline(ReussirRefAcquireOp op, RecordType type) const {
    if (!outlineRecord)
      return false;

    if (isTriviallyCopyable(type))
      return false;

    if (op.getInlined())
      return false;

    return type.getName() != nullptr;
  }

public:
  AcquireExpansionPattern(mlir::MLIRContext *context, bool outlineRecord)
      : mlir::OpRewritePattern<ReussirRefAcquireOp>(context),
        outlineRecord(outlineRecord) {}

  mlir::LogicalResult
  matchAndRewrite(ReussirRefAcquireOp op,
                  mlir::PatternRewriter &rewriter) const override {
    RefType refType = op.getRef().getType();
    mlir::Type elementType = refType.getElementType();

    if (isTriviallyCopyable(elementType)) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (auto recordType = llvm::dyn_cast<RecordType>(elementType)) {
      if (shouldOutline(op, recordType)) {
        mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
        mlir::func::FuncOp acquireFunc =
            emitOwnershipAcquisitionFuncIfNotExists(moduleOp, recordType,
                                                    rewriter);
        mlir::func::CallOp::create(rewriter, op.getLoc(), acquireFunc,
                                   op.getRef());
        rewriter.eraseOp(op);
        return mlir::success();
      }

      // For variant with known tag, coerce first then acquire
      if (recordType.isVariant() && op.getVariant()) {
        size_t tag = op.getVariant()->getZExtValue();
        auto targetType = recordType.getMembers()[tag];
        auto targetRefType =
            rewriter.getType<RefType>(targetType, refType.getCapability());
        auto targetRef = ReussirRecordCoerceOp::create(
            rewriter, op.getLoc(), targetRefType, rewriter.getIndexAttr(tag),
            op.getRef());
        ReussirRefAcquireOp::create(rewriter, op.getLoc(), targetRef);
        rewriter.eraseOp(op);
        return mlir::success();
      }
    }

    // Route through emitOwnershipAcquisition for all other cases
    if (emitOwnershipAcquisition(op.getRef(), rewriter, op.getLoc()).failed())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// AcquireDropExpansionPass
//===----------------------------------------------------------------------===//

namespace {
struct AcquireDropExpansionPass
    : public impl::ReussirAcquireDropExpansionPassBase<
          AcquireDropExpansionPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateAcquireDropExpansionConversionPatterns(patterns, outlineRecord);
    if (expandDecrement)
      populateRcDecrementExpansionConversionPatterns(patterns);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateAcquireDropExpansionConversionPatterns(
    mlir::RewritePatternSet &patterns, bool outlineRecord) {
  patterns.add<DropExpansionPattern>(patterns.getContext(), outlineRecord);
  patterns.add<AcquireExpansionPattern>(patterns.getContext(), outlineRecord);
}

} // namespace reussir
