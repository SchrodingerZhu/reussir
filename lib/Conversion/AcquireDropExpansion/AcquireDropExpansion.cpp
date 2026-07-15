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
#include <mlir/Dialect/UB/IR/UBOps.h>
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

namespace reussir {

#define GEN_PASS_DEF_REUSSIRACQUIREDROPEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Drop expansion pattern
//===----------------------------------------------------------------------===//

namespace {
struct CellAccess {
  CellType cellType;
  mlir::Value cellRef;
  reussir::AtomicKind atomicKind;
};

static CellAccess borrowCell(mlir::Value cell, mlir::Location loc,
                             mlir::PatternRewriter &rewriter) {
  RcType rcType = llvm::cast<RcType>(cell.getType());
  CellType cellType = llvm::cast<CellType>(rcType.getElementType());
  RefType cellRefType =
      RefType::get(rewriter.getContext(), cellType, Capability::unspecified,
                   rcType.getAtomicKind());
  mlir::Value cellRef =
      ReussirRcBorrowOp::create(rewriter, loc, cellRefType, cell);
  return {cellType, cellRef, rcType.getAtomicKind()};
}

static mlir::Value projectCellSlot(const CellAccess &access, mlir::Location loc,
                                   mlir::PatternRewriter &rewriter) {
  RefType slotRefType =
      RefType::get(rewriter.getContext(), access.cellType.getElementType(),
                   Capability::field, access.atomicKind);
  return ReussirRefProjectOp::create(rewriter, loc, slotRefType, access.cellRef,
                                     rewriter.getIndexAttr(0));
}

// The trailing i1 in-use flag of an exclusive cell, addressed as slot [1].
static mlir::Value projectCellFlag(const CellAccess &access, mlir::Location loc,
                                   mlir::PatternRewriter &rewriter) {
  assert(access.cellType.getExclusive() &&
         "only exclusive cells carry an in-use flag");
  RefType flagRefType =
      RefType::get(rewriter.getContext(), rewriter.getI1Type(),
                   Capability::field, access.atomicKind);
  return ReussirRefProjectOp::create(rewriter, loc, flagRefType, access.cellRef,
                                     rewriter.getIndexAttr(1));
}

static void storeCellFlag(const CellAccess &access, bool inUse,
                          mlir::Location loc, mlir::PatternRewriter &rewriter) {
  mlir::Value flagRef = projectCellFlag(access, loc, rewriter);
  mlir::Value flag = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getIntegerAttr(rewriter.getI1Type(), inUse));
  ReussirRefStoreOp::create(rewriter, loc, flagRef, flag);
}

// An exclusive-cell payload access lowers to a two-armed `scf.if` over the
// in-use flag: while an rmw body runs, the element is moved out of the slot,
// so a reentrant get/set/rmw through an alias of the same cell would clone
// or release stale storage. The in-use arm panics and yields poison results;
// the caller emits the actual access into the (returned) else arm, so even
// at the IR level the panicked path never touches the moved-out slot. Plain
// cells carry no flag and stay unguarded.
static mlir::scf::IfOp createGuardedAccess(const CellAccess &access,
                                           mlir::Location loc,
                                           mlir::PatternRewriter &rewriter,
                                           llvm::StringRef message,
                                           mlir::TypeRange resultTypes) {
  assert(access.cellType.getExclusive() &&
         "only exclusive cell accesses are guarded");
  mlir::Value flagRef = projectCellFlag(access, loc, rewriter);
  mlir::Value flag =
      ReussirRefLoadOp::create(rewriter, loc, rewriter.getI1Type(), flagRef);
  auto unlikelyInUse = ReussirExpectOp::create(rewriter, loc, flag, false);
  auto ifOp = mlir::scf::IfOp::create(rewriter, loc, resultTypes,
                                      unlikelyInUse.getLikely(),
                                      /*addThenBlock=*/true,
                                      /*addElseBlock=*/true);
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  ReussirPanicOp::create(rewriter, loc, rewriter.getStringAttr(message));
  llvm::SmallVector<mlir::Value> poisons;
  for (mlir::Type type : resultTypes)
    poisons.push_back(mlir::ub::PoisonOp::create(rewriter, loc, type));
  mlir::scf::YieldOp::create(rewriter, loc, poisons);
  return ifOp;
}

class DropExpansionPattern : public mlir::OpRewritePattern<ReussirRefDropOp> {
private:
  mlir::LogicalResult rewriteDropCell(CellType cellType, ReussirRefDropOp op,
                                      mlir::PatternRewriter &rewriter) const {
    RefType refType = op.getRef().getType();
    RefType slotType =
        RefType::get(rewriter.getContext(), cellType.getElementType(),
                     Capability::field, refType.getAtomicKind());
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

//===----------------------------------------------------------------------===//
// Cell operation expansion patterns
//===----------------------------------------------------------------------===//

class CellCreateExpansionPattern
    : public mlir::OpRewritePattern<ReussirCellCreateOp> {
public:
  using mlir::OpRewritePattern<ReussirCellCreateOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    CellType cellType =
        llvm::cast<CellType>(op.getCell().getType().getElementType());
    mlir::Value poison =
        mlir::ub::PoisonOp::create(rewriter, op.getLoc(), cellType);
    auto created = ReussirRcCreateOp::create(
        rewriter, op.getLoc(), op.getCell().getType(), poison, op.getToken(),
        mlir::Value{}, mlir::FlatSymbolRefAttr{}, mlir::UnitAttr{});
    CellAccess access = borrowCell(created.getRcPtr(), op.getLoc(), rewriter);
    mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
    ReussirRefStoreOp::create(rewriter, op.getLoc(), slotRef, op.getValue());
    // A fresh exclusive cell starts out not in use.
    if (cellType.getExclusive())
      storeCellFlag(access, /*inUse=*/false, op.getLoc(), rewriter);
    rewriter.replaceOp(op, created.getRcPtr());
    return mlir::success();
  }
};

class CellGetExpansionPattern
    : public mlir::OpRewritePattern<ReussirCellGetOp> {
public:
  using mlir::OpRewritePattern<ReussirCellGetOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellGetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Atomic cells remain intact until LLVM conversion, where the RC payload
    // slot can be loaded with acquire ordering. Expanding this to ref.load
    // would lose the fact that the cell storage itself is atomic (which is
    // independent of the RC box's refcount atomicity).
    CellType cellType =
        llvm::cast<CellType>(op.getCell().getType().getElementType());
    if (cellType.getAtomic())
      return mlir::failure();
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    auto emitLoadAndAcquire = [&]() -> mlir::Value {
      mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
      mlir::Value value = ReussirRefLoadOp::create(
          rewriter, op.getLoc(), access.cellType.getElementType(), slotRef);
      mlir::Type valueType = value.getType();
      // Cloning is triviality-driven: every non-trivial element must acquire
      // the managed ownership reachable from it — a bare RC pointer retains
      // directly, anything else (records, nullables, ...) routes through
      // `ref.acquire`.
      if (!isTriviallyCopyable(valueType)) {
        if (llvm::isa<RcType>(valueType)) {
          ReussirRcIncOp::create(rewriter, op.getLoc(), value);
        } else {
          // Acquire the exact SSA value loaded above, rather than loading
          // the mutable slot a second time. The spill only provides
          // addressability for recursive aggregate acquisition and is
          // eliminated after lowering.
          RefType spilledType = RefType::get(rewriter.getContext(), valueType);
          mlir::Value spilled = ReussirRefSpilledOp::create(
              rewriter, op.getLoc(), spilledType, value);
          ReussirRefAcquireOp::create(rewriter, op.getLoc(), spilled);
        }
      }
      return value;
    };
    if (!access.cellType.getExclusive()) {
      rewriter.replaceOp(op, emitLoadAndAcquire());
      return mlir::success();
    }
    auto guarded =
        createGuardedAccess(access, op.getLoc(), rewriter,
                            "cell.get on an exclusive cell that is in use",
                            {access.cellType.getElementType()});
    rewriter.setInsertionPointToStart(guarded.elseBlock());
    mlir::Value value = emitLoadAndAcquire();
    mlir::scf::YieldOp::create(rewriter, op.getLoc(), value);
    rewriter.replaceOp(op, guarded.getResults());
    return mlir::success();
  }
};

class CellSetExpansionPattern
    : public mlir::OpRewritePattern<ReussirCellSetOp> {
public:
  using mlir::OpRewritePattern<ReussirCellSetOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellSetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    CellType cellType =
        llvm::cast<CellType>(op.getCell().getType().getElementType());
    if (cellType.getAtomic())
      return mlir::failure();
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    auto emitDropAndStore = [&]() {
      mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
      ReussirRefDropOp::create(rewriter, op.getLoc(), slotRef);
      ReussirRefStoreOp::create(rewriter, op.getLoc(), slotRef, op.getValue());
    };
    if (access.cellType.getExclusive()) {
      auto guarded = createGuardedAccess(
          access, op.getLoc(), rewriter,
          "cell.set on an exclusive cell that is in use", {});
      rewriter.setInsertionPointToStart(guarded.elseBlock());
      emitDropAndStore();
      mlir::scf::YieldOp::create(rewriter, op.getLoc());
    } else {
      emitDropAndStore();
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CellRmwExpansionPattern
    : public mlir::OpRewritePattern<ReussirCellRmwOp> {
public:
  using mlir::OpRewritePattern<ReussirCellRmwOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellRmwOp op,
                  mlir::PatternRewriter &rewriter) const override {
    CellType cellType =
        llvm::cast<CellType>(op.getCell().getType().getElementType());
    // Atomic RMW keeps its retryable region until LLVM conversion, which
    // copies the body into a cmpxchg loop. The exclusive form continues to
    // use the in-use flag expansion below.
    if (cellType.getAtomic())
      return mlir::failure();
    // The verifier guarantees an exclusive cell here. The whole
    // read-modify-write becomes the guarded arm: raise the in-use flag for
    // the body so any reentrant access to the cell panics while its element
    // is moved out, and clear it with the replacement store.
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    auto guarded = createGuardedAccess(
        access, op.getLoc(), rewriter,
        "cell.rmw on an exclusive cell that is in use", op->getResultTypes());
    rewriter.setInsertionPointToStart(guarded.elseBlock());
    storeCellFlag(access, /*inUse=*/true, op.getLoc(), rewriter);
    mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
    mlir::Value oldValue = ReussirRefLoadOp::create(
        rewriter, op.getLoc(), access.cellType.getElementType(), slotRef);

    mlir::Block &body = op.getBody().front();
    auto yield = llvm::cast<ReussirCellYieldOp>(body.getTerminator());
    rewriter.inlineBlockBefore(&body, guarded.elseBlock(),
                               guarded.elseBlock()->end(),
                               mlir::ValueRange{oldValue});
    rewriter.setInsertionPoint(yield);
    ReussirRefStoreOp::create(rewriter, yield.getLoc(), slotRef,
                              yield.getNewValue());
    storeCellFlag(access, /*inUse=*/false, yield.getLoc(), rewriter);
    mlir::scf::YieldOp::create(rewriter, yield.getLoc(),
                               yield.getOutput()
                                   ? mlir::ValueRange{yield.getOutput()}
                                   : mlir::ValueRange{});
    rewriter.eraseOp(yield);
    if (op->getNumResults() != 0)
      rewriter.replaceOp(op, guarded.getResults());
    else
      rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CellInUseExpansionPattern
    : public mlir::OpRewritePattern<ReussirCellInUseOp> {
public:
  using mlir::OpRewritePattern<ReussirCellInUseOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellInUseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    mlir::Value flagRef = projectCellFlag(access, op.getLoc(), rewriter);
    rewriter.replaceOp(op,
                       ReussirRefLoadOp::create(rewriter, op.getLoc(),
                                                rewriter.getI1Type(), flagRef));
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
  patterns.add<CellCreateExpansionPattern, CellGetExpansionPattern,
               CellSetExpansionPattern, CellRmwExpansionPattern,
               CellInUseExpansionPattern>(patterns.getContext());
}

} // namespace reussir
