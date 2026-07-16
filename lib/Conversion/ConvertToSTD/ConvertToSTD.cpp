//===-- ConvertToSTD.cpp - Reussir standard conversion impl -----*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/ConvertToSTD.h"
#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Transformation/SpecialPointerTag.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/BLAKE3.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

#include "Sync/Conversion/ConvertSyncToSTD.h"
#include "Sync/IR/SyncDialect.h"
#include "Sync/IR/SyncOps.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRCONVERTTOSTDPASS
#include "Reussir/Conversion/Passes.h.inc"
//===----------------------------------------------------------------------===//
// String Pattern Trie
//===----------------------------------------------------------------------===//
namespace {

/// =======================
/// Hashing / naming
/// =======================

std::string decisionFunctionName(mlir::ArrayAttr patterns) {
  llvm::BLAKE3 hasher;
  for (auto attr : patterns.getValue())
    if (auto s = llvm::dyn_cast<mlir::StringAttr>(attr))
      hasher.update(s.getValue());
  return mangledBlake3Symbol("REUSSIR_STRING_DISPATCHER", blake3Words(hasher));
}

struct PatternInfo {
  size_t originalIdx;
  llvm::StringRef pattern;
};

static std::pair<mlir::Value, mlir::Value>
buildDecisionTree(mlir::Location loc, mlir::OpBuilder &builder,
                  mlir::Type indexType, mlir::Type i1Type,
                  mlir::Value currentSlice,
                  llvm::ArrayRef<PatternInfo> patterns) {
  if (patterns.empty()) {
    auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
    auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
    return {poison.getResult(), falseVal.getResult()};
  }

  if (patterns.size() == 1) {
    const auto &p = patterns[0];
    mlir::Value condition;
    if (p.pattern.empty()) {
      auto len = ReussirStrLenOp::create(builder, loc, builder.getIndexType(),
                                         currentSlice);
      auto zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
      condition = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, len.getResult(),
          zero.getResult());
    } else {
      auto startswith = ReussirStrUnsafeStartWithOp::create(
          builder, loc, i1Type, currentSlice, builder.getStringAttr(p.pattern));
      auto len = ReussirStrLenOp::create(builder, loc, builder.getIndexType(),
                                         currentSlice);
      auto expectedLen =
          mlir::arith::ConstantIndexOp::create(builder, loc, p.pattern.size());
      auto lenOk = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, len.getResult(),
          expectedLen.getResult());
      condition = mlir::arith::AndIOp::create(
          builder, loc, startswith.getResult(), lenOk.getResult());
    }

    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{indexType, i1Type}, condition,
        /*addThenRegion=*/true, /*addElseRegion=*/true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto idx =
          mlir::arith::ConstantIndexOp::create(builder, loc, p.originalIdx);
      auto trueVal = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
      mlir::scf::YieldOp::create(
          builder, loc, mlir::ValueRange{idx.getResult(), trueVal.getResult()});
    }

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(
          builder, loc,
          mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    }
    return {ifOp.getResult(0), ifOp.getResult(1)};
  }

  // Find LCP
  llvm::StringRef first = patterns[0].pattern;
  size_t lcpLen = first.size();
  for (size_t i = 1; i < patterns.size(); ++i) {
    size_t j = 0;
    while (j < lcpLen && j < patterns[i].pattern.size() &&
           first[j] == patterns[i].pattern[j]) {
      j++;
    }
    lcpLen = j;
  }

  if (lcpLen > 0) {
    auto lcp = first.substr(0, lcpLen);
    auto len = ReussirStrLenOp::create(builder, loc, builder.getIndexType(),
                                       currentSlice);
    auto minLen = mlir::arith::ConstantIndexOp::create(builder, loc, lcpLen);
    auto lenOk = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::uge, len.getResult(),
        minLen.getResult());

    auto ifLen = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{indexType, i1Type}, lenOk.getResult(),
        /*addThenRegion=*/true, /*addElseRegion=*/true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifLen.getThenRegion().front());

      auto startswith = ReussirStrUnsafeStartWithOp::create(
          builder, loc, i1Type, currentSlice, builder.getStringAttr(lcp));

      auto ifMatch = mlir::scf::IfOp::create(
          builder, loc, mlir::TypeRange{indexType, i1Type},
          startswith.getResult(),
          /*addThenRegion=*/true, /*addElseRegion=*/true);

      {
        mlir::OpBuilder::InsertionGuard thenGuard(builder);
        builder.setInsertionPointToStart(&ifMatch.getThenRegion().front());
        auto offset =
            mlir::arith::ConstantIndexOp::create(builder, loc, lcpLen);
        auto nextSlice =
            ReussirStrSliceOp::create(builder, loc, currentSlice.getType(),
                                      currentSlice, offset.getResult());

        llvm::SmallVector<PatternInfo> nextPatterns;
        for (const auto &p : patterns) {
          nextPatterns.push_back({p.originalIdx, p.pattern.substr(lcpLen)});
        }
        auto res = buildDecisionTree(loc, builder, indexType, i1Type,
                                     nextSlice.getResult(), nextPatterns);
        mlir::scf::YieldOp::create(builder, loc,
                                   mlir::ValueRange{res.first, res.second});
      }

      {
        mlir::OpBuilder::InsertionGuard elseGuard(builder);
        builder.setInsertionPointToStart(&ifMatch.getElseRegion().front());
        auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
        auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
        mlir::scf::YieldOp::create(
            builder, loc,
            mlir::ValueRange{poison.getResult(), falseVal.getResult()});
      }
      mlir::scf::YieldOp::create(builder, loc, ifMatch.getResults());
    }

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifLen.getElseRegion().front());
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(
          builder, loc,
          mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    }

    return {ifLen.getResult(0), ifLen.getResult(1)};
  }

  // LCP is 0, dispatch on byte
  llvm::SmallVector<PatternInfo> emptyPatterns;
  llvm::SmallVector<PatternInfo> nonEmptyPatterns;
  for (const auto &p : patterns) {
    if (p.pattern.empty()) {
      emptyPatterns.push_back(p);
    } else {
      nonEmptyPatterns.push_back(p);
    }
  }

  auto len = ReussirStrLenOp::create(builder, loc, builder.getIndexType(),
                                     currentSlice);
  auto zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  auto isZero =
      mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq,
                                  len.getResult(), zero.getResult());

  auto ifZero = mlir::scf::IfOp::create(
      builder, loc, mlir::TypeRange{indexType, i1Type}, isZero.getResult(),
      /*addThenRegion=*/true, /*addElseRegion=*/true);

  // Then: len == 0
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifZero.getThenRegion().front());
    if (!emptyPatterns.empty()) {
      auto idx = mlir::arith::ConstantIndexOp::create(
          builder, loc, emptyPatterns[0].originalIdx);
      auto trueVal = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
      mlir::scf::YieldOp::create(
          builder, loc, mlir::ValueRange{idx.getResult(), trueVal.getResult()});
    } else {
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(
          builder, loc,
          mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    }
  }

  // Else: len > 0, dispatch on byte
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifZero.getElseRegion().front());

    if (nonEmptyPatterns.empty()) {
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(
          builder, loc,
          mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    } else {
      auto byteAtZero = ReussirStrUnsafeByteAtOp::create(
          builder, loc, builder.getI8Type(), currentSlice, zero.getResult());
      auto byteIndex = mlir::arith::IndexCastOp::create(
          builder, loc, builder.getIndexType(), byteAtZero.getResult());

      llvm::SmallVector<int64_t> cases;
      llvm::MapVector<uint8_t, llvm::SmallVector<PatternInfo>> groups;

      for (const auto &p : nonEmptyPatterns) {
        uint8_t b = static_cast<uint8_t>(p.pattern[0]);
        if (groups.find(b) == groups.end()) {
          cases.push_back(b);
        }
        groups[b].push_back({p.originalIdx, p.pattern.substr(1)});
      }

      auto one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      auto nextSlice = ReussirStrSliceOp::create(
          builder, loc, currentSlice.getType(), currentSlice, one.getResult());

      auto switchOp = mlir::scf::IndexSwitchOp::create(
          builder, loc, mlir::TypeRange{indexType, i1Type},
          byteIndex.getResult(), cases, cases.size());

      for (auto [idx, b] : llvm::enumerate(cases)) {
        auto &region = switchOp.getCaseRegions()[idx];
        region.emplaceBlock();
        mlir::OpBuilder::InsertionGuard caseGuard(builder);
        builder.setInsertionPointToStart(&region.front());
        auto res = buildDecisionTree(loc, builder, indexType, i1Type,
                                     nextSlice.getResult(), groups[b]);
        mlir::scf::YieldOp::create(builder, loc,
                                   mlir::ValueRange{res.first, res.second});
      }

      // Default region
      {
        auto &region = switchOp.getDefaultRegion();
        region.emplaceBlock();
        mlir::OpBuilder::InsertionGuard defaultGuard(builder);
        builder.setInsertionPointToStart(&region.front());
        auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
        auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
        mlir::scf::YieldOp::create(
            builder, loc,
            mlir::ValueRange{poison.getResult(), falseVal.getResult()});
      }
      mlir::scf::YieldOp::create(builder, loc, switchOp.getResults());
    }
  }

  return {ifZero.getResult(0), ifZero.getResult(1)};
}

std::string emitDecisionFunction(mlir::ModuleOp module,
                                 mlir::OpBuilder &builder,
                                 mlir::ArrayAttr patterns) {
  std::string funcName = decisionFunctionName(patterns);
  if (module.lookupSymbol<mlir::func::FuncOp>(funcName)) {
    return funcName;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto indexType = builder.getIndexType();
  auto i1Type = builder.getI1Type();

  auto strType =
      reussir::StrType::get(builder.getContext(), reussir::LifeScope::local);
  auto funcType = builder.getFunctionType({strType}, {indexType, i1Type});

  auto func =
      mlir::func::FuncOp::create(builder, module.getLoc(), funcName, funcType);
  func.setVisibility(mlir::SymbolTable::Visibility::Private);
  func->setAttr("llvm.linkage",
                mlir::LLVM::LinkageAttr::get(builder.getContext(),
                                             mlir::LLVM::Linkage::Internal));
  inheritSanitizerPassthrough(module, func);
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::SmallVector<PatternInfo> patternInfos;
  for (auto [idx, attr] : llvm::enumerate(patterns.getValue())) {
    patternInfos.push_back(
        {idx, llvm::cast<mlir::StringAttr>(attr).getValue()});
  }

  auto res = buildDecisionTree(module.getLoc(), builder, indexType, i1Type,
                               entry->getArgument(0), patternInfos);

  mlir::func::ReturnOp::create(builder, module.getLoc(),
                               mlir::ValueRange{res.first, res.second});

  return funcName;
}

} // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct ReussirNullableDispatchOpRewritePattern
    : public mlir::OpConversionPattern<ReussirNullableDispatchOp> {
  using OpConversionPattern::OpConversionPattern;

  // Whether `result` of an expanded decrement's scf.if is non-null exactly
  // when the if's condition holds — i.e. it is the decrement's *own* token
  // (then yields `nullable.create` of a pointer, else yields a null
  // `nullable.create`). Token reuse widens these ifs with extra results that
  // carry *inner* member-decrement tokens out (see `escapeTrappedTokensOnce`);
  // such a result is yielded from a nested if and its nullness depends on the
  // inner decrement's count, not this condition, so it must not match.
  static bool tokenMatchesCondition(mlir::scf::IfOp ifOp,
                                    mlir::OpResult result) {
    unsigned idx = result.getResultNumber();
    auto thenCreate = ifOp.thenYield()
                          .getOperand(idx)
                          .getDefiningOp<ReussirNullableCreateOp>();
    auto elseCreate = ifOp.elseYield()
                          .getOperand(idx)
                          .getDefiningOp<ReussirNullableCreateOp>();
    return thenCreate && elseCreate && thenCreate.getPtr() &&
           !elseCreate.getPtr();
  }

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableDispatchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // First, create a check operation to get the null flag from the input.
    mlir::Value flag = reussir::ReussirNullableCheckOp::create(
        rewriter, op.getLoc(), op.getNullable());
    // mark expect not null
    if (op->hasAttr(REUSSIR_EXPANDED_ENSURE_ATTR))
      flag =
          reussir::ReussirExpectOp::create(rewriter, op.getLoc(), flag, true);

    auto scfIfOp =
        mlir::scf::IfOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                flag, /*addThenRegion=*/true,
                                /*addElseRegion=*/true);
    if (op->hasAttr(REUSSIR_EXPANDED_ENSURE_ATTR))
      if (auto producerIf = mlir::dyn_cast_if_present<mlir::scf::IfOp>(
              op.getNullable().getDefiningOp()))
        if (producerIf->hasAttr(kExpandedDecrementAttr) &&
            tokenMatchesCondition(producerIf,
                                  llvm::cast<mlir::OpResult>(op.getNullable())))
          scfIfOp.getConditionMutable().assign(producerIf.getCondition());
    // first, do the easy part, for else region, we can just inline the
    // operation
    rewriter.inlineBlockBefore(&*op.getNullRegion().begin(),
                               &*scfIfOp.getElseRegion().begin(),
                               scfIfOp.getElseRegion().begin()->begin());

    // Now, for the then region, we first create the coerced value
    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    auto coerced = reussir::ReussirNullableCoerceOp::create(
        rewriter, op.getLoc(), op.getNullable().getType().getPtrTy(),
        op.getNullable());
    // Then we inline the region, supplying coerced value as the argument
    rewriter.inlineBlockBefore(
        &*op.getNonNullRegion().begin(), &*scfIfOp.getThenRegion().begin(),
        scfIfOp.getThenRegion().begin()->end(), mlir::ValueRange{coerced});
    if (op->hasAttr(REUSSIR_EXPANDED_ENSURE_ATTR))
      scfIfOp->setAttr(REUSSIR_EXPANDED_ENSURE_ATTR, rewriter.getUnitAttr());
    rewriter.replaceOp(op, scfIfOp);
    return mlir::success();
  }
};

struct ReussirRecordDispatchOpRewritePattern
    : public mlir::OpConversionPattern<ReussirRecordDispatchOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  // Inline the arm region `idx` into `block`; a singleton arm receives the
  // scrutinee coerced to its payload type as the block argument.
  static void inlineArm(ReussirRecordDispatchOp op,
                        mlir::PatternRewriter &rewriter, unsigned idx,
                        std::optional<int64_t> singletonTag,
                        mlir::Block *block) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    llvm::SmallVector<mlir::Value, 1> args;
    if (singletonTag) {
      RefType variantRef = op.getVariant().getType();
      RecordType recordType =
          llvm::cast<RecordType>(variantRef.getElementType());
      mlir::Type targetVariantType =
          getProjectedType(recordType.getMembers()[*singletonTag],
                           recordType.getMemberIsField()[*singletonTag],
                           variantRef.getCapability());
      RefType coercedType = RefType::get(
          rewriter.getContext(), targetVariantType, variantRef.getCapability());
      auto coerced = reussir::ReussirRecordCoerceOp::create(
          rewriter, op.getLoc(), coercedType,
          rewriter.getIndexAttr(*singletonTag), op.getVariant());
      args.push_back(coerced);
    }
    rewriter.inlineBlockBefore(&op->getRegion(idx).front(), block, block->end(),
                               args);
  }

  // The generic dispatch over the arm subset `setIndices`: read the tag and
  // switch on it (through a pre-dispatcher when an arm covers several
  // tags). Emitted at the rewriter's insertion point; returns the switch's
  // results.
  static mlir::scf::IndexSwitchOp
  emitTagDispatch(ReussirRecordDispatchOp op, mlir::PatternRewriter &rewriter,
                  llvm::ArrayRef<unsigned> setIndices) {
    auto tag = reussir::ReussirRecordTagOp::create(rewriter, op.getLoc(),
                                                   op.getVariant());
    bool allSingletons = llvm::all_of(setIndices, [&](unsigned idx) {
      return llvm::cast<mlir::DenseI64ArrayAttr>(op.getTagSets()[idx]).size() ==
             1;
    });

    mlir::Value outerSwitchValue;
    llvm::SmallVector<int64_t> outerSwitchCases;
    if (allSingletons) {
      outerSwitchValue = tag.getResult();
      for (unsigned idx : setIndices)
        outerSwitchCases.push_back(
            llvm::cast<mlir::DenseI64ArrayAttr>(op.getTagSets()[idx])[0]);
    } else {
      // Pre-dispatch: map every covered tag to its arm's case position.
      llvm::SmallVector<int64_t> allTags;
      llvm::DenseMap<int64_t, int64_t> tagToCase;
      for (auto [pos, idx] : llvm::enumerate(setIndices))
        for (int64_t t :
             llvm::cast<mlir::DenseI64ArrayAttr>(op.getTagSets()[idx])
                 .asArrayRef()) {
          allTags.push_back(t);
          tagToCase[t] = static_cast<int64_t>(pos);
        }
      auto preDispatch = mlir::scf::IndexSwitchOp::create(
          rewriter, op.getLoc(), rewriter.getIndexType(), tag.getResult(),
          allTags, allTags.size());
      for (auto [t, region] :
           llvm::zip(allTags, preDispatch.getCaseRegions())) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block *block = rewriter.createBlock(&region, region.begin());
        rewriter.setInsertionPointToStart(block);
        auto constantIdx = mlir::arith::ConstantIndexOp::create(
            rewriter, op.getLoc(), tagToCase[t]);
        mlir::scf::YieldOp::create(rewriter, op.getLoc(),
                                   constantIdx->getResults());
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block *block =
            rewriter.createBlock(&preDispatch.getDefaultRegion(),
                                 preDispatch.getDefaultRegion().begin());
        rewriter.setInsertionPointToStart(block);
        auto poison = mlir::ub::PoisonOp::create(rewriter, op.getLoc(),
                                                 rewriter.getIndexType());
        mlir::scf::YieldOp::create(rewriter, op.getLoc(), poison->getResults());
      }
      outerSwitchValue = preDispatch.getResult(0);
      outerSwitchCases = llvm::to_vector(
          llvm::seq<int64_t>(0, static_cast<int64_t>(setIndices.size())));
    }

    auto indexSwitchOp = mlir::scf::IndexSwitchOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), outerSwitchValue,
        outerSwitchCases, outerSwitchCases.size());
    // mark default region as unreachable
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      mlir::Block *block =
          rewriter.createBlock(&indexSwitchOp.getDefaultRegion(),
                               indexSwitchOp.getDefaultRegion().begin());
      rewriter.setInsertionPointToStart(block);
      llvm::SmallVector<mlir::Value, 1> poisonValues;
      if (op.getValue()) {
        auto poison = mlir::ub::PoisonOp::create(rewriter, op.getLoc(),
                                                 op.getValue().getType());
        poisonValues.push_back(poison);
      }
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), poisonValues);
    }
    for (auto [idx, region] :
         llvm::zip(setIndices, indexSwitchOp.getCaseRegions())) {
      // `createBlock` moves the insertion point into the case; keep the
      // caller's position (right after the switch) intact.
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto tagArray = llvm::cast<mlir::DenseI64ArrayAttr>(op.getTagSets()[idx]);
      mlir::Block *block = rewriter.createBlock(&region, region.begin());
      inlineArm(op, rewriter, idx,
                tagArray.size() == 1 ? std::optional<int64_t>(tagArray[0])
                                     : std::nullopt,
                block);
    }
    return indexSwitchOp;
  }

public:
  mlir::LogicalResult
  matchAndRewrite(ReussirRecordDispatchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Under the immortal encoding a nullary arm's value is always the
    // per-tag immediate (the scheme invariant at `kSpecialPtrTagAttr`), so
    // membership is one pointer compare — while the generic tag read is a
    // dependent *load* through the dummy box on every dispatch (issue
    // #400). Peel such arms off into `rc.compare_immortal` guards and keep
    // the unchanged tag dispatch for the rest. The TBI encoding already
    // reads tags from the pointer's top byte without touching memory, so
    // only `immortal` peels.
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto encoding =
        moduleOp->getAttrOfType<mlir::StringAttr>(kSpecialPtrTagAttr);
    mlir::TypedValue<RcType> scrutinee;
    llvm::SmallVector<unsigned> peeled;
    llvm::SmallVector<unsigned> remaining;
    RecordType recordType =
        llvm::cast<RecordType>(op.getVariant().getType().getElementType());
    if (encoding && encoding.getValue() == kSpecialPtrTagImmortal)
      if (auto borrow = op.getVariant().getDefiningOp<ReussirRcBorrowOp>();
          borrow && borrow.getRcPtr().getType().mayCarrySpecialPointerTag())
        scrutinee = borrow.getRcPtr();
    for (auto [idx, tagSet] : llvm::enumerate(op.getTagSets())) {
      auto tagArray = llvm::cast<mlir::DenseI64ArrayAttr>(tagSet);
      auto arm =
          tagArray.size() == 1
              ? llvm::dyn_cast<RecordType>(recordType.getMembers()[tagArray[0]])
              : RecordType{};
      if (scrutinee && arm && arm.isCompound() && arm.getComplete() &&
          arm.getMembers().empty())
        peeled.push_back(idx);
      else
        remaining.push_back(idx);
    }

    if (peeled.empty()) {
      auto dispatch = emitTagDispatch(op, rewriter, remaining);
      rewriter.replaceOp(op, dispatch->getResults());
      return mlir::success();
    }

    // One `scf.if` per peeled arm; the innermost else carries the tag
    // dispatch over the remaining arms (or nothing — the guards are then
    // exhaustive).
    mlir::scf::IfOp outermost;
    for (unsigned idx : peeled) {
      int64_t tag =
          llvm::cast<mlir::DenseI64ArrayAttr>(op.getTagSets()[idx])[0];
      auto isImmediate = ReussirRcCompareImmortalOp::create(
          rewriter, op.getLoc(), rewriter.getI1Type(), scrutinee,
          rewriter.getIndexAttr(tag));
      auto ifOp =
          mlir::scf::IfOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                  isImmediate.getResult(), true, true);
      inlineArm(op, rewriter, idx, tag, ifOp.thenBlock());
      if (!outermost)
        outermost = ifOp;
      else
        mlir::scf::YieldOp::create(rewriter, op.getLoc(), ifOp->getResults());
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
    }
    if (remaining.empty()) {
      llvm::SmallVector<mlir::Value, 1> poisonValues;
      if (op.getValue())
        poisonValues.push_back(mlir::ub::PoisonOp::create(
            rewriter, op.getLoc(), op.getValue().getType()));
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), poisonValues);
    } else if (remaining.size() == 1) {
      // The guards decided the dispatch: the last arm needs no tag read at
      // all — inline it as the residual else (its own yield terminates the
      // branch). This is the common two-arm shape (rbtree's Leaf/Branch,
      // QList's Nil/Cons): the hot recursive arm keeps its exact
      // pre-rewrite body with zero dispatch memory traffic.
      auto tagArray = llvm::cast<mlir::DenseI64ArrayAttr>(
          op.getTagSets()[remaining.front()]);
      mlir::Block *elseBlock = rewriter.getInsertionBlock();
      inlineArm(op, rewriter, remaining.front(),
                tagArray.size() == 1 ? std::optional<int64_t>(tagArray[0])
                                     : std::nullopt,
                elseBlock);
    } else {
      auto dispatch = emitTagDispatch(op, rewriter, remaining);
      rewriter.setInsertionPointAfter(dispatch);
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), dispatch->getResults());
    }
    rewriter.replaceOp(op, outermost->getResults());
    return mlir::success();
  }
};

struct ReussirClosureUniqifyOpRewritePattern
    : public mlir::OpConversionPattern<ReussirClosureUniqifyOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirClosureUniqifyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Create a check operation to see if the closure is unique
    auto isUnique = reussir::ReussirRcIsUniqueOp::create(rewriter, op.getLoc(),
                                                         op.getClosure());

    // Create an SCF if-else operation
    auto scfIfOp =
        mlir::scf::IfOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                isUnique, /*addThenRegion=*/true,
                                /*addElseRegion=*/true);

    // In the then region (closure is unique), just return the original
    // closure
    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    mlir::scf::YieldOp::create(rewriter, op.getLoc(), op.getClosure());

    // In the else region (closure is not unique), clone the closure, dec the
    // original rc pointer
    rewriter.setInsertionPointToStart(&scfIfOp.getElseRegion().front());
    auto cloned = reussir::ReussirClosureCloneOp::create(
        rewriter, op.getLoc(), op.getClosure().getType(), op.getClosure());
    reussir::ReussirRcDecOp::create(rewriter, op.getLoc(),
                                    /*nullableToken=*/mlir::Type{},
                                    op.getClosure(),
                                    /*destructureTag=*/mlir::IntegerAttr{},
                                    /*boundMembers=*/mlir::DenseI64ArrayAttr{});
    mlir::scf::YieldOp::create(rewriter, op.getLoc(), cloned.getResult());

    rewriter.replaceOp(op, scfIfOp);
    return mlir::success();
  }
};

// Expand a FUSED `closure.eval` (one carrying a `with` pack — see the op
// documentation and `reussir-closure-beta-reduction`) into its defining
// sequence: an unchecked `closure.apply` per argument and the plain eval.
// No `closure.uniqify` is emitted: the fused form carries `apply`'s
// contract — uniqueness is the producer's obligation, and whenever a check
// was genuinely needed the beta-reduction pass left the original
// bottom-most uniqify in place as this op's operand. Only the redundant
// intermediate checks of the folded chain were removed.
struct ReussirClosureEvalOpRewritePattern
    : public mlir::OpConversionPattern<ReussirClosureEvalOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getArgs().empty())
      return mlir::failure();
    RcType rcType = op.getClosure().getType();
    auto closureType = llvm::cast<ClosureType>(rcType.getEleTy());
    auto inputTypes = closureType.getInputTypes();
    mlir::Value cur = op.getClosure();
    for (auto [index, arg] : llvm::enumerate(op.getArgs())) {
      auto appliedType =
          RcType::get(rewriter.getContext(),
                      ClosureType::get(rewriter.getContext(),
                                       inputTypes.drop_front(index + 1),
                                       closureType.getOutputType()),
                      rcType.getCapability(), rcType.getAtomicKind());
      cur = ReussirClosureApplyOp::create(rewriter, op.getLoc(), appliedType,
                                          arg, cur);
    }
    if (op.getNumResults()) {
      rewriter.replaceOpWithNewOp<ReussirClosureEvalOp>(
          op, op.getResult().getType(), cur, mlir::ValueRange{});
    } else {
      ReussirClosureEvalOp::create(rewriter, op.getLoc(), mlir::Type(), cur,
                                   mlir::ValueRange{});
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
};

static void cloneArrayWithUniqueViewBody(ReussirArrayWithUniqueViewOp op,
                                         mlir::Value arrayValue,
                                         mlir::Value viewValue,
                                         mlir::PatternRewriter &rewriter) {
  mlir::IRMapping mapping;
  mapping.map(op.getBody().front().getArgument(0), viewValue);
  for (mlir::Operation &nestedOp : op.getBody().front().without_terminator())
    rewriter.clone(nestedOp, mapping);

  auto yieldOp =
      llvm::cast<ReussirScfYieldOp>(op.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value> yieldedValues;
  if (yieldOp.getNumOperands() == 0 && op.getResult() &&
      op.getResult().getType() == op.getArray().getType()) {
    yieldedValues.push_back(arrayValue);
  } else {
    yieldedValues.reserve(yieldOp.getNumOperands());
    for (mlir::Value operand : yieldOp->getOperands())
      yieldedValues.push_back(mapping.lookupOrDefault(operand));
  }
  mlir::scf::YieldOp::create(rewriter, op.getLoc(), yieldedValues);
}

static mlir::MemRefType getArrayViewMemRefType(ArrayType arrayType) {
  return mlir::MemRefType::get(arrayType.getShape(),
                               arrayType.getElementType());
}

static mlir::Value
materializeArrayViewValue(mlir::Location loc, mlir::PatternRewriter &rewriter,
                          ArrayType arrayType, mlir::Value ref,
                          mlir::Type viewType, bool writable) {
  auto memrefType = getArrayViewMemRefType(arrayType);
  auto memrefView =
      ReussirArrayViewOp::create(rewriter, loc, memrefType, ref).getView();
  if (llvm::isa<mlir::MemRefType>(viewType))
    return memrefView;

  auto tensorType = llvm::cast<mlir::RankedTensorType>(viewType);
  return mlir::bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                 memrefView,
                                                 /*restrict=*/true, writable)
      .getResult();
}

struct ReussirArrayViewOpRewritePattern
    : public mlir::OpConversionPattern<ReussirArrayViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirArrayViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensorType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getView().getType());
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "expected tensor array view");

    auto arrayType = llvm::cast<ArrayType>(
        llvm::cast<RefType>(op.getRef().getType()).getElementType());
    auto value = materializeArrayViewValue(op.getLoc(), rewriter, arrayType,
                                           adaptor.getRef(), tensorType,
                                           /*writable=*/false);
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

struct ReussirArrayWithUniqueViewOpRewritePattern
    : public mlir::OpConversionPattern<ReussirArrayWithUniqueViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirArrayWithUniqueViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    RcType rcType = op.getArray().getType();
    ArrayType arrayType = llvm::cast<ArrayType>(rcType.getElementType());
    mlir::Type viewType = op.getBody().front().getArgument(0).getType();

    auto makeBorrowedView = [&](mlir::Value array) -> mlir::Value {
      auto borrowedType = RefType::get(rewriter.getContext(), arrayType);
      auto borrowed =
          ReussirRcBorrowOp::create(rewriter, loc, borrowedType, array);
      return materializeArrayViewValue(loc, rewriter, arrayType,
                                       borrowed.getResult(), viewType,
                                       /*writable=*/true);
    };

    auto makeClonedArray = [&]() -> std::pair<mlir::Value, mlir::Value> {
      auto borrowedType = RefType::get(rewriter.getContext(), arrayType);
      auto srcRef =
          ReussirRcBorrowOp::create(rewriter, loc, borrowedType, op.getArray());
      RcBoxType rcBoxType = RcBoxType::get(rewriter.getContext(), arrayType,
                                           /*regional=*/false);
      auto dataLayout = mlir::DataLayout::closest(op.getOperation());
      TokenType tokenType = TokenType::get(
          rewriter.getContext(), dataLayout.getTypeABIAlignment(rcBoxType),
          dataLayout.getTypeSize(rcBoxType).getFixedValue());
      auto token = ReussirTokenAllocOp::create(rewriter, loc, tokenType);
      auto poison = mlir::ub::PoisonOp::create(rewriter, loc, arrayType);
      auto cloned = ReussirRcCreateOp::create(
          rewriter, loc, rcType, poison.getResult(), token.getResult(),
          mlir::Value{}, mlir::FlatSymbolRefAttr{}, mlir::UnitAttr{});
      auto dstRef = ReussirRcBorrowOp::create(rewriter, loc, borrowedType,
                                              cloned.getResult());
      ReussirRefMemcpyOp::create(rewriter, loc, srcRef.getResult(),
                                 dstRef.getResult());
      ReussirRefAcquireOp::create(rewriter, loc, dstRef.getResult(), false,
                                  nullptr);

      auto refCount = ReussirRcFetchOp::create(rewriter, loc, op.getArray());
      auto decremented = mlir::arith::SubIOp::create(
          rewriter, loc, refCount.getRefCount(),
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
      ReussirRcSetOp::create(rewriter, loc, op.getArray(),
                             decremented.getResult());
      return {cloned.getResult(), dstRef.getResult()};
    };

    auto isUnique =
        reussir::ReussirRcIsUniqueOp::create(rewriter, loc, op.getArray());
    auto scfIfOp = mlir::scf::IfOp::create(rewriter, loc, op->getResultTypes(),
                                           isUnique, /*addThenRegion=*/true,
                                           /*addElseRegion=*/true);

    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    cloneArrayWithUniqueViewBody(op, op.getArray(),
                                 makeBorrowedView(op.getArray()), rewriter);

    rewriter.setInsertionPointToStart(&scfIfOp.getElseRegion().front());
    auto [clonedArray, clonedRef] = makeClonedArray();
    auto clonedView =
        materializeArrayViewValue(loc, rewriter, arrayType, clonedRef, viewType,
                                  /*writable=*/true);
    cloneArrayWithUniqueViewBody(op, clonedArray, clonedView, rewriter);

    rewriter.replaceOp(op, scfIfOp.getResults());
    return mlir::success();
  }
};

struct ReussirScfYieldOpRewritePattern
    : public mlir::OpConversionPattern<ReussirScfYieldOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirScfYieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op->getOperands());
    return mlir::success();
  }
};

struct ReussirTokenEnsureOpRewritePattern
    : public mlir::OpConversionPattern<ReussirTokenEnsureOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirTokenEnsureOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto nullableDispatchOp = ReussirNullableDispatchOp::create(
        rewriter, op.getLoc(), op.getType(), op.getNullableToken());

    {
      mlir::Block *thenBlock =
          rewriter.createBlock(&nullableDispatchOp.getNonNullRegion(), {},
                               op.getType(), {op.getLoc()});
      rewriter.setInsertionPointToStart(thenBlock);
      mlir::Value tokenSrc = thenBlock->getArgument(0);
      if (auto scfIf = mlir::dyn_cast_if_present<mlir::scf::IfOp>(
              op.getNullableToken().getDefiningOp()))
        if (scfIf->getAttr(kExpandedDecrementAttr)) {
          // Trace the *matching result* of the expanded decrement — an
          // escaped nested token rides along as an extra result, and
          // rebuilding it from result 0's box would alias two creates onto
          // one allocation. When the traced operand is not a direct
          // reinterpret (e.g. it is a nested decrement's result), keep the
          // dispatched payload argument, which is always correct.
          unsigned resultIndex =
              llvm::cast<mlir::OpResult>(op.getNullableToken())
                  .getResultNumber();
          auto thenYieldOp = mlir::dyn_cast_if_present<mlir::scf::YieldOp>(
              scfIf.getThenRegion().back().getTerminator());
          auto nullCreateOp =
              mlir::dyn_cast_if_present<reussir::ReussirNullableCreateOp>(
                  thenYieldOp->getOperands()[resultIndex].getDefiningOp());
          auto reinterpretOp =
              nullCreateOp
                  ? mlir::dyn_cast_if_present<reussir::ReussirRcReinterpretOp>(
                        nullCreateOp.getPtr().getDefiningOp())
                  : nullptr;
          // it is safe since RC must dominate this path
          if (reinterpretOp)
            tokenSrc = reussir::ReussirRcReinterpretOp::create(
                rewriter, op.getLoc(), op.getType(), reinterpretOp.getRcPtr());
        }
      auto launderedToken = ReussirTokenLaunderOp::create(
          rewriter, op.getLoc(), op.getType(), tokenSrc);
      mlir::scf::YieldOp::create(rewriter, op.getLoc(),
                                 launderedToken->getResults());
    }
    {
      mlir::Block *elseBlock =
          rewriter.createBlock(&nullableDispatchOp.getNullRegion());
      rewriter.setInsertionPointToStart(elseBlock);
      auto allocatedToken =
          ReussirTokenAllocOp::create(rewriter, op.getLoc(), op.getType());
      mlir::scf::YieldOp::create(rewriter, op.getLoc(),
                                 allocatedToken->getResults());
    }
    nullableDispatchOp->setAttr(REUSSIR_EXPANDED_ENSURE_ATTR,
                                rewriter.getUnitAttr());
    rewriter.replaceOp(op, nullableDispatchOp);
    return mlir::success();
  }
};

/// `token.free` of a *nullable* token lowers to an unconditional
/// `__reussir_deallocate` call whose null guard lives inside the runtime — so
/// the null case (the shared branch of an expanded rc decrement) still pays a
/// full call to do nothing. Guard it here, while structured control flow is
/// still available: coerce and free only when non-null. Plain-token frees are
/// left for the basic-ops lowering (their pointer is statically non-null).
struct ReussirNullableTokenFreeOpRewritePattern
    : public mlir::OpConversionPattern<ReussirTokenFreeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirTokenFreeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto nullableType = llvm::dyn_cast<NullableType>(op.getToken().getType());
    if (!nullableType)
      return mlir::failure();
    auto tokenType = llvm::dyn_cast<TokenType>(nullableType.getPtrTy());
    if (!tokenType)
      return mlir::failure();
    mlir::Location loc = op.getLoc();
    // `nullable.check`'s flag is true when non-null (the dispatch lowering
    // above routes the non-null region through the then-branch the same way).
    mlir::Value flag =
        reussir::ReussirNullableCheckOp::create(rewriter, loc, op.getToken());
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{}, flag,
                                        /*addThenRegion=*/true,
                                        /*addElseRegion=*/false);
    {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto coerced = reussir::ReussirNullableCoerceOp::create(
          rewriter, loc, tokenType, op.getToken());
      ReussirTokenFreeOp::create(rewriter, loc, coerced);
      mlir::scf::YieldOp::create(rewriter, loc);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReussirStrByteAtOpRewritePattern
    : public mlir::OpConversionPattern<ReussirStrByteAtOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirStrByteAtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Get string length
    auto lenOp = reussir::ReussirStrLenOp::create(
        rewriter, loc, rewriter.getIndexType(), op.getStr());

    // Check if index is within bounds (index < len)
    auto inBounds = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::ult, op.getIndex(),
        lenOp.getResult());

    // Create if-else block
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, op.getResult().getType(),
                                        inBounds, /*addThenRegion=*/true,
                                        /*addElseRegion=*/true);

    // Then region: Unsafe access
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPointToStart(&thenBlock);
      auto unsafeByte = reussir::ReussirStrUnsafeByteAtOp::create(
          rewriter, loc, rewriter.getI8Type(), op.getStr(), op.getIndex());
      mlir::scf::YieldOp::create(rewriter, loc, unsafeByte.getResult());
    }

    // Else region: Return 0
    {
      auto &elseBlock = ifOp.getElseRegion().front();
      rewriter.setInsertionPointToStart(&elseBlock);
      auto zero = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 8);
      mlir::scf::YieldOp::create(rewriter, loc, zero->getResult(0));
    }

    rewriter.replaceOp(op, ifOp.getResult(0));
    return mlir::success();
  }
};
struct ReussirStrSelectOpRewritePattern
    : public mlir::OpConversionPattern<ReussirStrSelectOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirStrSelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto funcName = emitDecisionFunction(module, rewriter, op.getPatterns());
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    auto call = mlir::func::CallOp::create(rewriter, op.getLoc(), func,
                                           mlir::ValueRange{op.getStr()});

    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};
struct ReussirStrStartWithOpRewritePattern
    : public mlir::OpConversionPattern<ReussirStrStartWithOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirStrStartWithOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto indexType = rewriter.getIndexType();

    // Get string length
    auto lenOp =
        reussir::ReussirStrLenOp::create(rewriter, loc, indexType, op.getStr());

    // Get prefix length
    size_t prefixLen = op.getPrefix().size();
    auto prefixLenVal =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, prefixLen);

    // Check if len >= prefixLen
    auto isSufficientLen = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::uge, lenOp.getResult(),
        prefixLenVal);

    auto resultType = op.getResult().getType();
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, resultType,
                                        isSufficientLen, /*addThenRegion=*/true,
                                        /*addElseRegion=*/true);

    // Then region: Unsafe check
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPointToStart(&thenBlock);
      auto unsafeCheck = reussir::ReussirStrUnsafeStartWithOp::create(
          rewriter, loc, resultType, op.getStr(), op.getPrefixAttr());
      mlir::scf::YieldOp::create(rewriter, loc, unsafeCheck.getResult());
    }

    // Else region: Return false
    {
      auto &elseBlock = ifOp.getElseRegion().front();
      rewriter.setInsertionPointToStart(&elseBlock);
      auto falseVal = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 1);
      mlir::scf::YieldOp::create(rewriter, loc, falseVal->getResult(0));
    }

    rewriter.replaceOp(op, ifOp.getResult(0));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Cell operation lowering patterns
//===----------------------------------------------------------------------===//

struct CellAccess {
  CellType cellType;
  mlir::Value cellRef;
  reussir::AtomicKind atomicKind;
};

static CellType getCellType(mlir::Value cell) {
  return llvm::cast<CellType>(
      llvm::cast<RcType>(cell.getType()).getElementType());
}

static bool isAtomicCell(mlir::Value cell) {
  return getCellType(cell).getAtomic();
}

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

static RefType getCellSlotRefType(const CellAccess &access) {
  return RefType::get(access.cellType.getContext(),
                      access.cellType.getElementType(), Capability::field,
                      access.atomicKind);
}

static mlir::Value projectCellSlot(const CellAccess &access, mlir::Location loc,
                                   mlir::PatternRewriter &rewriter) {
  return ReussirRefProjectOp::create(rewriter, loc, getCellSlotRefType(access),
                                     access.cellRef, rewriter.getIndexAttr(0));
}

static mlir::Value getMutexView(const CellAccess &access, mlir::Location loc,
                                mlir::PatternRewriter &rewriter) {
  assert(access.cellType.getKind() == CellKind::mutex &&
         "only mutex cells have a mutex view");
  auto mutexType = mlir::sync::MutexType::get(rewriter.getContext(),
                                              access.cellType.getElementType());
  auto mutexViewType = mlir::MemRefType::get({}, mutexType);
  return ReussirRefToMemrefOp::create(rewriter, loc, mutexViewType,
                                      access.cellRef);
}

static mlir::Block *
addMutexCriticalSectionBody(mlir::sync::SyncMutexCriticalSectionOp critical,
                            const CellAccess &access, mlir::Location loc,
                            mlir::PatternRewriter &rewriter) {
  auto payloadViewType =
      mlir::MemRefType::get({}, access.cellType.getElementType());
  return rewriter.createBlock(&critical.getBody(), critical.getBody().begin(),
                              {payloadViewType}, {loc});
}

static mlir::Value getMutexPayloadRef(mlir::Block *body,
                                      const CellAccess &access,
                                      mlir::Location loc,
                                      mlir::PatternRewriter &rewriter) {
  return ReussirRefFromMemrefOp::create(
      rewriter, loc, getCellSlotRefType(access), body->getArgument(0));
}

static mlir::Value acquireThenLoadCellSlot(mlir::Value slotRef,
                                           mlir::Location loc,
                                           mlir::PatternRewriter &rewriter) {
  auto slotRefType = llvm::cast<RefType>(slotRef.getType());
  // Retain a bare RC through the loaded SSA value. Its count update may alias
  // the slot, so LLVM cannot reliably eliminate a second load across it.
  if (llvm::isa<RcType>(slotRefType.getElementType())) {
    mlir::Value value = ReussirRefLoadOp::create(
        rewriter, loc, slotRefType.getElementType(), slotRef);
    ReussirRcIncOp::create(rewriter, loc, value);
    return value;
  }
  ReussirRefAcquireOp::create(rewriter, loc, slotRef);
  return ReussirRefLoadOp::create(rewriter, loc, slotRefType.getElementType(),
                                  slotRef);
}

static void dropThenStoreCellSlot(mlir::Value slotRef, mlir::Value replacement,
                                  mlir::Location loc,
                                  mlir::PatternRewriter &rewriter) {
  ReussirRefDropOp::create(rewriter, loc, slotRef);
  ReussirRefStoreOp::create(rewriter, loc, slotRef, replacement);
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

class ReussirCellCreateOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellCreateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellCreateOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    CellType cellType = getCellType(op.getCell());
    mlir::Value poison =
        mlir::ub::PoisonOp::create(rewriter, op.getLoc(), cellType);
    auto created = ReussirRcCreateOp::create(
        rewriter, op.getLoc(), op.getCell().getType(), poison, op.getToken(),
        mlir::Value{}, mlir::FlatSymbolRefAttr{}, mlir::UnitAttr{});
    CellAccess access = borrowCell(created.getRcPtr(), op.getLoc(), rewriter);
    if (cellType.getKind() == CellKind::mutex) {
      mlir::Value mutexView = getMutexView(access, op.getLoc(), rewriter);
      mlir::sync::SyncMutexInitOp::create(rewriter, op.getLoc(), mutexView,
                                          op.getValue());
      rewriter.replaceOp(op, created.getRcPtr());
      return mlir::success();
    }
    mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
    ReussirRefStoreOp::create(rewriter, op.getLoc(), slotRef, op.getValue());
    // A fresh exclusive cell starts out not in use.
    if (cellType.getExclusive())
      storeCellFlag(access, /*inUse=*/false, op.getLoc(), rewriter);
    rewriter.replaceOp(op, created.getRcPtr());
    return mlir::success();
  }
};

class ReussirCellGetOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellGetOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isAtomicCell(op.getCell()))
      return mlir::failure();
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    if (access.cellType.getKind() == CellKind::mutex) {
      mlir::Value mutexView = getMutexView(access, op.getLoc(), rewriter);
      auto critical = mlir::sync::SyncMutexCriticalSectionOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), mutexView);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block *body = addMutexCriticalSectionBody(critical, access,
                                                        op.getLoc(), rewriter);
        rewriter.setInsertionPointToStart(body);
        mlir::Value slotRef =
            getMutexPayloadRef(body, access, op.getLoc(), rewriter);
        mlir::Value value =
            acquireThenLoadCellSlot(slotRef, op.getLoc(), rewriter);
        mlir::sync::SyncYieldOp::create(rewriter, op.getLoc(), value);
      }
      rewriter.replaceOp(op, critical.getResults());
      return mlir::success();
    }
    auto emitLoadAndAcquire = [&]() -> mlir::Value {
      mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
      return acquireThenLoadCellSlot(slotRef, op.getLoc(), rewriter);
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

class ReussirCellSetOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellSetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellSetOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isAtomicCell(op.getCell()))
      return mlir::failure();
    CellAccess access = borrowCell(op.getCell(), op.getLoc(), rewriter);
    if (access.cellType.getKind() == CellKind::mutex) {
      mlir::Value mutexView = getMutexView(access, op.getLoc(), rewriter);
      auto critical = mlir::sync::SyncMutexCriticalSectionOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), mutexView);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block *body = addMutexCriticalSectionBody(critical, access,
                                                        op.getLoc(), rewriter);
        rewriter.setInsertionPointToStart(body);
        mlir::Value slotRef =
            getMutexPayloadRef(body, access, op.getLoc(), rewriter);
        dropThenStoreCellSlot(slotRef, op.getValue(), op.getLoc(), rewriter);
        mlir::sync::SyncYieldOp::create(rewriter, op.getLoc());
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    auto emitDropAndStore = [&]() {
      mlir::Value slotRef = projectCellSlot(access, op.getLoc(), rewriter);
      dropThenStoreCellSlot(slotRef, op.getValue(), op.getLoc(), rewriter);
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

class ReussirCellRmwOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellRmwOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellRmwOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isAtomicCell(op.getCell()))
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

// LLVM's atomicrmw has no multiply flavor; direct multiply takes the same
// retry loop as the region form. Every other direct kind is a straight-line
// llvm.atomicrmw and stays high-level until the LLVM lowering.
static bool hasNativeAtomicRMWLowering(mlir::arith::AtomicRMWKind kind) {
  return kind != mlir::arith::AtomicRMWKind::muli &&
         kind != mlir::arith::AtomicRMWKind::mulf;
}

// An atomic read-modify-write that LLVM cannot express as a single atomicrmw
// (the region form and the multiply kinds) is loop-shaped, so its control
// flow is crafted here as structured IR: an initial monotonic observation
// feeds an `scf.while` whose before-region computes the replacement and
// commits it with the straight-line `reussir.ref.cmpxchg` bridge; a failed
// exchange retries with the element the comparison observed. Only the
// successful attempt's values leave the loop.
class ReussirAtomicCellRmwOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellRmwOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellRmwOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isAtomicCell(op.getCell()))
      return mlir::failure();
    if (auto kind = op.getKind(); kind && hasNativeAtomicRMWLowering(*kind))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    CellAccess access = borrowCell(op.getCell(), loc, rewriter);
    mlir::Value slotRef = projectCellSlot(access, loc, rewriter);
    mlir::Type elementType = access.cellType.getElementType();
    auto orderingAttr = mlir::LLVM::AtomicOrderingAttr::get(
        rewriter.getContext(),
        op.getOrdering().value_or(mlir::LLVM::AtomicOrdering::acq_rel));
    auto monotonicAttr = mlir::LLVM::AtomicOrderingAttr::get(
        rewriter.getContext(), mlir::LLVM::AtomicOrdering::monotonic);
    mlir::Value initial = ReussirCellGetOp::create(rewriter, loc, elementType,
                                                   monotonicAttr, op.getCell());

    // The loop always carries the observed element; a region body may thread
    // one extra output out of the successful attempt.
    bool direct = static_cast<bool>(op.getKindAttr());
    llvm::SmallVector<mlir::Type> loopResultTypes{elementType};
    if (!direct && op.getOutput())
      loopResultTypes.push_back(op.getOutput().getType());

    auto whileOp = mlir::scf::WhileOp::create(rewriter, loc, loopResultTypes,
                                              mlir::ValueRange{initial});
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      mlir::Block *before =
          rewriter.createBlock(&whileOp.getBefore(), whileOp.getBefore().end(),
                               {elementType}, {loc});
      mlir::Value expected = before->getArgument(0);
      mlir::Value desired;
      mlir::Value bodyOutput;
      mlir::Location commitLoc = loc;
      ReussirCellYieldOp yield;
      if (direct) {
        if (*op.getKind() == mlir::arith::AtomicRMWKind::mulf)
          desired = mlir::arith::MulFOp::create(rewriter, loc, expected,
                                                op.getValue());
        else
          desired = mlir::arith::MulIOp::create(rewriter, loc, expected,
                                                op.getValue());
      } else {
        mlir::Block &body = op.getBody().front();
        yield = llvm::cast<ReussirCellYieldOp>(body.getTerminator());
        rewriter.inlineBlockBefore(&body, before, before->end(),
                                   mlir::ValueRange{expected});
        rewriter.setInsertionPoint(yield);
        commitLoc = yield.getLoc();
        desired = yield.getNewValue();
        bodyOutput = yield.getOutput();
      }
      // Weak exchange: the surrounding loop retries on failure anyway, and
      // weakness lets LLVM pick LL/SC sequences where beneficial.
      auto cmpxchg = ReussirRefCmpXchgOp::create(
          rewriter, commitLoc, elementType, rewriter.getI1Type(), orderingAttr,
          rewriter.getUnitAttr(), expected, desired, slotRef);
      mlir::Value trueValue = mlir::arith::ConstantOp::create(
          rewriter, commitLoc,
          rewriter.getIntegerAttr(rewriter.getI1Type(), true));
      mlir::Value retry = mlir::arith::XOrIOp::create(
          rewriter, commitLoc, cmpxchg.getSuccess(), trueValue);
      llvm::SmallVector<mlir::Value> forwarded{cmpxchg.getObserved()};
      if (bodyOutput)
        forwarded.push_back(bodyOutput);
      mlir::scf::ConditionOp::create(rewriter, commitLoc, retry, forwarded);
      if (yield)
        rewriter.eraseOp(yield);

      mlir::Block *after = rewriter.createBlock(
          &whileOp.getAfter(), whileOp.getAfter().end(), loopResultTypes,
          llvm::SmallVector<mlir::Location>(loopResultTypes.size(), loc));
      mlir::scf::YieldOp::create(rewriter, loc,
                                 mlir::ValueRange{after->getArgument(0)});
    }

    if (op->getNumResults() == 0)
      rewriter.eraseOp(op);
    else if (direct)
      // The direct form returns the old element: the successful comparison
      // observed exactly the value it replaced.
      rewriter.replaceOp(op, whileOp.getResult(0));
    else
      rewriter.replaceOp(op, whileOp.getResult(1));
    return mlir::success();
  }
};

class ReussirCellInUseOpRewritePattern
    : public mlir::OpConversionPattern<ReussirCellInUseOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCellInUseOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
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
// ConvertToSTDPass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertToSTDPass
    : public impl::ReussirConvertToSTDPassBase<ConvertToSTDPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateConvertToSTDConversionPatterns(patterns);

    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::bufferization::BufferizationDialect,
                           mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                           mlir::math::MathDialect, mlir::memref::MemRefDialect,
                           mlir::ptr::PtrDialect, mlir::scf::SCFDialect,
                           mlir::sync::SyncDialect, mlir::tensor::TensorDialect,
                           mlir::ub::UBDialect, reussir::ReussirDialect>();
    target.addDynamicallyLegalOp<ReussirArrayViewOp>([](ReussirArrayViewOp op) {
      return llvm::isa<mlir::MemRefType>(op.getView().getType());
    });
    // `token.realloc` is a real resize; it always lowers to the direct
    // `__reussir_reallocate` call (BasicOpsLowering), sound on any allocator.
    // A free of a still-nullable token must be null-guarded here; a free of a
    // plain token is legal and lowers to the runtime call directly.
    target.addDynamicallyLegalOp<ReussirTokenFreeOp>([](ReussirTokenFreeOp op) {
      return llvm::isa<TokenType>(op.getToken().getType());
    });
    // A fused eval (non-empty `with` pack) must be expanded here — the
    // basic-ops lowering only dispatches the plain, fully-applied form.
    target.addDynamicallyLegalOp<ReussirClosureEvalOp>(
        [](ReussirClosureEvalOp op) { return op.getArgs().empty(); });
    // Straight-line atomic cell accesses retain their high-level operation
    // until the LLVM conversion can preserve their ordering. Cell creation,
    // all non-atomic Cell operations, and loop-shaped atomic RMWs (region
    // bodies and multiply, which lower to retry loops) are structurally
    // lowered here, before inc/dec cancellation.
    target.addDynamicallyLegalOp<ReussirCellGetOp>(
        [](ReussirCellGetOp op) { return isAtomicCell(op.getCell()); });
    target.addDynamicallyLegalOp<ReussirCellSetOp>(
        [](ReussirCellSetOp op) { return isAtomicCell(op.getCell()); });
    target.addDynamicallyLegalOp<ReussirCellRmwOp>([](ReussirCellRmwOp op) {
      if (!isAtomicCell(op.getCell()))
        return false;
      auto kind = op.getKind();
      return kind && hasNativeAtomicRMWLowering(*kind);
    });

    target.addIllegalOp<
        ReussirNullableDispatchOp, ReussirRecordDispatchOp, ReussirScfYieldOp,
        ReussirClosureUniqifyOp, ReussirArrayWithUniqueViewOp,
        ReussirTokenEnsureOp, ReussirStrByteAtOp, ReussirStrSelectOp,
        ReussirStrStartWithOp, ReussirCellCreateOp, ReussirCellInUseOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Reussir conversion may introduce high-level synchronization operations.
    // Expand their structured control flow here as part of the same standard
    // conversion stage. The remaining straight-line sync bridge operations
    // are intentionally kept for the final LLVM conversion.
    // Seed the greedy driver only with sync operations: it will still visit
    // the operations created by their multi-step expansion, without folding
    // unrelated standard IR elsewhere in the module.
    llvm::SmallVector<mlir::Operation *> syncOps;
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          mlir::sync::SyncDialect::getDialectNamespace())
        syncOps.push_back(op);
    });
    if (syncOps.empty())
      return;

    mlir::RewritePatternSet syncPatterns(&getContext());
    mlir::sync::populateConvertSyncToSTDPatterns(syncPatterns);
    mlir::GreedyRewriteConfig config;
    config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
    if (failed(mlir::applyOpPatternsGreedily(
            syncOps, mlir::FrozenRewritePatternSet(std::move(syncPatterns)),
            config)))
      signalPassFailure();
  }
};
} // namespace

void populateConvertToSTDConversionPatterns(mlir::RewritePatternSet &patterns) {
  // Add patterns for high-level Reussir operations.
  patterns.add<
      ReussirNullableDispatchOpRewritePattern, ReussirArrayViewOpRewritePattern,
      ReussirRecordDispatchOpRewritePattern,
      ReussirClosureUniqifyOpRewritePattern, ReussirClosureEvalOpRewritePattern,
      ReussirArrayWithUniqueViewOpRewritePattern,
      ReussirScfYieldOpRewritePattern, ReussirTokenEnsureOpRewritePattern,
      ReussirNullableTokenFreeOpRewritePattern,
      ReussirStrByteAtOpRewritePattern, ReussirStrSelectOpRewritePattern,
      ReussirStrStartWithOpRewritePattern, ReussirCellCreateOpRewritePattern,
      ReussirCellGetOpRewritePattern, ReussirCellSetOpRewritePattern,
      ReussirCellRmwOpRewritePattern, ReussirAtomicCellRmwOpRewritePattern,
      ReussirCellInUseOpRewritePattern>(patterns.getContext());
}

} // namespace reussir
