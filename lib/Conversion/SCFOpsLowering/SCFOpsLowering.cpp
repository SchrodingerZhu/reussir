//===-- SCFOpsLowering.cpp - Reussir SCF ops lowering impl -----*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/SCFOpsLowering.h"
#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
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
#include <utility>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRSCFOPSLOWERINGPASS
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
      condition = mlir::arith::CmpIOp::create(builder, 
          loc, mlir::arith::CmpIPredicate::eq, len.getResult(),
          zero.getResult());
    } else {
      auto startswith = ReussirStrUnsafeStartWithOp::create(builder, 
          loc, i1Type, currentSlice, builder.getStringAttr(p.pattern));
      auto len = ReussirStrLenOp::create(builder, loc, builder.getIndexType(),
                                                 currentSlice);
      auto expectedLen =
          mlir::arith::ConstantIndexOp::create(builder, loc, p.pattern.size());
      auto lenOk = mlir::arith::CmpIOp::create(builder, 
          loc, mlir::arith::CmpIPredicate::eq, len.getResult(),
          expectedLen.getResult());
      condition = mlir::arith::AndIOp::create(builder, 
          loc, startswith.getResult(), lenOk.getResult());
    }

    auto ifOp = mlir::scf::IfOp::create(builder, 
        loc, mlir::TypeRange{indexType, i1Type}, condition,
        /*addThenRegion=*/true, /*addElseRegion=*/true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto idx =
          mlir::arith::ConstantIndexOp::create(builder, loc, p.originalIdx);
      auto trueVal = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{idx.getResult(), trueVal.getResult()});
    }

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
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
    auto lenOk = mlir::arith::CmpIOp::create(builder, 
        loc, mlir::arith::CmpIPredicate::uge, len.getResult(),
        minLen.getResult());

    auto ifLen = mlir::scf::IfOp::create(builder, 
        loc, mlir::TypeRange{indexType, i1Type}, lenOk.getResult(),
        /*addThenRegion=*/true, /*addElseRegion=*/true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifLen.getThenRegion().front());

      auto startswith = ReussirStrUnsafeStartWithOp::create(builder, 
          loc, i1Type, currentSlice, builder.getStringAttr(lcp));

      auto ifMatch = mlir::scf::IfOp::create(builder, 
          loc, mlir::TypeRange{indexType, i1Type}, startswith.getResult(),
          /*addThenRegion=*/true, /*addElseRegion=*/true);

      {
        mlir::OpBuilder::InsertionGuard thenGuard(builder);
        builder.setInsertionPointToStart(&ifMatch.getThenRegion().front());
        auto offset = mlir::arith::ConstantIndexOp::create(builder, loc, lcpLen);
        auto nextSlice = ReussirStrSliceOp::create(builder, 
            loc, currentSlice.getType(), currentSlice, offset.getResult());

        llvm::SmallVector<PatternInfo> nextPatterns;
        for (const auto &p : patterns) {
          nextPatterns.push_back({p.originalIdx, p.pattern.substr(lcpLen)});
        }
        auto res = buildDecisionTree(loc, builder, indexType, i1Type,
                                     nextSlice.getResult(), nextPatterns);
        mlir::scf::YieldOp::create(builder, 
            loc, mlir::ValueRange{res.first, res.second});
      }

      {
        mlir::OpBuilder::InsertionGuard elseGuard(builder);
        builder.setInsertionPointToStart(&ifMatch.getElseRegion().front());
        auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
        auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
        mlir::scf::YieldOp::create(builder, 
            loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
      }
      mlir::scf::YieldOp::create(builder, loc, ifMatch.getResults());
    }

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifLen.getElseRegion().front());
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
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
  auto isZero = mlir::arith::CmpIOp::create(builder, 
      loc, mlir::arith::CmpIPredicate::eq, len.getResult(), zero.getResult());

  auto ifZero = mlir::scf::IfOp::create(builder, 
      loc, mlir::TypeRange{indexType, i1Type}, isZero.getResult(),
      /*addThenRegion=*/true, /*addElseRegion=*/true);

  // Then: len == 0
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifZero.getThenRegion().front());
    if (!emptyPatterns.empty()) {
      auto idx = mlir::arith::ConstantIndexOp::create(builder, 
          loc, emptyPatterns[0].originalIdx);
      auto trueVal = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{idx.getResult(), trueVal.getResult()});
    } else {
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    }
  }

  // Else: len > 0, dispatch on byte
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifZero.getElseRegion().front());

    if (nonEmptyPatterns.empty()) {
      auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
      auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
      mlir::scf::YieldOp::create(builder, 
          loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
    } else {
      auto byteAtZero = ReussirStrUnsafeByteAtOp::create(builder, 
          loc, builder.getI8Type(), currentSlice, zero.getResult());
      auto byteIndex = mlir::arith::IndexCastOp::create(builder, 
          loc, builder.getIndexType(), byteAtZero.getResult());

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
      auto nextSlice = ReussirStrSliceOp::create(builder, 
          loc, currentSlice.getType(), currentSlice, one.getResult());

      auto switchOp = mlir::scf::IndexSwitchOp::create(builder, 
          loc, mlir::TypeRange{indexType, i1Type}, byteIndex.getResult(), cases,
          cases.size());

      for (auto [idx, b] : llvm::enumerate(cases)) {
        auto &region = switchOp.getCaseRegions()[idx];
        region.emplaceBlock();
        mlir::OpBuilder::InsertionGuard caseGuard(builder);
        builder.setInsertionPointToStart(&region.front());
        auto res = buildDecisionTree(loc, builder, indexType, i1Type,
                                     nextSlice.getResult(), groups[b]);
        mlir::scf::YieldOp::create(builder, 
            loc, mlir::ValueRange{res.first, res.second});
      }

      // Default region
      {
        auto &region = switchOp.getDefaultRegion();
        region.emplaceBlock();
        mlir::OpBuilder::InsertionGuard defaultGuard(builder);
        builder.setInsertionPointToStart(&region.front());
        auto poison = mlir::ub::PoisonOp::create(builder, loc, indexType);
        auto falseVal = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
        mlir::scf::YieldOp::create(builder, 
            loc, mlir::ValueRange{poison.getResult(), falseVal.getResult()});
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
    auto thenCreate =
        ifOp.thenYield().getOperand(idx).getDefiningOp<ReussirNullableCreateOp>();
    auto elseCreate =
        ifOp.elseYield().getOperand(idx).getDefiningOp<ReussirNullableCreateOp>();
    return thenCreate && elseCreate && thenCreate.getPtr() &&
           !elseCreate.getPtr();
  }

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableDispatchOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // First, create a check operation to get the null flag from the input.
    mlir::Value flag = reussir::ReussirNullableCheckOp::create(rewriter,
        op.getLoc(), op.getNullable());
    // mark expect not null
    if (op->hasAttr(REUSSIR_EXPANDED_ENSURE_ATTR))
      flag = reussir::ReussirExpectOp::create(rewriter, op.getLoc(), flag, true);

    auto scfIfOp = mlir::scf::IfOp::create(rewriter,
        op.getLoc(), op->getResultTypes(), flag, /*addThenRegion=*/true,
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
    auto coerced = reussir::ReussirNullableCoerceOp::create(rewriter, 
        op.getLoc(), op.getNullable().getType().getPtrTy(), op.getNullable());
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
    auto indexSwitchOp = mlir::scf::IndexSwitchOp::create(rewriter, 
        op.getLoc(), rewriter.getIndexType(), tag.getResult(), allTags,
        allTags.size());

    for (auto [tag, region] :
         llvm::zip(allTags, indexSwitchOp.getCaseRegions())) {
      mlir::Block *block = rewriter.createBlock(&region, region.begin());
      rewriter.setInsertionPointToStart(block);
      auto constantIdx = mlir::arith::ConstantIndexOp::create(rewriter, 
          op.getLoc(), tagToRegionIdx[tag]);
      mlir::scf::YieldOp::create(rewriter, op.getLoc(),
                                          constantIdx->getResults());
    }
    {
      mlir::Block *block =
          rewriter.createBlock(&indexSwitchOp.getDefaultRegion(),
                               indexSwitchOp.getDefaultRegion().begin());
      rewriter.setInsertionPointToStart(block);
      auto poison = mlir::ub::PoisonOp::create(rewriter, 
          op.getLoc(), rewriter.getIndexType());
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), poison->getResults());
    }
    return indexSwitchOp.getResult(0);
  }

public:
  mlir::LogicalResult
  matchAndRewrite(ReussirRecordDispatchOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // First, create a RecordTagOps operation to get the tag from the input.
    auto tag = reussir::ReussirRecordTagOp::create(rewriter, op.getLoc(),
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

    auto indexSwitchOp = mlir::scf::IndexSwitchOp::create(rewriter, 
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
        auto poison = mlir::ub::PoisonOp::create(rewriter, 
            op.getLoc(), op.getValue().getType());
        poisonValues.push_back(poison);
      }
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), poisonValues);
    }
    for (auto [idx, tagSet, region] :
         llvm::enumerate(op.getTagSets(), indexSwitchOp.getCaseRegions())) {
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
        mlir::Type targetVariantType =
            getProjectedType(recordType.getMembers()[tagArray[0]],
                             recordType.getMemberIsField()[tagArray[0]],
                             variantRef.getCapability());
        RefType coercedType =
            RefType::get(rewriter.getContext(), targetVariantType,
                         variantRef.getCapability());
        auto coerced = reussir::ReussirRecordCoerceOp::create(rewriter, 
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

struct ReussirClosureUniqifyOpRewritePattern
    : public mlir::OpConversionPattern<ReussirClosureUniqifyOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirClosureUniqifyOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Create a check operation to see if the closure is unique
    auto isUnique = reussir::ReussirRcIsUniqueOp::create(rewriter, 
        op.getLoc(), op.getClosure());

    // Create an SCF if-else operation
    auto scfIfOp = mlir::scf::IfOp::create(rewriter, 
        op.getLoc(), op->getResultTypes(), isUnique, /*addThenRegion=*/true,
        /*addElseRegion=*/true);

    // In the then region (closure is unique), just return the original
    // closure
    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    mlir::scf::YieldOp::create(rewriter, op.getLoc(), op.getClosure());

    // In the else region (closure is not unique), clone the closure, dec the
    // original rc pointer
    rewriter.setInsertionPointToStart(&scfIfOp.getElseRegion().front());
    auto cloned = reussir::ReussirClosureCloneOp::create(rewriter, 
        op.getLoc(), op.getClosure().getType(), op.getClosure());
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
      auto appliedType = RcType::get(
          rewriter.getContext(),
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
  return mlir::MemRefType::get(arrayType.getShape(), arrayType.getElementType());
}

static mlir::Value materializeArrayViewValue(mlir::Location loc,
                                             mlir::PatternRewriter &rewriter,
                                             ArrayType arrayType,
                                             mlir::Value ref,
                                             mlir::Type viewType,
                                             bool writable) {
  auto memrefType = getArrayViewMemRefType(arrayType);
  auto memrefView =
      ReussirArrayViewOp::create(rewriter, loc, memrefType, ref).getView();
  if (llvm::isa<mlir::MemRefType>(viewType))
    return memrefView;

  auto tensorType = llvm::cast<mlir::RankedTensorType>(viewType);
  return mlir::bufferization::ToTensorOp::create(rewriter, loc, tensorType, memrefView,
                                               /*restrict=*/true, writable)
      .getResult();
}

struct ReussirArrayViewOpRewritePattern
    : public mlir::OpConversionPattern<ReussirArrayViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirArrayViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getView().getType());
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "expected tensor array view");

    auto arrayType =
        llvm::cast<ArrayType>(llvm::cast<RefType>(op.getRef().getType()).getElementType());
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
  matchAndRewrite(ReussirArrayWithUniqueViewOp op,
                  OpAdaptor adaptor,
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

    auto makeClonedArray =
        [&]() -> std::pair<mlir::Value, mlir::Value> {
      auto borrowedType = RefType::get(rewriter.getContext(), arrayType);
      auto srcRef =
          ReussirRcBorrowOp::create(rewriter, loc, borrowedType, op.getArray());
      RcBoxType rcBoxType = RcBoxType::get(rewriter.getContext(), arrayType,
                                           /*regional=*/false);
      auto dataLayout = mlir::DataLayout::closest(op.getOperation());
      TokenType tokenType =
          TokenType::get(rewriter.getContext(),
                         dataLayout.getTypeABIAlignment(rcBoxType),
                         dataLayout.getTypeSize(rcBoxType).getFixedValue());
      auto token = ReussirTokenAllocOp::create(rewriter, loc, tokenType);
      auto poison = mlir::ub::PoisonOp::create(rewriter, loc, arrayType);
      auto cloned = ReussirRcCreateOp::create(rewriter, 
          loc, rcType, poison.getResult(), token.getResult(), mlir::Value{},
          mlir::FlatSymbolRefAttr{}, mlir::UnitAttr{});
      auto dstRef =
          ReussirRcBorrowOp::create(rewriter, loc, borrowedType, cloned.getResult());
      ReussirRefMemcpyOp::create(rewriter, loc, srcRef.getResult(),
                                          dstRef.getResult());
      ReussirRefAcquireOp::create(rewriter, loc, dstRef.getResult(), false,
                                           nullptr);

      auto refCount = ReussirRcFetchOp::create(rewriter, loc, op.getArray());
      auto decremented = mlir::arith::SubIOp::create(rewriter, 
          loc, refCount.getRefCount(),
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
      ReussirRcSetOp::create(rewriter, loc, op.getArray(), decremented.getResult());
      return {cloned.getResult(), dstRef.getResult()};
    };

    auto isUnique =
        reussir::ReussirRcIsUniqueOp::create(rewriter, loc, op.getArray());
    auto scfIfOp = mlir::scf::IfOp::create(rewriter, 
        loc, op->getResultTypes(), isUnique, /*addThenRegion=*/true,
        /*addElseRegion=*/true);

    rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
    cloneArrayWithUniqueViewBody(op, op.getArray(),
                                 makeBorrowedView(op.getArray()), rewriter);

    rewriter.setInsertionPointToStart(&scfIfOp.getElseRegion().front());
    auto [clonedArray, clonedRef] = makeClonedArray();
    auto clonedView = materializeArrayViewValue(loc, rewriter, arrayType,
                                                clonedRef, viewType,
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
  matchAndRewrite(ReussirScfYieldOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op->getOperands());
    return mlir::success();
  }
};

struct ReussirTokenEnsureOpRewritePattern
    : public mlir::OpConversionPattern<ReussirTokenEnsureOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirTokenEnsureOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto nullableDispatchOp = ReussirNullableDispatchOp::create(rewriter, 
        op.getLoc(), op.getType(), op.getNullableToken());

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
            tokenSrc = reussir::ReussirRcReinterpretOp::create(rewriter, 
                op.getLoc(), op.getType(), reinterpretOp.getRcPtr());
        }
      auto launderedToken = ReussirTokenLaunderOp::create(rewriter, 
          op.getLoc(), op.getType(), tokenSrc);
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
  matchAndRewrite(ReussirStrByteAtOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Get string length
    auto lenOp = reussir::ReussirStrLenOp::create(rewriter, 
        loc, rewriter.getIndexType(), op.getStr());

    // Check if index is within bounds (index < len)
    auto inBounds = mlir::arith::CmpIOp::create(rewriter, 
        loc, mlir::arith::CmpIPredicate::ult, op.getIndex(), lenOp.getResult());

    // Create if-else block
    auto ifOp = mlir::scf::IfOp::create(rewriter, 
        loc, op.getResult().getType(), inBounds, /*addThenRegion=*/true,
        /*addElseRegion=*/true);

    // Then region: Unsafe access
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPointToStart(&thenBlock);
      auto unsafeByte = reussir::ReussirStrUnsafeByteAtOp::create(rewriter, 
          loc, rewriter.getI8Type(), op.getStr(), op.getIndex());
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
  matchAndRewrite(ReussirStrSelectOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto funcName = emitDecisionFunction(module, rewriter, op.getPatterns());
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    auto call = mlir::func::CallOp::create(rewriter, 
        op.getLoc(), func, mlir::ValueRange{op.getStr()});

    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};
struct ReussirStrStartWithOpRewritePattern
    : public mlir::OpConversionPattern<ReussirStrStartWithOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ReussirStrStartWithOp op,
                  OpAdaptor adaptor,
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
    auto isSufficientLen = mlir::arith::CmpIOp::create(rewriter, 
        loc, mlir::arith::CmpIPredicate::uge, lenOp.getResult(), prefixLenVal);

    auto resultType = op.getResult().getType();
    auto ifOp = mlir::scf::IfOp::create(rewriter, 
        loc, resultType, isSufficientLen, /*addThenRegion=*/true,
        /*addElseRegion=*/true);

    // Then region: Unsafe check
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPointToStart(&thenBlock);
      auto unsafeCheck = reussir::ReussirStrUnsafeStartWithOp::create(rewriter, 
          loc, resultType, op.getStr(), op.getPrefixAttr());
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

    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::bufferization::BufferizationDialect,
                           mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                           mlir::math::MathDialect, mlir::memref::MemRefDialect,
                           mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                           mlir::ub::UBDialect, reussir::ReussirDialect>();
    target.addDynamicallyLegalOp<ReussirArrayViewOp>(
        [](ReussirArrayViewOp op) {
          return llvm::isa<mlir::MemRefType>(op.getView().getType());
        });
    // `token.realloc` is a real resize; it always lowers to the direct
    // `__reussir_reallocate` call (BasicOpsLowering), sound on any allocator.
    // A free of a still-nullable token must be null-guarded here; a free of a
    // plain token is legal and lowers to the runtime call directly.
    target.addDynamicallyLegalOp<ReussirTokenFreeOp>(
        [](ReussirTokenFreeOp op) {
          return llvm::isa<TokenType>(op.getToken().getType());
        });
    // A fused eval (non-empty `with` pack) must be expanded here — the
    // basic-ops lowering only dispatches the plain, fully-applied form.
    target.addDynamicallyLegalOp<ReussirClosureEvalOp>(
        [](ReussirClosureEvalOp op) { return op.getArgs().empty(); });

    target.addIllegalOp<ReussirNullableDispatchOp, ReussirRecordDispatchOp,
                        ReussirScfYieldOp, ReussirClosureUniqifyOp,
                        ReussirArrayWithUniqueViewOp,
                        ReussirTokenEnsureOp, ReussirStrByteAtOp,
                        ReussirStrSelectOp, ReussirStrStartWithOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateSCFOpsLoweringConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  // Add conversion patterns for Reussir SCF operations
  patterns
      .add<ReussirNullableDispatchOpRewritePattern,
           ReussirArrayViewOpRewritePattern,
           ReussirRecordDispatchOpRewritePattern,
           ReussirClosureUniqifyOpRewritePattern,
           ReussirClosureEvalOpRewritePattern,
           ReussirArrayWithUniqueViewOpRewritePattern,
           ReussirScfYieldOpRewritePattern, ReussirTokenEnsureOpRewritePattern,
           ReussirNullableTokenFreeOpRewritePattern,
           ReussirStrByteAtOpRewritePattern, ReussirStrSelectOpRewritePattern,
           ReussirStrStartWithOpRewritePattern>(patterns.getContext());
}

} // namespace reussir
