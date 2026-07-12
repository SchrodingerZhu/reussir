//===-- RcDecrementExpansion.cpp -------------------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
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

#define GEN_PASS_DEF_REUSSIRRCDECREMENTEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct RcDecrementExpansionPattern
    : public mlir::OpRewritePattern<ReussirRcDecOp> {
  using mlir::OpRewritePattern<ReussirRcDecOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcDecOp op,
                  mlir::PatternRewriter &rewriter) const override {
    RcType type = op.getRcPtr().getType();
    // No need to proceed if dec operation is applied to a rigid type.
    // Also delay the FFI object type clean up until basic ops lowering pass.
    if (type.getCapability() == Capability::rigid ||
        mlir::isa<FFIObjectType, ClosureType>(type.getElementType()))
      return mlir::failure();

    auto prevRcCount =
        ReussirRcFetchOp::create(rewriter, op.getLoc(), op.getRcPtr());
    auto isOne = mlir::arith::CmpIOp::create(rewriter, 
        op.getLoc(), mlir::arith::CmpIPredicate::eq, prevRcCount.getRefCount(),
        mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1));
    auto likelyUnique =
        ReussirExpectOp::create(rewriter, op.getLoc(), isOne.getResult(), true);
    auto ifOp =
        mlir::scf::IfOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                         likelyUnique.getLikely(), true, true);
    RefType borrowedRefType = rewriter.getType<RefType>(
        type.getElementType(), Capability::unspecified, type.getAtomicKind());
    TokenType tokenType = llvm::cast<TokenType>(
        llvm::cast<NullableType>(op.getNullableToken().getType()).getPtrTy());
    // A *destructuring* decrement (see `reussir-rc-dispatch-fusion`) knows
    // the pattern arm that consumed the box: bound members transfer with the
    // arm, so the unique path releases only the *unbound* content and the
    // shared path retains the bound members in place of the fused-away
    // per-binding retains. Everything else expands through the transitive
    // drop glue as before.
    {
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      if (op.isDestructuring()) {
        llvm::SmallDenseSet<int64_t> bound;
        for (int64_t index : op.getBoundMembersAttr().asArrayRef())
          bound.insert(index);
        auto [payload, coerced] = op.destructuredPayloadAndRef(rewriter);
        for (auto [idx, memberTy, memberIsField] : llvm::enumerate(
                 payload.getMembers(), payload.getMemberIsField())) {
          if (bound.contains(static_cast<int64_t>(idx)) || memberIsField)
            continue;
          auto projectedTy = getProjectedType(memberTy, memberIsField,
                                              Capability::unspecified);
          if (isTriviallyCopyable(projectedTy))
            continue;
          // Unbound members release through `ref.drop` — the same route the
          // transitive glue took. The acquire/drop expansion later turns an
          // rc slot's drop into a *plain* member decrement, which keeps it
          // visible to the post-expansion cancellation window (`inc %m`
          // against the unique-path release moves the retain to the shared
          // branch); materializing the decrement here would see it expanded
          // immediately and hide it from that optimization.
          auto projectedRefTy = rewriter.getType<RefType>(
              projectedTy, Capability::unspecified, type.getAtomicKind());
          mlir::Value slot = ReussirRefProjectOp::create(
              rewriter, op.getLoc(), projectedRefTy, coerced,
              rewriter.getIndexAttr(idx));
          ReussirRefDropOp::create(rewriter, op.getLoc(), slot);
        }
      } else {
        mlir::Value ref = ReussirRcBorrowOp::create(rewriter, 
            op.getLoc(), borrowedRefType, op.getRcPtr());
        ReussirRefDropOp::create(rewriter, op.getLoc(), ref);
      }
      mlir::Value token = ReussirRcReinterpretOp::create(rewriter, 
          op.getLoc(), tokenType, op.getRcPtr());
      mlir::Value nonnull = ReussirNullableCreateOp::create(rewriter, 
          op.getLoc(), op.getNullableToken().getType(), token);
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), nonnull);
    }
    {
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      auto decremented = mlir::arith::SubIOp::create(rewriter, 
          op.getLoc(), prevRcCount.getRefCount(),
          mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1));
      ReussirRcSetOp::create(rewriter, op.getLoc(), op.getRcPtr(),
                                      decremented.getResult());
      if (op.isDestructuring()) {
        // The shared path keeps the box alive, so the consumer's bound
        // members need their own references — the retains the fusion
        // erased.
        op.rematerializeBoundRetains(rewriter);
      }
      auto null = ReussirNullableCreateOp::create(rewriter, 
          op.getLoc(), op.getNullableToken().getType(), nullptr);
      mlir::scf::YieldOp::create(rewriter, op.getLoc(), null->getResults());
    }
    ifOp->setAttr(kExpandedDecrementAttr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RcDecrementExpansionPass
//===----------------------------------------------------------------------===//

namespace {
struct RcDecrementExpansionPass
    : public impl::ReussirRcDecrementExpansionPassBase<
          RcDecrementExpansionPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    populateRcDecrementExpansionConversionPatterns(patterns);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateRcDecrementExpansionConversionPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<RcDecrementExpansionPattern>(patterns.getContext());
}

} // namespace reussir
