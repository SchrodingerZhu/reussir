//===-- SpecialPointerTag.cpp - Nullary variants as immediates --*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 With LLVM Exceptions OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/SpecialPointerTag.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace reussir {
#define GEN_PASS_DEF_REUSSIRSPECIALPOINTERTAGPASS
#include "Reussir/Transformation/Passes.h.inc"

// Rewrites every *nullary* variant construction of a taggable shared rc-boxed
// enum into a `reussir.rc.tagged` immediate — no token, no allocation, no
// observable reference count — and stamps the module with
// `kSpecialPtrTagAttr` so the LLVM lowering steers the few stores that must
// not reach an immediate's dummy box (see BasicOpsLowering's
// special-pointer-tag notes; the hot reads need no guard at all).
//
// The driver adds this pass only when the scheme is enabled (aarch64
// targets by default). The encoding hard-depends on TBI: loads through a
// tagged value are expected to reach the per-tag dummy box, i.e. the
// hardware must ignore the pointer's top byte on data accesses. It must run
// BEFORE token instantiation, so no allocation token is ever attached to a
// rewritten construction.

namespace {
// The frontend boxes a variant as `record.compound {} → record.variant →
// rc.create`; the fused `rc.create_variant` only appears much later
// (RcCreateFusion), after tokens are already attached — so the nullary
// rewrite matches the unfused chain here, before token instantiation. The
// dead `record.variant`/`record.compound` producers are cleaned up by the
// greedy driver.
struct NullaryVariantPattern
    : public mlir::OpRewritePattern<ReussirRcCreateOp> {
  using mlir::OpRewritePattern<ReussirRcCreateOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Region-allocated boxes have a different header and lifecycle; a token
    // already attached means instantiation ran — too late to rewrite.
    if (op.getRegion() || op.getToken() || op.getSkipRc())
      return mlir::failure();
    RcType rcType = op.getRcPtr().getType();
    if (!rcType.mayCarrySpecialPointerTag())
      return mlir::failure();
    auto variant = op.getValue().getDefiningOp<ReussirRecordVariantOp>();
    if (!variant)
      return mlir::failure();
    // The payload must be a *nullary* arm: an empty compound record.
    auto payloadTy = llvm::dyn_cast<RecordType>(variant.getValue().getType());
    if (!payloadTy || !payloadTy.isCompound() ||
        !payloadTy.getMembers().empty())
      return mlir::failure();
    size_t tag = variant.getTag().getZExtValue();
    if (tag + 1 > 0xFF)
      return mlir::failure();
    auto variantType = llvm::dyn_cast<RecordType>(rcType.getElementType());
    if (tag >= variantType.getMembers().size())
      return mlir::failure();
    auto arm = llvm::dyn_cast<RecordType>(variantType.getMembers()[tag]);
    if (!arm || !arm.isCompound() || !arm.getMembers().empty())
      return mlir::failure();
    rewriter.replaceOpWithNewOp<ReussirRcTaggedOp>(op, rcType,
                                                   variant.getTagAttr());
    return mlir::success();
  }
};

struct SpecialPointerTagPass
    : public impl::ReussirSpecialPointerTagPassBase<SpecialPointerTagPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module->setAttr(kSpecialPtrTagAttr,
                    mlir::UnitAttr::get(module.getContext()));
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<NullaryVariantPattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace reussir
