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
/// This file implements Reussir variant tag inference.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/InferVariantTag.h"
#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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

/// A dominance-scoped fact: `ref` is known to point at the variant with
/// `tag`. Facts come from `record.coerce` (from the coercion to the end of
/// the enclosing dominance scope) and from entering a `record.dispatch`
/// region whose tag set is a singleton.
struct TagFact {
  mlir::TypedValue<RefType> ref;
  int64_t tag;
};

/// One walk over the function in dominance order, carrying the active facts
/// as a scoped stack. Every `ref.drop`/`ref.acquire` resolves against the
/// facts live on its own path, so the work is linear in the function instead
/// of the blocks × coercions dominance queries the previous formulation
/// made (which went quadratic on large inlined match trees).
///
/// A coercion's fact holds only where the coercion *dominates the block*:
/// nested regions of later ops in its own block, and the blocks its block
/// dominates — never `ref.drop`/`ref.acquire` ops later in its own block.
/// Same-block propagation looks sound (the coercion executed earlier on
/// this path) but is not: with token reuse a later ref can must-alias the
/// same address holding a *different* variant, so the stale tag would
/// mis-direct the drop.
class TagInferenceDriver {
public:
  TagInferenceDriver(mlir::func::FuncOp func,
                     mlir::AliasAnalysis &aliasAnalysis)
      : aliasAnalysis(aliasAnalysis), domInfo(func) {}

  void run(mlir::func::FuncOp func) { processRegion(func.getBody()); }

private:
  /// The innermost fact aliasing `ref` among those visible to the current
  /// block — facts the current block introduced itself are excluded.
  std::optional<int64_t> lookup(mlir::TypedValue<RefType> ref) {
    for (const TagFact &fact :
         llvm::reverse(llvm::ArrayRef(facts).take_front(blockStart)))
      if (aliasAnalysis.alias(ref, fact.ref) == mlir::AliasResult::MustAlias)
        return fact.tag;
    return std::nullopt;
  }

  void processOp(mlir::Operation *op) {
    if (auto dropOp = llvm::dyn_cast<ReussirRefDropOp>(op)) {
      if (std::optional<int64_t> tag = lookup(dropOp.getRef()))
        dropOp.setVariantAttr(mlir::IntegerAttr::get(
            mlir::IndexType::get(dropOp.getContext()), *tag));
      return;
    }
    if (auto acquireOp = llvm::dyn_cast<ReussirRefAcquireOp>(op)) {
      if (std::optional<int64_t> tag = lookup(acquireOp.getRef()))
        acquireOp.setVariantAttr(mlir::IntegerAttr::get(
            mlir::IndexType::get(acquireOp.getContext()), *tag));
      return;
    }
    if (auto coerceOp = llvm::dyn_cast<ReussirRecordCoerceOp>(op)) {
      facts.push_back({coerceOp.getVariant(),
                       static_cast<int64_t>(coerceOp.getTag().getZExtValue())});
      return;
    }
    if (auto dispatchOp = llvm::dyn_cast<ReussirRecordDispatchOp>(op)) {
      for (auto [idx, region] : llvm::enumerate(dispatchOp->getRegions())) {
        auto regionTags = llvm::cast<mlir::DenseI64ArrayAttr>(
            dispatchOp.getTagSets()[idx]);
        size_t mark = facts.size();
        if (regionTags.size() == 1)
          facts.push_back({dispatchOp.getVariant(), regionTags[0]});
        processRegion(region);
        facts.truncate(mark);
      }
      return;
    }
    for (mlir::Region &region : op->getRegions())
      processRegion(region);
  }

  /// Process the ops of `block` in order, leaving any facts they introduce
  /// on the stack for the caller's scope to pop. Facts pushed while inside
  /// `block` stay invisible to `block`'s own drops/acquires (see `lookup`),
  /// while nested regions and dominated blocks — entered deeper in the
  /// recursion, where `blockStart` has moved past them — see them.
  void processBlockOps(mlir::Block &block) {
    size_t savedBlockStart = blockStart;
    blockStart = facts.size();
    for (mlir::Operation &op : block)
      processOp(&op);
    blockStart = savedBlockStart;
  }

  /// Process `node`'s block, then its dominator-tree children under the
  /// facts the block introduced — a fact flows exactly into the blocks the
  /// introducing block dominates.
  void processDomNode(mlir::DominanceInfoNode *node) {
    size_t mark = facts.size();
    processBlockOps(*node->getBlock());
    for (mlir::DominanceInfoNode *child : *node)
      processDomNode(child);
    facts.truncate(mark);
  }

  void processRegion(mlir::Region &region) {
    if (region.empty())
      return;
    if (region.hasOneBlock()) {
      size_t mark = facts.size();
      processBlockOps(region.front());
      facts.truncate(mark);
      return;
    }
    processDomNode(domInfo.getRootNode(&region));
  }

  mlir::AliasAnalysis &aliasAnalysis;
  mlir::DominanceInfo domInfo;
  llvm::SmallVector<TagFact> facts;
  /// Number of facts that predate the block currently being processed.
  size_t blockStart = 0;
};
} // namespace

void runTagInference(mlir::func::FuncOp func) {
  mlir::AliasAnalysis aliasAnalysis(func);
  registerAliasAnalysisImplementations(aliasAnalysis);
  TagInferenceDriver(func, aliasAnalysis).run(func);
}

} // namespace reussir
