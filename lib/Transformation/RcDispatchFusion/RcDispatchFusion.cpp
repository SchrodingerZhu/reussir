//===-- RcDispatchFusion.cpp ------------------------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Fuses a pattern match's consumption of its scrutinee into a *destructuring*
// decrement (Koka's `dropn_reuse` shape).
//
// The frontend lowers `match v { Ctor(a, b, ..) => .. }` into a borrowing
// dispatch whose taken arm retains every bound member and then decrements the
// scrutinee:
//
//   %ref = reussir.rc.borrow %v
//   reussir.record.dispatch(%ref) {
//     [k] -> ^arm(%payload):
//       %a = ref.load(ref.project %payload [i])
//       rc.inc %a                      // retain the binding
//       ...
//       rc.dec %v                      // release the box (transitive glue)
//   }
//
// The retain/release pair over each bound member is pure count traffic on the
// hot unique path: the box dies and its slots could simply transfer. This
// pass erases the bound retains and tags the decrement with the arm's tag and
// the bound member indices; `reussir-rc-decrement-expansion` then expands it
// shallowly — unique: release only *unbound* rc members and take the box as a
// token; shared: retain the bound members and drop the count. A preceding
// increment cancels against the destructuring decrement in
// `reussir-inc-dec-cancellation` by rematerializing the bound retains
// (borrow semantics), which is what turns inlined read-only predicates like
// rbtree's `is_red` into pure tag reads.
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRRCDISPATCHFUSIONPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

// The member index a retained value was extracted from, when `value` is a
// direct `ref.load(ref.project(payload, i))` of the arm's payload reference.
std::optional<int64_t> extractedMemberIndex(mlir::Value value,
                                            mlir::Value payloadRef) {
  auto load =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!load)
    return std::nullopt;
  auto project = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      load.getRef().getDefiningOp());
  if (!project || project.getRef() != payloadRef)
    return std::nullopt;
  return project.getIndex().getSExtValue();
}

// Fuse one arm: find the scrutinee's release, collect the bound retains
// before it, and rewrite. Returns whether the arm was fused.
bool fuseArm(mlir::Region &region, int64_t tag,
             mlir::TypedValue<RcType> scrutinee) {
  if (!region.hasOneBlock() || region.front().getNumArguments() != 1)
    return false;
  mlir::Value payloadRef = region.front().getArgument(0);

  // Locate the first release of the scrutinee at the arm's top level. Every
  // op before it must leave the box's count alone: borrows, projections,
  // loads, and retains of *other* values are fine; anything else that
  // touches the scrutinee (or is opaque) aborts.
  ReussirRcDecOp dec;
  llvm::SmallVector<ReussirRcIncOp> boundIncs;
  for (mlir::Operation &op : region.front()) {
    if (auto candidate = llvm::dyn_cast<ReussirRcDecOp>(op)) {
      if (candidate.getRcPtr() == scrutinee) {
        dec = candidate;
        break;
      }
      // A release of a value loaded from the scrutinee's box would race the
      // shallow expansion; only accept releases of unrelated values.
      if (extractedMemberIndex(candidate.getRcPtr(), payloadRef))
        return false;
      continue;
    }
    if (auto inc = llvm::dyn_cast<ReussirRcIncOp>(op)) {
      if (extractedMemberIndex(inc.getRcPtr(), payloadRef))
        boundIncs.push_back(inc);
      continue;
    }
    if (llvm::is_contained(op.getOperands(), mlir::Value(scrutinee)) &&
        !llvm::isa<ReussirRcBorrowOp>(op))
      return false;
    // Region-bearing or opaque ops before the release: bail (the release
    // may be conditional or the box may escape).
    if (op.getNumRegions() > 0 || llvm::isa<mlir::CallOpInterface>(op))
      return false;
  }
  if (!dec || dec.isDestructuring())
    return false;
  if (dec.getNullableToken() && !dec.getNullableToken().use_empty())
    return false;

  llvm::SmallVector<int64_t> bound;
  for (ReussirRcIncOp inc : boundIncs)
    bound.push_back(*extractedMemberIndex(inc.getRcPtr(), payloadRef));

  mlir::OpBuilder builder(dec);
  dec.setDestructureTagAttr(builder.getIndexAttr(tag));
  dec.setBoundMembersAttr(builder.getDenseI64ArrayAttr(bound));
  for (ReussirRcIncOp inc : boundIncs)
    inc.erase();
  return true;
}

struct RcDispatchFusionPass
    : public impl::ReussirRcDispatchFusionPassBase<RcDispatchFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](ReussirRecordDispatchOp dispatch) {
      auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
          dispatch.getVariant().getDefiningOp());
      if (!borrow)
        return;
      mlir::TypedValue<RcType> scrutinee = borrow.getRcPtr();
      RcType rcType = scrutinee.getType();
      if (rcType.getAtomicKind() != AtomicKind::normal || rcType.isRegional())
        return;
      auto recordType =
          llvm::dyn_cast<RecordType>(rcType.getElementType());
      if (!recordType || !recordType.isVariant() || !recordType.getComplete())
        return;
      for (auto [tagSetAttr, region] :
           llvm::zip(dispatch.getTagSets(), dispatch.getRegions())) {
        auto tagSet = llvm::cast<mlir::DenseI64ArrayAttr>(tagSetAttr);
        if (tagSet.size() != 1)
          continue;
        // The arm's payload must be a plain record view: fused member
        // handling only understands value/shared members.
        auto payload = llvm::dyn_cast<RecordType>(
            recordType.getMembers()[tagSet[0]]);
        if (!payload || !payload.getComplete() || !payload.hasNoRegionalFields())
          continue;
        fuseArm(region, tagSet[0], scrutinee);
      }
    });
  }
};

} // namespace
} // namespace reussir
