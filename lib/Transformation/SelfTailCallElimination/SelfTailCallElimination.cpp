//===-- SelfTailCallElimination.cpp - self tail calls to loops --*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Rewrites direct self tail recursion into an `scf.while` loop while the IR
// is still structured, independent of the return type.
//
// TRMC covers self recursion that builds an rc-boxed result; everything else
// — notably a `[value]` enum return like functional-queue's `churn` — used
// to lean on LLVM's TailCallElimination, whose eligibility is heuristic (it
// bails on non-static allocas, and DeadArgumentElimination's
// aggregate-return shrinking can hide the `ret` from it). This pass makes
// the loop structural instead: the function body becomes the before-region
// of an `scf.while` whose carried values are the function arguments, every
// return-position spine op (`scf.if`/`scf.index_switch`) is widened to also
// yield a continue flag and the next iteration's arguments, and each leaf
// yields either "continue with the self call's operands" or "stop with this
// value". Stack slots stay outside the loop: the slot lowering materializes
// them in the entry block (see BasicOpsLowering), so an iteration reuses its
// frame instead of growing it — the reason a naive per-iteration alloca
// would defeat the very stack guarantee this pass exists to give.
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/Passes.h"
#include "Reussir/IR/ReussirDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRSELFTAILCALLELIMINATIONPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

// Whether every op between `op` (exclusive) and its block's terminator is
// side-effect free. Only then does jumping back to the loop header instead
// of returning through the spine preserve behavior.
bool pureUntilTerminator(mlir::Operation *op) {
  for (mlir::Operation *cursor = op->getNextNode();
       cursor != op->getBlock()->getTerminator();
       cursor = cursor->getNextNode())
    if (!mlir::isMemoryEffectFree(cursor))
      return false;
  return true;
}

// A direct self tail call: calls `func`, single-result, its only use is the
// enclosing terminator, and nothing effectful runs between call and
// terminator.
mlir::func::CallOp asSelfTailCall(mlir::Value value, mlir::func::FuncOp func) {
  auto call =
      llvm::dyn_cast_if_present<mlir::func::CallOp>(value.getDefiningOp());
  if (!call || call.getCallee() != func.getName() ||
      call.getNumResults() != 1 || !value.hasOneUse() ||
      !pureUntilTerminator(call))
    return nullptr;
  return call;
}

// A spine op the widening can rebuild: a single-result `scf.if` or
// `scf.index_switch` whose only use is the enclosing terminator, with
// single-block regions each yielding one value, and nothing effectful
// between it and the terminator.
bool isSpineOp(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def || !llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(def) ||
      def->getNumResults() != 1 || !value.hasOneUse() ||
      !pureUntilTerminator(def))
    return false;
  return llvm::all_of(def->getRegions(), [](mlir::Region &region) {
    if (!region.hasOneBlock())
      return false;
    auto yield =
        llvm::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
    return yield && yield.getNumOperands() == 1;
  });
}

// Whether the return-position spine below `value` contains at least one
// rewritable self tail call.
bool spineHasSelfTailCall(mlir::Value value, mlir::func::FuncOp func) {
  if (asSelfTailCall(value, func))
    return true;
  if (!isSpineOp(value))
    return false;
  return llvm::any_of(value.getDefiningOp()->getRegions(),
                      [&](mlir::Region &region) {
                        auto yield = llvm::cast<mlir::scf::YieldOp>(
                            region.front().getTerminator());
                        return spineHasSelfTailCall(yield.getOperand(0), func);
                      });
}

// Rewrites the spine below `value` into widened form and returns the SSA
// values (continue, nextArgs..., result) available right before
// `terminator`. Consumed spine ops and self calls are queued on `toErase`;
// they lose their last use when the caller replaces `terminator`.
llvm::SmallVector<mlir::Value>
widenSpine(mlir::OpBuilder &builder, mlir::Operation *terminator,
           mlir::Value value, mlir::func::FuncOp func,
           llvm::SmallVectorImpl<mlir::Operation *> &toErase) {
  mlir::Location loc = terminator->getLoc();
  mlir::Type resultType = func.getResultTypes().front();

  if (mlir::func::CallOp call = asSelfTailCall(value, func)) {
    builder.setInsertionPoint(terminator);
    llvm::SmallVector<mlir::Value> widened;
    widened.push_back(mlir::arith::ConstantOp::create(
        builder, loc, builder.getBoolAttr(true)));
    llvm::append_range(widened, call.getOperands());
    widened.push_back(mlir::ub::PoisonOp::create(builder, loc, resultType));
    toErase.push_back(call);
    return widened;
  }

  if (isSpineOp(value) && spineHasSelfTailCall(value, func)) {
    mlir::Operation *def = value.getDefiningOp();
    // Rewrite each region's yield to the widened form first, then rebuild
    // the op with the widened result list and steal its regions.
    for (mlir::Region &region : def->getRegions()) {
      auto yield =
          llvm::cast<mlir::scf::YieldOp>(region.front().getTerminator());
      llvm::SmallVector<mlir::Value> widened =
          widenSpine(builder, yield, yield.getOperand(0), func, toErase);
      builder.setInsertionPoint(yield);
      mlir::scf::YieldOp::create(builder, yield.getLoc(), widened);
      yield.erase();
    }
    llvm::SmallVector<mlir::Type> widenedTypes{builder.getI1Type()};
    llvm::append_range(widenedTypes, func.getArgumentTypes());
    widenedTypes.push_back(resultType);
    mlir::OperationState state(def->getLoc(), def->getName().getStringRef());
    state.addOperands(def->getOperands());
    state.addTypes(widenedTypes);
    state.addAttributes(def->getAttrs());
    for (unsigned i = 0, e = def->getNumRegions(); i < e; ++i)
      state.addRegion();
    builder.setInsertionPoint(def);
    mlir::Operation *widenedOp = builder.create(state);
    for (auto [target, source] :
         llvm::zip(widenedOp->getRegions(), def->getRegions()))
      target.takeBody(source);
    toErase.push_back(def);
    return llvm::SmallVector<mlir::Value>(widenedOp->getResults());
  }

  // A finished value: stop the loop with it.
  builder.setInsertionPoint(terminator);
  llvm::SmallVector<mlir::Value> widened;
  widened.push_back(mlir::arith::ConstantOp::create(
      builder, loc, builder.getBoolAttr(false)));
  for (mlir::Type argType : func.getArgumentTypes())
    widened.push_back(mlir::ub::PoisonOp::create(builder, loc, argType));
  widened.push_back(value);
  return widened;
}

void convertToLoop(mlir::func::FuncOp func) {
  mlir::Region &body = func.getBody();
  mlir::Block *oldEntry = &body.front();
  auto returnOp = llvm::cast<mlir::func::ReturnOp>(oldEntry->getTerminator());
  mlir::OpBuilder builder(func.getContext());
  mlir::Location loc = func.getLoc();

  // The loop carries the function arguments and forwards them plus the
  // finished result out of the before-region; the after-region only feeds
  // the carried arguments back.
  llvm::SmallVector<mlir::Type> carriedTypes(func.getArgumentTypes());
  llvm::SmallVector<mlir::Type> forwardedTypes(carriedTypes);
  forwardedTypes.push_back(func.getResultTypes().front());

  llvm::SmallVector<mlir::Location> argLocs(carriedTypes.size(), loc);
  mlir::Block *newEntry =
      builder.createBlock(&body, body.begin(), carriedTypes, argLocs);
  builder.setInsertionPointToEnd(newEntry);
  auto whileOp = mlir::scf::WhileOp::create(builder, loc, forwardedTypes,
                                            newEntry->getArguments());
  mlir::func::ReturnOp::create(builder, loc,
                               whileOp.getResults().take_back(1));

  // The original body becomes the before-region; its block arguments (the
  // old function arguments) become the loop-carried values.
  whileOp.getBefore().getBlocks().splice(whileOp.getBefore().end(),
                                         body.getBlocks(),
                                         oldEntry->getIterator());

  // The after-region feeds the carried arguments back to the next iteration.
  mlir::Block *afterBlock = builder.createBlock(
      &whileOp.getAfter(), whileOp.getAfter().end(), forwardedTypes,
      llvm::SmallVector<mlir::Location>(forwardedTypes.size(), loc));
  builder.setInsertionPointToEnd(afterBlock);
  mlir::scf::YieldOp::create(
      builder, loc, afterBlock->getArguments().take_front(carriedTypes.size()));

  // Widen the return spine and terminate the before-region with the
  // condition.
  llvm::SmallVector<mlir::Operation *> toErase;
  llvm::SmallVector<mlir::Value> widened =
      widenSpine(builder, returnOp, returnOp.getOperand(0), func, toErase);
  builder.setInsertionPoint(returnOp);
  mlir::scf::ConditionOp::create(builder, loc, widened.front(),
                                 llvm::ArrayRef(widened).drop_front());
  returnOp.erase();
  for (mlir::Operation *op : toErase)
    op->erase();
}

struct SelfTailCallEliminationPass
    : public impl::ReussirSelfTailCallEliminationPassBase<
          SelfTailCallEliminationPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (func.isDeclaration() || func.getNumResults() != 1 ||
        !func.getBody().hasOneBlock())
      return;
    auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(
        func.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      return;
    if (!spineHasSelfTailCall(returnOp.getOperand(0), func))
      return;
    convertToLoop(func);
  }
};

} // namespace
} // namespace reussir
