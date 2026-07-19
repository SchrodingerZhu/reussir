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
/// This file implements the Reussir invariant group analysis.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/InvariantGroupAnalysis.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRINVARIANTGROUPANALYSISPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Lattice value: three-state per SSA value
//===----------------------------------------------------------------------===//

enum class InvariantState : uint8_t {
  Unknown = 0, // bottom — not yet analyzed
  Safe = 1,    // provably from rc.borrow / closure.inspect_payload, no
               // flex+field projection
  Unsafe = 2,  // conservative top — not proven safe
};

struct InvariantGroupValue {
  InvariantState state = InvariantState::Unknown;

  static InvariantGroupValue getUnknown() { return {InvariantState::Unknown}; }
  static InvariantGroupValue getSafe() { return {InvariantState::Safe}; }
  static InvariantGroupValue getUnsafe() { return {InvariantState::Unsafe}; }

  bool operator==(const InvariantGroupValue &rhs) const {
    return state == rhs.state;
  }

  /// Join: Unknown ⊔ X = X, Safe ⊔ Safe = Safe, otherwise Unsafe
  static InvariantGroupValue join(const InvariantGroupValue &lhs,
                                  const InvariantGroupValue &rhs) {
    if (lhs.state == InvariantState::Unknown)
      return rhs;
    if (rhs.state == InvariantState::Unknown)
      return lhs;
    if (lhs.state == InvariantState::Safe && rhs.state == InvariantState::Safe)
      return getSafe();
    return getUnsafe();
  }

  void print(llvm::raw_ostream &os) const {
    switch (state) {
    case InvariantState::Unknown:
      os << "Unknown";
      break;
    case InvariantState::Safe:
      os << "Safe";
      break;
    case InvariantState::Unsafe:
      os << "Unsafe";
      break;
    }
  }
};

//===----------------------------------------------------------------------===//
// Lattice element wrapping InvariantGroupValue
//===----------------------------------------------------------------------===//

class InvariantGroupLattice
    : public mlir::dataflow::Lattice<InvariantGroupValue> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InvariantGroupLattice)
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// Sparse forward dataflow analysis
//===----------------------------------------------------------------------===//

class InvariantGroupAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<
          InvariantGroupLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const InvariantGroupLattice *> operands,
                 llvm::ArrayRef<InvariantGroupLattice *> results) override;

  void setToEntryState(InvariantGroupLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(InvariantGroupValue::getUnsafe()));
  }
};

mlir::LogicalResult InvariantGroupAnalysis::visitOperation(
    mlir::Operation *op, llvm::ArrayRef<const InvariantGroupLattice *> operands,
    llvm::ArrayRef<InvariantGroupLattice *> results) {

  // rc.borrow → Safe
  if (llvm::isa<ReussirRcBorrowOp>(op)) {
    for (auto *result : results)
      propagateIfChanged(result, result->join(InvariantGroupValue::getSafe()));
    return mlir::success();
  }

  // closure.inspect_payload → Safe
  if (llvm::isa<ReussirClosureInspectPayloadOp>(op)) {
    for (auto *result : results)
      propagateIfChanged(result, result->join(InvariantGroupValue::getSafe()));
    return mlir::success();
  }

  // ref.project: propagate input state, but check flex+field soundness
  if (auto projectOp = llvm::dyn_cast<ReussirRefProjectOp>(op)) {
    assert(operands.size() >= 1 && "ref.project must have at least 1 operand");
    const auto &inputState = operands[0]->getValue();

    // A cell projection exposes the mutable slot. Even though rc.borrow makes
    // the containing box identity stable, repeated accesses through this
    // address may observe different values, so invariant.group must never
    // propagate across the cell boundary.
    auto inputRefType = llvm::cast<RefType>(projectOp.getRef().getType());
    if (llvm::isa<CellType>(inputRefType.getElementType())) {
      for (auto *result : results)
        propagateIfChanged(result,
                           result->join(InvariantGroupValue::getUnsafe()));
      return mlir::success();
    }

    if (inputState.state == InvariantState::Safe) {
      // Check whether the input ref has flex capability AND the projected
      // field is mutable (memberIsField). If so → Unsafe.
      auto recordType =
          llvm::dyn_cast<RecordType>(inputRefType.getElementType());

      if (recordType && recordType.getComplete()) {
        size_t index = projectOp.getIndex().getZExtValue();
        bool isField = recordType.getMemberIsField()[index];
        bool isFlex = inputRefType.getCapability() == Capability::flex;

        if (isFlex && isField) {
          // flex + field projection → Unsafe
          for (auto *result : results)
            propagateIfChanged(result,
                               result->join(InvariantGroupValue::getUnsafe()));
          return mlir::success();
        }
      }

      // Safe propagation
      for (auto *result : results)
        propagateIfChanged(result,
                           result->join(InvariantGroupValue::getSafe()));
      return mlir::success();
    }

    // Otherwise propagate input state as-is
    for (auto *result : results)
      propagateIfChanged(result, result->join(inputState));
    return mlir::success();
  }

  // All other ops: check results. Those producing RefType get Unsafe,
  // others get default (unchanged).
  for (auto [idx, result] : llvm::enumerate(results)) {
    mlir::Value resVal = op->getResult(idx);
    if (llvm::isa<RefType>(resVal.getType()))
      propagateIfChanged(result,
                         result->join(InvariantGroupValue::getUnsafe()));
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct InvariantGroupAnalysisPass
    : public impl::ReussirInvariantGroupAnalysisPassBase<
          InvariantGroupAnalysisPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *ctx = funcOp.getContext();

    // Set up the dataflow solver
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<InvariantGroupAnalysis>();

    if (failed(solver.initializeAndRun(funcOp))) {
      signalPassFailure();
      return;
    }

    // Walk all RefLoadOps and annotate those whose ref operand is Safe
    funcOp.walk([&](ReussirRefLoadOp loadOp) {
      auto *lattice =
          solver.lookupState<InvariantGroupLattice>(loadOp.getRef());
      if (lattice && lattice->getValue().state == InvariantState::Safe)
        loadOp.setInvariantGroupAttr(mlir::UnitAttr::get(ctx));
    });
  }
};

} // namespace
} // namespace reussir
