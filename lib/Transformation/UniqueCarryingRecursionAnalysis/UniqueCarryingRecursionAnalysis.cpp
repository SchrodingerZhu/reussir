//===-- UniqueCarryingRecursionAnalysis.cpp ---------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Uniqueness-carrying analysis and `.unique` specialization.
//
// # The claim
//
// A function *carries uniqueness* when every rc-typed result it returns is,
// on every path, either **fresh** (produced by an `rc.create*` whose count
// is initialized to 1 and never shared afterwards) or **carried** (one of
// the function's own rc arguments, passed through unshared). The payoff:
// at a call site where every carried argument is *itself* proven unique,
// the results are unique too — so a directly self-recursive carrying
// function can run a specialized `.unique` clone that asserts argument
// uniqueness at entry (`rc.assume_unique`, an `llvm.assume(count == 1)`
// after lowering), letting the reuse machinery take unique fast paths
// through the whole recursion.
//
// # Provenance lattice
//
// Per rc value: `Unknown` (bottom) or a pair `{fresh, carriedArgs}` meaning
// "on some path this is a fresh create / argument i". Join is the pointwise
// union: the value is unique iff every contributor is, so a joined value is
// proven unique under an assumption set exactly when *all* its carried bits
// are assumed (freshness needs no assumption). `Unknown` is absorbing in
// proofs (never unique) and the identity of the join.
//
// # Sharing discipline (the part provenance alone cannot see)
//
// Freshness at the definition does not survive sharing: a created value
// that is also retained elsewhere (`rc.inc`, a second consuming user)
// reaches the recursive call with count > 1, and asserting uniqueness on
// it would be undefined behavior. Provenance is therefore *disqualified*
// (demoted to `Unknown`) when the value has any retaining user, or more
// than one consuming user. Forwarding terminators (`scf.yield`,
// `reussir.scf.yield`, `func.return`) and read-only observers (`rc.borrow`,
// `rc.fetch`, `rc.is_unique`, `rc.assume_unique`) do not count: forwards
// sit on mutually exclusive paths (their sharing is checked at the level of
// the forwarded result), and reads do not retain. This is conservative —
// consuming uses on mutually exclusive paths are also rejected — but it is
// what makes the entry assumption sound without a path-sensitive count
// analysis. The frontend's ownership discipline guarantees any genuine
// sharing materializes as an explicit `rc.inc`, which this check sees.
//
// # Control flow
//
// Region results are traced only through operations whose results are
// *exhaustively* defined by their regions' yields: `scf.if`,
// `scf.index_switch`, `scf.execute_region`, and `reussir.record.dispatch`.
// Loop-carrying ops (`scf.for`, `scf.while`) are deliberately Unknown:
// their results may come from the *init* operands on a zero-trip count, a
// path the yield-only join would miss. `reussir.array.with_unique_view` is
// fresh only in its implicit-result form (an empty yield returns the
// uniquified array itself, count 1 by construction); an explicit yield is
// traced like any other forward.
//
// # Fixpoint
//
// Function summaries start at bottom and are recomputed from the previous
// round until stable — a Kleene iteration of a monotone map (joins only
// accumulate bits, bounded by the argument count), so it terminates at the
// least fixpoint. Optimism about recursion is sound coinductively: any
// value a terminating execution actually returns is produced by a finite
// chain of creates/argument passes, which some iteration covers.
//
// # Specialization
//
// For each carrying, directly self-recursive function, the set of provable
// assumption vectors is closed off: starting from the empty assumption,
// each self call contributes the argument subset it can prove, and each
// discovered subset is re-run (the clone bodies are copies of the base, so
// walking the base under the subset's assumptions is equivalent). Each
// subset gets one clone (`f.unique` when it is the only one, else
// `f.unique_<bits>`), marked private/internal and tagged
// `reussir.unique_clone`; debug info is stripped (the clone has no distinct
// source location). Reruns are idempotent: self calls are reset to the base
// symbol before re-deciding, and existing clones are found by the marker.
//
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Transformation/Passes.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRUNIQUECARRYINGRECURSIONANALYSISPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kCarryingUniquenessAttr =
    "reussir.carrying_uniqueness";
static constexpr llvm::StringLiteral kUniqueCloneAttr = "reussir.unique_clone";

struct UniqueCarryingValue {
  bool unknown = true;
  llvm::BitVector carriedArgs;
  bool fresh = false;

  static UniqueCarryingValue getUnknown() { return {}; }

  static UniqueCarryingValue getArgument(unsigned index) {
    UniqueCarryingValue value;
    value.unknown = false;
    value.carriedArgs.resize(index + 1);
    value.carriedArgs.set(index);
    return value;
  }

  static UniqueCarryingValue getFresh() {
    UniqueCarryingValue value;
    value.unknown = false;
    value.fresh = true;
    return value;
  }

  bool operator==(const UniqueCarryingValue &rhs) const {
    return unknown == rhs.unknown && fresh == rhs.fresh &&
           carriedArgs == rhs.carriedArgs;
  }

  static UniqueCarryingValue join(const UniqueCarryingValue &lhs,
                                  const UniqueCarryingValue &rhs) {
    if (lhs.unknown)
      return rhs;
    if (rhs.unknown)
      return lhs;

    UniqueCarryingValue joined;
    joined.unknown = false;
    joined.fresh = lhs.fresh || rhs.fresh;
    joined.carriedArgs.resize(
        std::max(lhs.carriedArgs.size(), rhs.carriedArgs.size()));
    joined.carriedArgs |= lhs.carriedArgs;
    joined.carriedArgs |= rhs.carriedArgs;
    return joined;
  }

  bool isCarrying() const { return !unknown && (fresh || carriedArgs.any()); }

  bool isProvenUniqueUnder(const llvm::BitVector &assumedArgs) const {
    if (!isCarrying())
      return false;
    for (int bit = carriedArgs.find_first(); bit >= 0;
         bit = carriedArgs.find_next(bit)) {
      if (static_cast<unsigned>(bit) >= assumedArgs.size() ||
          !assumedArgs.test(bit))
        return false;
    }
    return true;
  }

  void print(llvm::raw_ostream &os) const {
    if (unknown) {
      os << "Unknown";
      return;
    }
    os << "{fresh=" << (fresh ? "true" : "false") << ", args=[";
    bool first = true;
    for (int bit = carriedArgs.find_first(); bit >= 0;
         bit = carriedArgs.find_next(bit)) {
      if (!first)
        os << ", ";
      os << bit;
      first = false;
    }
    os << "]}";
  }
};

static bool isRcType(mlir::Type type) { return llvm::isa<RcType>(type); }

static bool isAssumableRcType(mlir::Type type) {
  auto rcType = llvm::dyn_cast<RcType>(type);
  return rcType && !rcType.isRegional();
}

static bool hasRcResults(mlir::TypeRange types) {
  return llvm::any_of(types, isRcType);
}

using FunctionSummaryMap =
    llvm::StringMap<llvm::SmallVector<UniqueCarryingValue>>;
using SpecializationEntry = std::pair<std::string, llvm::BitVector>;
using SpecializationMap = llvm::SmallVector<SpecializationEntry>;
using SpecializationNameEntry = std::pair<std::string, std::string>;
using SpecializationNameMap = llvm::SmallVector<SpecializationNameEntry>;
using SpecializationCloneEntry = std::pair<std::string, mlir::func::FuncOp>;
using SpecializationCloneMap = llvm::SmallVector<SpecializationCloneEntry>;

template <typename ValueT>
static ValueT *findSpecializationValue(
    llvm::SmallVectorImpl<std::pair<std::string, ValueT>> &entries,
    llvm::StringRef key) {
  for (auto &entry : entries)
    if (entry.first == key)
      return &entry.second;
  return nullptr;
}

template <typename ValueT>
static const ValueT *findSpecializationValue(
    const llvm::SmallVectorImpl<std::pair<std::string, ValueT>> &entries,
    llvm::StringRef key) {
  for (const auto &entry : entries)
    if (entry.first == key)
      return &entry.second;
  return nullptr;
}

class ProvenanceEvaluator {
public:
  explicit ProvenanceEvaluator(const FunctionSummaryMap &summaries)
      : summaries(summaries) {}

  UniqueCarryingValue evaluate(mlir::Value value) {
    if (!isRcType(value.getType()))
      return UniqueCarryingValue::getUnknown();
    if (auto it = cache.find(value); it != cache.end())
      return it->second;
    if (!active.insert(value).second)
      return UniqueCarryingValue::getUnknown();

    UniqueCarryingValue result = UniqueCarryingValue::getUnknown();
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      auto *owner = blockArg.getOwner();
      if (owner && owner->isEntryBlock()) {
        if (llvm::isa<mlir::func::FuncOp>(owner->getParentOp()))
          result = UniqueCarryingValue::getArgument(blockArg.getArgNumber());
      }
    } else if (mlir::Operation *def = value.getDefiningOp()) {
      result = evaluateResult(
          def, llvm::cast<mlir::OpResult>(value).getResultNumber());
    }

    // Provenance does not survive sharing: a value retained elsewhere (an
    // `rc.inc` alias) or consumed more than once reaches its consumer with
    // count > 1 no matter how it was produced. See the sharing-discipline
    // note in the file header.
    if (result.isCarrying() && isSharedAfterDefinition(value))
      result = UniqueCarryingValue::getUnknown();

    active.erase(value);
    cache[value] = result;
    return result;
  }

  // Whether `value`'s uses can elevate its count on some path before its
  // consumer runs. Forwarding terminators and read-only observers are
  // harmless; an explicit retain always disqualifies; any second consuming
  // user does too (conservatively, even on a mutually exclusive path).
  static bool isSharedAfterDefinition(mlir::Value value) {
    unsigned consumingUses = 0;
    for (mlir::Operation *user : value.getUsers()) {
      if (llvm::isa<ReussirRcIncOp>(user))
        return true;
      if (llvm::isa<mlir::scf::YieldOp, ReussirScfYieldOp,
                    mlir::func::ReturnOp>(user))
        continue;
      if (llvm::isa<ReussirRcBorrowOp, ReussirRcFetchOp, ReussirRcIsUniqueOp,
                    ReussirRcAssumeUniqueOp>(user))
        continue;
      if (++consumingUses > 1)
        return true;
    }
    return false;
  }

private:
  UniqueCarryingValue evaluateResult(mlir::Operation *op,
                                     unsigned resultIndex) {
    // A create initializes its count to 1 — unless `skip_rc` elides that
    // initialization (reuse/TRMC machinery), in which case the box keeps
    // whatever count it already had.
    if (auto create = llvm::dyn_cast<ReussirRcCreateOp>(op))
      return create.getSkipRc() ? UniqueCarryingValue::getUnknown()
                                : UniqueCarryingValue::getFresh();
    if (auto create = llvm::dyn_cast<ReussirRcCreateCompoundOp>(op))
      return create.getSkipRc() ? UniqueCarryingValue::getUnknown()
                                : UniqueCarryingValue::getFresh();
    if (auto create = llvm::dyn_cast<ReussirRcCreateVariantOp>(op))
      return create.getSkipRc() ? UniqueCarryingValue::getUnknown()
                                : UniqueCarryingValue::getFresh();

    if (auto callOp = llvm::dyn_cast<mlir::func::CallOp>(op)) {
      auto it = summaries.find(callOp.getCallee());
      if (it == summaries.end() || resultIndex >= it->second.size())
        return UniqueCarryingValue::getUnknown();
      const UniqueCarryingValue &summary = it->second[resultIndex];
      if (summary.unknown)
        return summary;
      UniqueCarryingValue mapped = UniqueCarryingValue::getUnknown();
      mapped.unknown = false;
      mapped.fresh = summary.fresh;
      for (int argIndex = summary.carriedArgs.find_first(); argIndex >= 0;
           argIndex = summary.carriedArgs.find_next(argIndex)) {
        if (static_cast<unsigned>(argIndex) >= callOp.getNumOperands())
          continue;
        mapped = UniqueCarryingValue::join(
            mapped, evaluate(callOp.getOperand(argIndex)));
      }
      return mapped;
    }

    if (auto withUniqueView =
            llvm::dyn_cast<ReussirArrayWithUniqueViewOp>(op)) {
      auto yieldOp = llvm::dyn_cast<ReussirScfYieldOp>(
          withUniqueView.getBody().front().getTerminator());
      if (!yieldOp)
        return UniqueCarryingValue::getUnknown();
      // The implicit-result form (an empty yield) returns the uniquified
      // array itself: count 1 by the op's contract. An explicit yield is an
      // ordinary forward — same-typed or not, it need not be the array.
      if (withUniqueView.getResult() && yieldOp.getNumOperands() == 0)
        return UniqueCarryingValue::getFresh();
      if (resultIndex >= yieldOp.getNumOperands())
        return UniqueCarryingValue::getUnknown();
      return evaluate(yieldOp.getOperand(resultIndex));
    }

    // A variant dispatch's results are exhaustively defined by its arms'
    // yields — the dominant control-flow shape of frontend-emitted code
    // (every `match`).
    if (llvm::isa<ReussirRecordDispatchOp>(op))
      return joinRegionYields<ReussirScfYieldOp>(op, resultIndex);

    // Only structured ops whose results come *exclusively* from region
    // yields may be traced this way. Loop ops are excluded on purpose:
    // `scf.for`/`scf.while` results can be the init operands (a zero-trip
    // loop), a path a yield-only join would silently miss.
    if (llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp,
                  mlir::scf::ExecuteRegionOp>(op))
      return joinRegionYields<mlir::scf::YieldOp>(op, resultIndex);

    return UniqueCarryingValue::getUnknown();
  }

  template <typename YieldOpT>
  UniqueCarryingValue joinRegionYields(mlir::Operation *op,
                                       unsigned resultIndex) {
    UniqueCarryingValue joined = UniqueCarryingValue::getUnknown();
    for (mlir::Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      auto yieldOp = llvm::dyn_cast<YieldOpT>(region.front().getTerminator());
      if (!yieldOp || resultIndex >= yieldOp->getNumOperands())
        return UniqueCarryingValue::getUnknown();
      joined = UniqueCarryingValue::join(
          joined, evaluate(yieldOp->getOperand(resultIndex)));
    }
    return joined;
  }

  const FunctionSummaryMap &summaries;
  llvm::DenseMap<mlir::Value, UniqueCarryingValue> cache;
  llvm::SmallDenseSet<mlir::Value, 16> active;
};

static void setCarryingUniquenessAttr(mlir::Operation *op, bool enabled) {
  if (enabled)
    op->setAttr(kCarryingUniquenessAttr, mlir::UnitAttr::get(op->getContext()));
  else
    op->removeAttr(kCarryingUniquenessAttr);
}

static bool areAllRcValuesCarrying(mlir::ValueRange values,
                                   ProvenanceEvaluator &evaluator) {
  bool foundRcValue = false;
  for (mlir::Value value : values) {
    if (!isRcType(value.getType()))
      continue;
    foundRcValue = true;
    if (!evaluator.evaluate(value).isCarrying())
      return false;
  }
  return foundRcValue;
}

static bool areEqualSummaries(const FunctionSummaryMap &lhs,
                              const FunctionSummaryMap &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &entry : lhs) {
    auto rhsIt = rhs.find(entry.getKey());
    if (rhsIt == rhs.end() || rhsIt->second != entry.getValue())
      return false;
  }
  return true;
}

static llvm::SmallVector<UniqueCarryingValue>
computeFunctionSummary(mlir::func::FuncOp funcOp,
                       const FunctionSummaryMap &summaries) {
  ProvenanceEvaluator evaluator(summaries);
  llvm::SmallVector<UniqueCarryingValue> summary(
      funcOp.getNumResults(), UniqueCarryingValue::getUnknown());
  funcOp.walk([&](mlir::func::ReturnOp returnOp) {
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (!isRcType(operand.getType()))
        continue;
      summary[index] = UniqueCarryingValue::join(summary[index],
                                                 evaluator.evaluate(operand));
    }
  });
  return summary;
}

static bool isCarryingUniquenessRegionOp(mlir::Operation *op,
                                         const FunctionSummaryMap &summaries) {
  auto *dialect = op->getDialect();
  bool supportedOp = (dialect && dialect->getNamespace() == "scf") ||
                     llvm::isa<ReussirArrayWithUniqueViewOp>(op);
  if (!supportedOp || !hasRcResults(op->getResultTypes()))
    return false;
  ProvenanceEvaluator evaluator(summaries);
  return areAllRcValuesCarrying(op->getResults(), evaluator);
}

static bool isCarryingUniquenessFunction(mlir::func::FuncOp funcOp,
                                         const FunctionSummaryMap &summaries) {
  auto it = summaries.find(funcOp.getName());
  if (it == summaries.end() || funcOp.isDeclaration() ||
      !hasRcResults(funcOp.getResultTypes()))
    return false;
  for (auto [index, type] : llvm::enumerate(funcOp.getResultTypes())) {
    if (!isRcType(type))
      continue;
    if (index >= it->second.size() || !it->second[index].isCarrying())
      return false;
  }
  return true;
}

static bool hasSelfRecursiveCall(mlir::func::FuncOp funcOp) {
  bool found = false;
  llvm::StringRef name = funcOp.getName();
  funcOp.walk([&](mlir::func::CallOp callOp) {
    if (callOp.getCallee() == name)
      found = true;
  });
  return found;
}

// A clone is recognized by its marker attribute; the name check is a
// belt-and-suspenders fallback for IR that predates the marker.
static bool isUniqueClone(mlir::func::FuncOp funcOp) {
  return funcOp->hasAttr(kUniqueCloneAttr) ||
         funcOp.getName().contains(".unique");
}

static bool isUniqueCloneOf(llvm::StringRef baseName, llvm::StringRef callee) {
  if (callee == baseName)
    return true;
  return callee.starts_with((baseName + ".unique").str());
}

static void resetDirectSelfCallsToBase(mlir::func::FuncOp funcOp,
                                       llvm::StringRef baseName) {
  funcOp.walk([&](mlir::func::CallOp callOp) {
    if (isUniqueCloneOf(baseName, callOp.getCallee()))
      callOp.setCallee(baseName);
  });
}

static mlir::LocationAttr stripDebugMetadata(mlir::LocationAttr loc) {
  if (!loc)
    return loc;
  auto *context = loc.getContext();

  if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    llvm::SmallVector<mlir::Location> strippedLocations;
    strippedLocations.reserve(fusedLoc.getLocations().size());
    for (mlir::Location innerLoc : fusedLoc.getLocations())
      strippedLocations.push_back(stripDebugMetadata(innerLoc));
    return mlir::FusedLoc::get(context, strippedLocations);
  }

  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc))
    return mlir::NameLoc::get(nameLoc.getName(),
                              stripDebugMetadata(nameLoc.getChildLoc()));

  if (auto callSiteLoc = llvm::dyn_cast<mlir::CallSiteLoc>(loc))
    return mlir::CallSiteLoc::get(stripDebugMetadata(callSiteLoc.getCallee()),
                                  stripDebugMetadata(callSiteLoc.getCaller()));

  return loc;
}

static void stripCloneDebugInfo(mlir::func::FuncOp funcOp) {
  funcOp->removeAttr("reussir.dbg_func_args");
  funcOp->setLoc(stripDebugMetadata(funcOp.getLoc()));
  funcOp.walk([&](mlir::Operation *op) {
    if (op == funcOp)
      return;
    op->setLoc(stripDebugMetadata(op->getLoc()));
  });
}

static llvm::BitVector
collectCarriedRcArguments(mlir::func::FuncOp funcOp,
                          const FunctionSummaryMap &summaries) {
  llvm::BitVector carriedArgs(funcOp.getNumArguments());
  auto it = summaries.find(funcOp.getName());
  if (it == summaries.end())
    return carriedArgs;
  for (const UniqueCarryingValue &value : it->second) {
    if (value.unknown)
      continue;
    carriedArgs.resize(std::max(carriedArgs.size(), value.carriedArgs.size()));
    carriedArgs |= value.carriedArgs;
  }
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i)
    if (!isAssumableRcType(funcOp.getArgument(i).getType()))
      carriedArgs.reset(i);
  return carriedArgs;
}

static void insertAssumeUniqueAtEntry(mlir::func::FuncOp funcOp,
                                      const llvm::BitVector &carriedArgs) {
  if (funcOp.empty())
    return;

  mlir::OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  for (int index = carriedArgs.find_first(); index >= 0;
       index = carriedArgs.find_next(index)) {
    if (static_cast<unsigned>(index) >= funcOp.getNumArguments())
      continue;
    mlir::BlockArgument arg = funcOp.getArgument(index);
    if (!isAssumableRcType(arg.getType()))
      continue;
    ReussirRcAssumeUniqueOp::create(builder, funcOp.getLoc(), arg);
  }
}

static std::string encodeSpecializationKey(const llvm::BitVector &carriedArgs) {
  std::string key;
  llvm::raw_string_ostream os(key);
  bool first = true;
  for (int bit = carriedArgs.find_first(); bit >= 0;
       bit = carriedArgs.find_next(bit)) {
    if (!first)
      os << '_';
    os << bit;
    first = false;
  }
  return key;
}

static llvm::BitVector computeUniqueCallArguments(
    mlir::func::CallOp callOp, ProvenanceEvaluator &evaluator,
    const llvm::BitVector &assumedArgs, const llvm::BitVector &candidateArgs) {
  llvm::BitVector provenArgs(candidateArgs.size());
  for (int argIndex = candidateArgs.find_first(); argIndex >= 0;
       argIndex = candidateArgs.find_next(argIndex)) {
    if (static_cast<unsigned>(argIndex) >= callOp.getNumOperands())
      continue;
    if (evaluator.evaluate(callOp.getOperand(argIndex))
            .isProvenUniqueUnder(assumedArgs))
      provenArgs.set(argIndex);
  }
  return provenArgs;
}

static SpecializationMap
collectUniqueSpecializations(mlir::func::FuncOp funcOp,
                             const FunctionSummaryMap &summaries,
                             const llvm::BitVector &candidateArgs) {
  SpecializationMap specializations;
  llvm::SmallVector<llvm::BitVector> worklist;

  auto enqueue = [&](const llvm::BitVector &assumedArgs) {
    if (!assumedArgs.any())
      return;
    std::string key = encodeSpecializationKey(assumedArgs);
    if (findSpecializationValue(specializations, key))
      return;
    specializations.push_back({std::move(key), assumedArgs});
    worklist.push_back(assumedArgs);
  };

  auto collectFromAssumedArgs = [&](const llvm::BitVector &assumedArgs) {
    ProvenanceEvaluator evaluator(summaries);
    funcOp.walk([&](mlir::func::CallOp callOp) {
      if (callOp.getCallee() != funcOp.getName())
        return;
      enqueue(computeUniqueCallArguments(callOp, evaluator, assumedArgs,
                                         candidateArgs));
    });
  };

  llvm::BitVector emptyAssumptions(candidateArgs.size());
  collectFromAssumedArgs(emptyAssumptions);
  while (!worklist.empty())
    collectFromAssumedArgs(worklist.pop_back_val());
  return specializations;
}

static SpecializationNameMap
buildSpecializationNames(llvm::StringRef baseName,
                         const SpecializationMap &specializations) {
  SpecializationNameMap names;
  bool usePlainUniqueName = specializations.size() == 1;
  for (const auto &entry : specializations) {
    std::string cloneName = usePlainUniqueName
                                ? (baseName + ".unique").str()
                                : (baseName + ".unique_" + entry.first).str();
    names.push_back({entry.first, std::move(cloneName)});
  }
  return names;
}

static mlir::func::FuncOp getOrCreateUniqueClone(
    mlir::func::FuncOp funcOp, mlir::SymbolTable &symbolTable,
    llvm::StringRef uniqueName, const llvm::BitVector &carriedArgs) {
  if (auto existing = symbolTable.lookup<mlir::func::FuncOp>(uniqueName))
    return existing;

  auto uniqueFunc = llvm::cast<mlir::func::FuncOp>(funcOp->clone());
  uniqueFunc.setName(uniqueName);
  uniqueFunc.setPrivate();
  uniqueFunc->removeAttr("llvm.visibility");
  uniqueFunc->setAttr("llvm.linkage",
                      mlir::LLVM::LinkageAttr::get(
                          funcOp.getContext(), mlir::LLVM::Linkage::Internal));
  uniqueFunc->setAttr(kUniqueCloneAttr,
                      mlir::UnitAttr::get(funcOp.getContext()));
  stripCloneDebugInfo(uniqueFunc);
  symbolTable.insert(uniqueFunc);

  insertAssumeUniqueAtEntry(uniqueFunc, carriedArgs);
  return uniqueFunc;
}

static void rewriteSelfCallsForSpecialization(
    mlir::func::FuncOp funcOp, llvm::StringRef baseName,
    const FunctionSummaryMap &summaries, const llvm::BitVector &assumedArgs,
    const llvm::BitVector &candidateArgs,
    const SpecializationNameMap &specializationNames) {
  ProvenanceEvaluator evaluator(summaries);
  funcOp.walk([&](mlir::func::CallOp callOp) {
    if (callOp.getCallee() != baseName)
      return;
    llvm::BitVector targetArgs = computeUniqueCallArguments(
        callOp, evaluator, assumedArgs, candidateArgs);
    if (!targetArgs.any())
      return;
    const std::string *targetName = findSpecializationValue(
        specializationNames, encodeSpecializationKey(targetArgs));
    if (!targetName)
      return;
    callOp.setCallee(*targetName);
  });
}

struct UniqueCarryingRecursionAnalysisPass
    : public impl::ReussirUniqueCarryingRecursionAnalysisPassBase<
          UniqueCarryingRecursionAnalysisPass> {
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::func::FuncOp>(op) ||
          (op->getDialect() && op->getDialect()->getNamespace() == "scf"))
        op->removeAttr(kCarryingUniquenessAttr);
    });

    FunctionSummaryMap summaries;

    while (true) {
      FunctionSummaryMap nextSummaries;
      moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        nextSummaries[funcOp.getName()] =
            computeFunctionSummary(funcOp, summaries);
      });

      if (areEqualSummaries(summaries, nextSummaries)) {
        summaries = std::move(nextSummaries);
        break;
      }
      summaries = std::move(nextSummaries);
    }

    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      setCarryingUniquenessAttr(
          funcOp, isCarryingUniquenessFunction(funcOp, summaries));
    });
    moduleOp.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::func::FuncOp>(op))
        return;
      setCarryingUniquenessAttr(op,
                                isCarryingUniquenessRegionOp(op, summaries));
    });

    mlir::SymbolTable symbolTable(moduleOp);
    llvm::SmallVector<mlir::func::FuncOp> recursiveFunctions;
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      if (!funcOp->hasAttr(kCarryingUniquenessAttr) || funcOp.isDeclaration() ||
          isUniqueClone(funcOp) || !hasSelfRecursiveCall(funcOp))
        return;
      if (!collectCarriedRcArguments(funcOp, summaries).any())
        return;
      recursiveFunctions.push_back(funcOp);
    });

    for (mlir::func::FuncOp funcOp : recursiveFunctions) {
      llvm::BitVector carriedArgs =
          collectCarriedRcArguments(funcOp, summaries);
      SpecializationMap specializations =
          collectUniqueSpecializations(funcOp, summaries, carriedArgs);
      if (specializations.empty())
        continue;

      SpecializationNameMap specializationNames =
          buildSpecializationNames(funcOp.getName(), specializations);
      SpecializationCloneMap clones;

      resetDirectSelfCallsToBase(funcOp, funcOp.getName());
      for (const auto &entry : specializations) {
        const std::string *cloneName =
            findSpecializationValue(specializationNames, entry.first);
        if (!cloneName)
          continue;
        clones.push_back(
            {entry.first, getOrCreateUniqueClone(funcOp, symbolTable,
                                                 *cloneName, entry.second)});
      }

      llvm::BitVector emptyAssumptions(funcOp.getNumArguments());
      rewriteSelfCallsForSpecialization(funcOp, funcOp.getName(), summaries,
                                        emptyAssumptions, carriedArgs,
                                        specializationNames);
      for (const auto &entry : specializations) {
        mlir::func::FuncOp *cloneFunc =
            findSpecializationValue(clones, entry.first);
        if (!cloneFunc)
          continue;
        resetDirectSelfCallsToBase(*cloneFunc, funcOp.getName());
        rewriteSelfCallsForSpecialization(*cloneFunc, funcOp.getName(),
                                          summaries, entry.second, carriedArgs,
                                          specializationNames);
      }
    }
  }
};

} // namespace
} // namespace reussir
