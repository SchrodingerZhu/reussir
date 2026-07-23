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
/// This file implements recursion linearization: rewriting pure, directly
/// self-recursive functions of the shape
///
///   f(x, c...) = combine(x, c..., f(x - d_1, c...), ..., f(x - d_k, c...))
///
/// into a dynamic-programming loop over a sliding window of the last
/// `k = max d_i` results. `combine` may be any side-effect-free, speculatable
/// computation (so modular sums, min/max-plus combinations, and other
/// generalized recurrences are all covered), and the base cases may be any
/// call-free paths guarded by comparisons of `x` against constants.
///
/// The transformation is semantics-preserving without any linearity
/// assumption, because it never reassociates the user's arithmetic. It
/// outlines the original body into a `step` helper in which each recursive
/// call `f(x - d, c...)` is replaced by an extra parameter `y_d`, then
/// rewrites `f` as
///
///   f(x, c...):
///     if (x < C)  return step(x, c..., poison...)      // call-free region
///     w_j = step(C - j, c..., poison...)  for j = 1..k // seeds, also < C
///     for (v = C; ; ++v) {
///       cur = step(v, c..., w_1, ..., w_k)
///       if (v == x) return cur
///       shift window: w_1 = cur, w_j = w_{j-1}
///     }
///
/// where `C` is a proven lower bound of `x` at every recursive call site
/// (derived from dominating comparisons against constants). Whenever `x < C`
/// no recursive call executes in the original body, so on those paths the
/// window parameters of `step` are dead and may be poison; whenever `x >= C`
/// the window carries exactly `f(v-1), ..., f(v-k)`, so `step` computes
/// `f(v)`. Every instruction is required to be speculatable because the loop
/// evaluates `f` at points the original call tree may have skipped.
///
/// The helper is marked `alwaysinline`; after the standard pipeline inlines
/// and simplifies it, first-order *linear* recurrences surface as single-block
/// affine loops that `LinearRecurrenceMatExpPass` can further rewrite into
/// logarithmic-time matrix exponentiation.
///
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/LinearRecurrence.h"

#include <optional>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

namespace reussir::llvmpass {
namespace {

/// Largest recursion depth window (max `d_i`) we are willing to carry.
constexpr uint64_t kMaxWindow = 16;
/// Narrow induction integers interact badly with the window arithmetic below
/// (e.g. `floor - window` wrapping through the whole domain), so require a
/// reasonable width.
constexpr unsigned kMinInductionBits = 8;

struct SelfCall {
  llvm::CallInst *call;
  /// Positive decrement of the induction argument: the call is
  /// `f(..., x - offset, ...)`.
  uint64_t offset;
};

struct RecursionShape {
  unsigned inductionIndex = 0;
  /// Window size `k = max offset`.
  uint64_t window = 0;
  /// Lower bound `C` of the induction argument at every recursive call.
  llvm::APInt floor;
  /// Lattice in which `floor` is a lower bound (and in which the rewritten
  /// entry guard compares).
  bool isSigned = false;
  llvm::SmallVector<SelfCall, 4> calls;
};

/// Matches `v == x - d` for a positive constant `d`, accepting both the
/// `sub x, d` and the canonical `add x, -d` spellings.
std::optional<llvm::APInt> matchDecrement(llvm::Value *value,
                                          llvm::Argument *induction) {
  auto *bin = llvm::dyn_cast<llvm::BinaryOperator>(value);
  if (!bin)
    return std::nullopt;
  llvm::Value *lhs = bin->getOperand(0);
  llvm::Value *rhs = bin->getOperand(1);
  if (bin->getOpcode() == llvm::Instruction::Add) {
    if (lhs != induction)
      std::swap(lhs, rhs);
    if (lhs != induction)
      return std::nullopt;
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(rhs))
      return -ci->getValue();
    return std::nullopt;
  }
  if (bin->getOpcode() == llvm::Instruction::Sub) {
    if (lhs != induction)
      return std::nullopt;
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(rhs))
      return ci->getValue();
  }
  return std::nullopt;
}

/// Whether `inst` is admissible inside a linearizable body: side-effect free,
/// non-trapping for *arbitrary* inputs (the rewritten loop evaluates the body
/// at points the original call tree may never have reached), and free of
/// memory access. Recursive calls are validated separately and passed in via
/// `selfCalls`.
bool isAllowedInstruction(llvm::Instruction &inst,
                          const llvm::SmallPtrSetImpl<llvm::CallInst *> &selfCalls) {
  using namespace llvm;
  if (isa<PHINode>(inst) || isa<SelectInst>(inst) || isa<CmpInst>(inst) ||
      isa<CastInst>(inst) || isa<FreezeInst>(inst) ||
      isa<ExtractValueInst>(inst) || isa<InsertValueInst>(inst) ||
      isa<ExtractElementInst>(inst) || isa<InsertElementInst>(inst) ||
      isa<ShuffleVectorInst>(inst) || isa<UnaryOperator>(inst) ||
      isa<BranchInst>(inst) || isa<ReturnInst>(inst))
    return true;
  if (auto *bin = dyn_cast<BinaryOperator>(&inst)) {
    switch (bin->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::URem:
    case Instruction::SDiv:
    case Instruction::SRem: {
      // Division traps on a zero divisor (and signed division on
      // INT_MIN / -1), which would be genuine UB introduced at the extra
      // evaluation points; only constant, provably benign divisors pass.
      auto *divisor = dyn_cast<ConstantInt>(bin->getOperand(1));
      if (!divisor || divisor->isZero())
        return false;
      bool isSignedDiv = bin->getOpcode() == Instruction::SDiv ||
                         bin->getOpcode() == Instruction::SRem;
      return !(isSignedDiv && divisor->isMinusOne());
    }
    default:
      return true;
    }
  }
  if (auto *call = dyn_cast<CallInst>(&inst)) {
    if (selfCalls.contains(call))
      return true;
    // Speculatable, effect-free intrinsics (smax, ctpop, fshl, ...) are fine;
    // anything else (including any non-intrinsic call) is rejected.
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(call))
      return !intrinsic->mayHaveSideEffects() &&
             !intrinsic->mayReadFromMemory() &&
             isSafeToSpeculativelyExecute(intrinsic);
    return false;
  }
  return false;
}

/// Computes the recursion floor `C`: a constant such that every recursive
/// call site is dominated by comparisons proving `x >= C` (in the returned
/// signedness lattice), hence `x < C` implies the body is call-free.
std::optional<std::pair<llvm::APInt, bool>>
computeFloor(llvm::Function &function, llvm::Argument *induction,
             llvm::ArrayRef<SelfCall> calls, llvm::DominatorTree &domTree,
             uint64_t window) {
  using namespace llvm;
  unsigned bits = induction->getType()->getIntegerBitWidth();
  bool sawSigned = false;
  bool sawUnsigned = false;
  SmallVector<ConstantRange, 4> ranges;

  for (const SelfCall &selfCall : calls) {
    ConstantRange range = ConstantRange::getFull(bits);
    BasicBlock *callBlock = selfCall.call->getParent();
    for (BasicBlock &block : function) {
      auto *branch = dyn_cast<BranchInst>(block.getTerminator());
      if (!branch || !branch->isConditional() ||
          branch->getSuccessor(0) == branch->getSuccessor(1))
        continue;
      auto *cmp = dyn_cast<ICmpInst>(branch->getCondition());
      if (!cmp)
        continue;
      Value *lhs = cmp->getOperand(0);
      Value *rhs = cmp->getOperand(1);
      ICmpInst::Predicate pred = cmp->getPredicate();
      if (lhs != induction) {
        std::swap(lhs, rhs);
        pred = ICmpInst::getSwappedPredicate(pred);
      }
      if (lhs != induction)
        continue;
      auto *bound = dyn_cast<ConstantInt>(rhs);
      if (!bound)
        continue;
      for (unsigned succIdx : {0U, 1U}) {
        BasicBlockEdge edge(&block, branch->getSuccessor(succIdx));
        if (!domTree.dominates(edge, callBlock))
          continue;
        ICmpInst::Predicate edgePred =
            succIdx == 0 ? pred : ICmpInst::getInversePredicate(pred);
        range = range.intersectWith(
            ConstantRange::makeAllowedICmpRegion(edgePred, bound->getValue()));
        if (ICmpInst::isSigned(edgePred))
          sawSigned = true;
        if (ICmpInst::isUnsigned(edgePred))
          sawUnsigned = true;
      }
    }
    // A full range means the call is unguarded; an empty range means it is
    // dominated by contradictory conditions (unreachable) — be conservative
    // in both cases.
    if (range.isFullSet() || range.isEmptySet())
      return std::nullopt;
    ranges.push_back(range);
  }

  auto tryLattice =
      [&](bool isSigned) -> std::optional<std::pair<llvm::APInt, bool>> {
    APInt floor = isSigned ? APInt::getSignedMaxValue(bits)
                           : APInt::getMaxValue(bits);
    for (const ConstantRange &range : ranges) {
      APInt low = isSigned ? range.getSignedMin() : range.getUnsignedMin();
      if (isSigned ? low.slt(floor) : low.ult(floor))
        floor = low;
    }
    // The seeds evaluate `step` at `C - 1 .. C - window`; require that these
    // stay above the lattice minimum so the guard `x < C` covers them and no
    // wrap-around occurs.
    APInt latticeMin = isSigned ? APInt::getSignedMinValue(bits)
                                : APInt::getZero(bits);
    APInt slack = latticeMin + window;
    bool inBounds = isSigned ? floor.sge(slack) : floor.uge(slack);
    if (!inBounds)
      return std::nullopt;
    return std::make_pair(floor, isSigned);
  };

  // Prefer the lattice the source's own guards speak: a signed-guarded
  // recursion linearized with an unsigned floor would still be correct, but
  // negative inputs would walk the whole unsigned domain instead of taking
  // the base path directly.
  bool signedFirst = sawSigned && !sawUnsigned;
  if (auto result = tryLattice(signedFirst))
    return result;
  return tryLattice(!signedFirst);
}

std::optional<RecursionShape> matchRecursion(llvm::Function &function,
                                             llvm::DominatorTree &domTree) {
  using namespace llvm;
  if (function.isDeclaration() || function.isVarArg() ||
      function.hasFnAttribute(Attribute::OptimizeNone))
    return std::nullopt;
  Type *retTy = function.getReturnType();
  if (retTy->isVoidTy() || retTy->isTokenTy() || !retTy->isFirstClassType())
    return std::nullopt;

  // The loop below evaluates the body at every point from the floor up to
  // the query, so internal loops (whose termination at unevaluated points is
  // unknown) are not admissible.
  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 8> backedges;
  FindFunctionBackedges(function, backedges);
  if (!backedges.empty())
    return std::nullopt;

  // Collect direct self-calls.
  SmallVector<CallInst *, 4> selfCallInsts;
  for (BasicBlock &block : function)
    for (Instruction &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst))
        if (call->getCalledFunction() == &function)
          selfCallInsts.push_back(call);
  if (selfCallInsts.empty())
    return std::nullopt;

  // Exactly one argument position may vary across recursive calls: the
  // induction argument. All other arguments must be passed through untouched
  // so the recursion is a one-dimensional sequence.
  std::optional<unsigned> inductionIndex;
  for (CallInst *call : selfCallInsts) {
    if (call->arg_size() != function.arg_size() || call->hasOperandBundles() ||
        call->getFunctionType() != function.getFunctionType())
      return std::nullopt;
    std::optional<unsigned> varying;
    for (unsigned i = 0; i < call->arg_size(); ++i) {
      if (call->getArgOperand(i) == function.getArg(i))
        continue;
      if (varying)
        return std::nullopt;
      varying = i;
    }
    if (!varying)
      return std::nullopt; // `f(x) = ... f(x) ...` never makes progress.
    if (inductionIndex && *inductionIndex != *varying)
      return std::nullopt;
    inductionIndex = *varying;
  }

  auto *inductionTy =
      dyn_cast<IntegerType>(function.getArg(*inductionIndex)->getType());
  if (!inductionTy || inductionTy->getBitWidth() < kMinInductionBits)
    return std::nullopt;
  Argument *induction = function.getArg(*inductionIndex);

  RecursionShape shape;
  shape.inductionIndex = *inductionIndex;
  for (CallInst *call : selfCallInsts) {
    std::optional<APInt> offset =
        matchDecrement(call->getArgOperand(*inductionIndex), induction);
    if (!offset || offset->isZero() || offset->ugt(kMaxWindow))
      return std::nullopt;
    uint64_t d = offset->getZExtValue();
    shape.calls.push_back({call, d});
    shape.window = std::max(shape.window, d);
  }

  SmallPtrSet<CallInst *, 4> selfCallSet(selfCallInsts.begin(),
                                         selfCallInsts.end());
  for (BasicBlock &block : function)
    for (Instruction &inst : block)
      if (!isAllowedInstruction(inst, selfCallSet))
        return std::nullopt;

  auto floor = computeFloor(function, induction, shape.calls, domTree,
                            shape.window);
  if (!floor)
    return std::nullopt;
  shape.floor = floor->first;
  shape.isSigned = floor->second;
  return shape;
}

/// Outlines the body of `function` into a private `step` helper whose extra
/// trailing parameters stand in for the recursive results, then rebuilds
/// `function` as the guarded dynamic-programming loop described in the file
/// header.
void rewriteFunction(llvm::Function &function, const RecursionShape &shape) {
  using namespace llvm;
  Module &module = *function.getParent();
  LLVMContext &ctx = function.getContext();
  Type *retTy = function.getReturnType();
  auto *inductionTy =
      cast<IntegerType>(function.getArg(shape.inductionIndex)->getType());
  uint64_t window = shape.window;

  // Build the step helper by cloning the body and substituting the window
  // parameters for the recursive calls.
  SmallVector<Type *, 8> paramTys(function.getFunctionType()->params().begin(),
                                  function.getFunctionType()->params().end());
  for (uint64_t i = 0; i < window; ++i)
    paramTys.push_back(retTy);
  auto *stepTy = FunctionType::get(retTy, paramTys, /*isVarArg=*/false);
  Function *step =
      Function::Create(stepTy, GlobalValue::PrivateLinkage,
                       function.getName() + ".reussir.step", &module);
  ValueToValueMapTy valueMap;
  for (unsigned i = 0; i < function.arg_size(); ++i) {
    valueMap[function.getArg(i)] = step->getArg(i);
    step->getArg(i)->setName(function.getArg(i)->getName());
  }
  for (uint64_t j = 1; j <= window; ++j)
    step->getArg(function.arg_size() + j - 1)->setName("prev" + Twine(j));
  SmallVector<ReturnInst *, 4> returns;
  CloneFunctionInto(step, &function, valueMap,
                    CloneFunctionChangeType::LocalChangesOnly, returns);
  // copyAttributesFrom inside the clone reproduces the original's visibility
  // and dso_local flag; a private helper must be dso_local with default
  // visibility.
  step->setVisibility(GlobalValue::DefaultVisibility);
  step->setDSOLocal(true);
  // The helper is an internal implementation detail; detach it from the
  // original's debug identity rather than carrying a duplicated subprogram.
  step->setSubprogram(nullptr);
  stripDebugInfo(*step);
  for (const SelfCall &selfCall : shape.calls) {
    auto *cloned = cast<CallInst>(valueMap[selfCall.call]);
    Argument *replacement =
        step->getArg(function.arg_size() + selfCall.offset - 1);
    cloned->replaceAllUsesWith(replacement);
    cloned->eraseFromParent();
  }
  step->removeFnAttr(Attribute::NoInline);
  step->addFnAttr(Attribute::AlwaysInline);
  step->addFnAttr(Attribute::NoUnwind);

  // Rebuild `function` from scratch.
  for (BasicBlock &block : function)
    block.dropAllReferences();
  while (!function.empty())
    function.begin()->eraseFromParent();

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", &function);
  BasicBlock *baseBlock = BasicBlock::Create(ctx, "reussir.base", &function);
  BasicBlock *seedBlock = BasicBlock::Create(ctx, "reussir.seed", &function);
  BasicBlock *loopBlock = BasicBlock::Create(ctx, "reussir.dp.loop", &function);
  BasicBlock *exitBlock = BasicBlock::Create(ctx, "reussir.dp.exit", &function);

  Argument *induction = function.getArg(shape.inductionIndex);
  Value *poison = PoisonValue::get(retTy);
  auto makeArgs = [&](Value *inductionValue, ArrayRef<Value *> windowValues) {
    SmallVector<Value *, 8> args;
    for (unsigned i = 0; i < function.arg_size(); ++i)
      args.push_back(i == shape.inductionIndex
                         ? inductionValue
                         : static_cast<Value *>(function.getArg(i)));
    for (uint64_t j = 0; j < window; ++j)
      args.push_back(j < windowValues.size() ? windowValues[j] : poison);
    return args;
  };

  IRBuilder<> builder(entry);
  Constant *floorC = ConstantInt::get(inductionTy, shape.floor);
  Value *isBase = builder.CreateICmp(
      shape.isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT, induction,
      floorC, "reussir.isbase");
  builder.CreateCondBr(isBase, baseBlock, seedBlock);

  // Below the floor the body is call-free, so the window is dead.
  builder.SetInsertPoint(baseBlock);
  CallInst *baseResult =
      builder.CreateCall(step, makeArgs(induction, {}), "reussir.basecase");
  builder.CreateRet(baseResult);

  // Seed the window with f(C-1) .. f(C-k); all of these are below the floor
  // as well.
  builder.SetInsertPoint(seedBlock);
  SmallVector<Value *, 8> seeds;
  for (uint64_t j = 1; j <= window; ++j) {
    Constant *seedPoint =
        ConstantInt::get(inductionTy, shape.floor - APInt(inductionTy->getBitWidth(), j));
    seeds.push_back(builder.CreateCall(step, makeArgs(seedPoint, {}),
                                       "reussir.seed" + Twine(j)));
  }
  builder.CreateBr(loopBlock);

  // Walk v from C to x, maintaining w_j = f(v - j).
  builder.SetInsertPoint(loopBlock);
  PHINode *iv = builder.CreatePHI(inductionTy, 2, "reussir.iv");
  SmallVector<PHINode *, 8> windowPhis;
  for (uint64_t j = 1; j <= window; ++j)
    windowPhis.push_back(
        builder.CreatePHI(retTy, 2, "reussir.w" + Twine(j)));
  SmallVector<Value *, 8> windowValues(windowPhis.begin(), windowPhis.end());
  CallInst *current =
      builder.CreateCall(step, makeArgs(iv, windowValues), "reussir.cur");
  Value *ivNext = builder.CreateAdd(iv, ConstantInt::get(inductionTy, 1),
                                    "reussir.iv.next");
  Value *done = builder.CreateICmpEQ(iv, induction, "reussir.done");
  builder.CreateCondBr(done, exitBlock, loopBlock);
  iv->addIncoming(floorC, seedBlock);
  iv->addIncoming(ivNext, loopBlock);
  for (uint64_t j = 0; j < window; ++j) {
    windowPhis[j]->addIncoming(seeds[j], seedBlock);
    windowPhis[j]->addIncoming(j == 0 ? static_cast<Value *>(current)
                                      : windowPhis[j - 1],
                               loopBlock);
  }

  builder.SetInsertPoint(exitBlock);
  builder.CreateRet(current);
}

} // namespace

llvm::PreservedAnalyses
RecursionLinearizationPass::run(llvm::Module &module,
                                llvm::ModuleAnalysisManager &mam) {
  using namespace llvm;
  auto &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(module).getManager();
  bool changed = false;
  // Snapshot: rewriting appends the outlined helpers to the module.
  SmallVector<Function *, 16> functions;
  for (Function &function : module)
    functions.push_back(&function);
  for (Function *function : functions) {
    if (function->isDeclaration())
      continue;
    auto &domTree = fam.getResult<DominatorTreeAnalysis>(*function);
    if (auto shape = matchRecursion(*function, domTree)) {
      rewriteFunction(*function, *shape);
      changed = true;
    }
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace reussir::llvmpass
