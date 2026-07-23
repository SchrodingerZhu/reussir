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
/// This file implements loop-to-matrix-exponentiation strength reduction for
/// linear recurrences over machine integers.
///
/// A single-block loop whose loop-carried integer state evolves as an affine
/// map with compile-time constant coefficients,
///
///   s_{i+1} = M * s_i + c        (all arithmetic wrapping, i.e. in Z/2^N)
///
/// computes `s_n = A^n * s_0` for the augmented companion matrix
/// `A = [[M, c], [0, 1]]`. Because Z/2^N is a commutative ring, `A^n` can be
/// evaluated with square-and-multiply in O(K^3 log n) ring operations, so the
/// pass replaces the loop (O(n) iterations) with an O(log n) exponentiation
/// loop over the bits of the backedge-taken count.
///
/// Recognition requirements:
///   - the loop is a single block with a preheader and one exit;
///   - ScalarEvolution can compute the exact backedge-taken count;
///   - no instruction in the loop has side effects;
///   - every value used outside the loop is an affine combination (constant
///     coefficients) of the header PHIs of one common integer type, and the
///     closure of those PHIs under their update expressions stays affine.
///
/// Header PHIs outside that closure (e.g. an exit-condition induction
/// variable of a different width) simply die with the loop. Wrapping
/// semantics make the rewrite bit-exact; `nsw`/`nuw` flags on the original
/// arithmetic only license the removal of poison, so dropping them in the
/// emitted code is a sound refinement.
///
/// The recursion linearization pass feeds this one: linear self-recursions
/// become single-block affine loops after inlining and simplification, and
/// this pass then reduces them to logarithmic time — `fib(n)` in O(log n)
/// from a naive doubly-recursive definition.
///
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/LinearRecurrence.h"

#include <optional>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/ScalarEvolutionExpander.h>

namespace reussir::llvmpass {
namespace {

using namespace llvm;

/// Maximum number of state PHIs (excluding the augmentation row). The emitted
/// exponentiation body is O(K^3) instructions for K = state + 1.
constexpr unsigned kMaxStateSize = 6;
/// Cap on expression-walk depth during affine decomposition.
constexpr unsigned kMaxDecomposeDepth = 512;

/// An affine expression `sum coeff_i * phi_i + constant` over the header
/// PHIs, with all arithmetic in Z/2^N.
struct AffineExpr {
  DenseMap<PHINode *, APInt> coeffs;
  APInt constant;
};

/// Decomposes in-loop values into affine expressions over the header PHIs of
/// a single integer type. Results are cached; failure is cached as nullopt.
class AffineDecomposer {
public:
  AffineDecomposer(BasicBlock *header, IntegerType *type)
      : header(header), type(type), bits(type->getBitWidth()) {}

  std::optional<AffineExpr> decompose(Value *value, unsigned depth = 0) {
    auto cached = cache.find(value);
    if (cached != cache.end())
      return cached->second;
    std::optional<AffineExpr> result = decomposeImpl(value, depth);
    cache.try_emplace(value, result);
    return result;
  }

private:
  std::optional<AffineExpr> decomposeImpl(Value *value, unsigned depth) {
    if (depth > kMaxDecomposeDepth)
      return std::nullopt;
    if (value->getType() != type)
      return std::nullopt;
    if (auto *ci = dyn_cast<ConstantInt>(value)) {
      AffineExpr expr;
      expr.constant = ci->getValue();
      return expr;
    }
    if (auto *phi = dyn_cast<PHINode>(value)) {
      if (phi->getParent() != header)
        return std::nullopt;
      AffineExpr expr;
      expr.constant = APInt::getZero(bits);
      expr.coeffs.try_emplace(phi, APInt(bits, 1));
      return expr;
    }
    auto *inst = dyn_cast<Instruction>(value);
    if (!inst || inst->getParent() != header)
      return std::nullopt;
    auto *bin = dyn_cast<BinaryOperator>(inst);
    if (!bin)
      return std::nullopt;
    switch (bin->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub: {
      auto lhs = decompose(bin->getOperand(0), depth + 1);
      auto rhs = decompose(bin->getOperand(1), depth + 1);
      if (!lhs || !rhs)
        return std::nullopt;
      bool negate = bin->getOpcode() == Instruction::Sub;
      AffineExpr expr = *lhs;
      expr.constant += negate ? -rhs->constant : rhs->constant;
      for (const auto &[phi, coeff] : rhs->coeffs) {
        APInt addend = negate ? -coeff : coeff;
        auto [it, inserted] = expr.coeffs.try_emplace(phi, addend);
        if (!inserted)
          it->second += addend;
      }
      return expr;
    }
    case Instruction::Mul: {
      auto lhs = decompose(bin->getOperand(0), depth + 1);
      auto rhs = decompose(bin->getOperand(1), depth + 1);
      if (!lhs || !rhs)
        return std::nullopt;
      // One side must be a pure constant to stay affine.
      const AffineExpr *scalar = lhs->coeffs.empty() ? &*lhs : nullptr;
      const AffineExpr *vector = scalar ? &*rhs : &*lhs;
      if (!scalar) {
        if (!rhs->coeffs.empty())
          return std::nullopt;
        scalar = &*rhs;
      }
      return scale(*vector, scalar->constant);
    }
    case Instruction::Shl: {
      auto *shamt = dyn_cast<ConstantInt>(bin->getOperand(1));
      if (!shamt || shamt->getValue().uge(bits))
        return std::nullopt;
      auto lhs = decompose(bin->getOperand(0), depth + 1);
      if (!lhs)
        return std::nullopt;
      APInt factor = APInt(bits, 1).shl(shamt->getValue());
      return scale(*lhs, factor);
    }
    default:
      return std::nullopt;
    }
  }

  static AffineExpr scale(const AffineExpr &expr, const APInt &factor) {
    AffineExpr result;
    result.constant = expr.constant * factor;
    for (const auto &[phi, coeff] : expr.coeffs)
      result.coeffs.try_emplace(phi, coeff * factor);
    return result;
  }

  BasicBlock *header;
  IntegerType *type;
  unsigned bits;
  DenseMap<Value *, std::optional<AffineExpr>> cache;
};

struct LoopCandidate {
  Loop *loop = nullptr;
  BasicBlock *header = nullptr;
  BasicBlock *preheader = nullptr;
  BasicBlock *exitBlock = nullptr;
  IntegerType *type = nullptr;
  /// Ordered state PHIs; row i of the matrix updates state[i].
  SmallVector<PHINode *, 4> state;
  SmallVector<AffineExpr, 4> updates;
  const SCEV *backedgeCount = nullptr;
  /// Materialized backedge-taken count (filled in during expansion).
  Value *exponent = nullptr;
  bool needsAugment = false;
};

std::optional<LoopCandidate> analyzeLoop(Loop *loop, ScalarEvolution &se) {
  if (loop->getNumBlocks() != 1)
    return std::nullopt;
  BasicBlock *header = loop->getHeader();
  BasicBlock *preheader = loop->getLoopPreheader();
  if (!preheader || !isa<BranchInst>(preheader->getTerminator()))
    return std::nullopt;
  auto *latchBranch = dyn_cast<BranchInst>(header->getTerminator());
  if (!latchBranch || !latchBranch->isConditional())
    return std::nullopt;
  BasicBlock *succ0 = latchBranch->getSuccessor(0);
  BasicBlock *succ1 = latchBranch->getSuccessor(1);
  if ((succ0 == header) == (succ1 == header))
    return std::nullopt;
  BasicBlock *exitBlock = succ0 == header ? succ1 : succ0;

  const SCEV *backedgeCount = se.getBackedgeTakenCount(loop);
  if (isa<SCEVCouldNotCompute>(backedgeCount) ||
      !backedgeCount->getType()->isIntegerTy())
    return std::nullopt;

  for (Instruction &inst : *header)
    if (inst.mayHaveSideEffects())
      return std::nullopt;

  // Gather in-loop values used outside the loop; each must be affine over
  // header PHIs of one shared integer type.
  SmallVector<Instruction *, 4> liveOuts;
  for (Instruction &inst : *header)
    for (User *user : inst.users())
      if (cast<Instruction>(user)->getParent() != header) {
        liveOuts.push_back(&inst);
        break;
      }
  if (liveOuts.empty())
    return std::nullopt; // Dead loop; leave it to DCE.

  auto *type = dyn_cast<IntegerType>(liveOuts.front()->getType());
  if (!type)
    return std::nullopt;
  AffineDecomposer decomposer(header, type);

  SetVector<PHINode *> statePhis;
  SmallVector<AffineExpr, 4> liveOutExprs;
  for (Instruction *liveOut : liveOuts) {
    auto expr = decomposer.decompose(liveOut);
    if (!expr)
      return std::nullopt;
    for (const auto &[phi, coeff] : expr->coeffs)
      statePhis.insert(phi);
  }
  // Close the state set under the update expressions.
  SmallVector<AffineExpr, 4> updates;
  for (unsigned i = 0; i < statePhis.size(); ++i) {
    PHINode *phi = statePhis[i];
    auto update = decomposer.decompose(phi->getIncomingValueForBlock(header));
    if (!update)
      return std::nullopt;
    for (const auto &[usedPhi, coeff] : update->coeffs)
      statePhis.insert(usedPhi);
    updates.push_back(std::move(*update));
  }
  if (statePhis.size() > kMaxStateSize)
    return std::nullopt;

  LoopCandidate candidate;
  candidate.loop = loop;
  candidate.header = header;
  candidate.preheader = preheader;
  candidate.exitBlock = exitBlock;
  candidate.type = type;
  candidate.state.assign(statePhis.begin(), statePhis.end());
  candidate.updates = std::move(updates);
  candidate.backedgeCount = backedgeCount;
  for (const AffineExpr &update : candidate.updates)
    if (!update.constant.isZero())
      candidate.needsAugment = true;
  return candidate;
}

/// Emits `out = lhs * rhs` for row-major K x K matrices of SSA values.
SmallVector<Value *, 16> emitMatMul(IRBuilder<> &builder,
                                    ArrayRef<Value *> lhs,
                                    ArrayRef<Value *> rhs, unsigned dim) {
  SmallVector<Value *, 16> out(dim * dim);
  for (unsigned i = 0; i < dim; ++i)
    for (unsigned j = 0; j < dim; ++j) {
      Value *acc = nullptr;
      for (unsigned l = 0; l < dim; ++l) {
        Value *prod =
            builder.CreateMul(lhs[i * dim + l], rhs[l * dim + j]);
        acc = acc ? builder.CreateAdd(acc, prod) : prod;
      }
      out[i * dim + j] = acc;
    }
  return out;
}

/// Emits `coeff * value` folding the trivial coefficients.
Value *emitScaled(IRBuilder<> &builder, const APInt &coeff, Value *value,
                  IntegerType *type) {
  if (coeff.isZero())
    return nullptr;
  if (coeff.isOne())
    return value;
  return builder.CreateMul(ConstantInt::get(type, coeff), value);
}

Value *emitAffine(IRBuilder<> &builder, const AffineExpr &expr,
                  ArrayRef<PHINode *> state, ArrayRef<Value *> stateValues,
                  IntegerType *type) {
  Value *acc = nullptr;
  for (unsigned i = 0; i < state.size(); ++i) {
    auto it = expr.coeffs.find(state[i]);
    if (it == expr.coeffs.end())
      continue;
    Value *term = emitScaled(builder, it->second, stateValues[i], type);
    if (!term)
      continue;
    acc = acc ? builder.CreateAdd(acc, term) : term;
  }
  if (!expr.constant.isZero() || !acc) {
    Constant *c = ConstantInt::get(type, expr.constant);
    acc = acc ? builder.CreateAdd(acc, c) : static_cast<Value *>(c);
  }
  return acc;
}

/// Rewrites one candidate loop into a square-and-multiply exponentiation of
/// its augmented companion matrix. Returns false (leaving the function
/// untouched) if new live-outs appeared since analysis that are not affine —
/// e.g. uses introduced by expanding another candidate's trip count.
bool rewriteLoop(LoopCandidate &candidate) {
  BasicBlock *header = candidate.header;
  BasicBlock *preheader = candidate.preheader;
  BasicBlock *exitBlock = candidate.exitBlock;
  IntegerType *type = candidate.type;
  Function *function = header->getParent();
  LLVMContext &ctx = function->getContext();
  unsigned stateSize = candidate.state.size();
  unsigned dim = stateSize + (candidate.needsAugment ? 1 : 0);

  // Re-collect live-outs: expansion of other candidates' trip counts may
  // have added uses of header values since analysis ran.
  AffineDecomposer decomposer(header, type);
  SmallVector<std::pair<Instruction *, AffineExpr>, 4> liveOuts;
  for (Instruction &inst : *header) {
    bool usedOutside = false;
    for (User *user : inst.users())
      if (cast<Instruction>(user)->getParent() != header) {
        usedOutside = true;
        break;
      }
    if (!usedOutside)
      continue;
    auto expr = decomposer.decompose(&inst);
    if (!expr)
      return false;
    for (const auto &[phi, coeff] : expr->coeffs)
      if (!llvm::is_contained(candidate.state, phi))
        return false;
    liveOuts.emplace_back(&inst, std::move(*expr));
  }

  // Companion matrix (row-major, row i updates state[i]; optional trailing
  // augmentation row [0 ... 0 1] carrying the affine constants).
  SmallVector<Constant *, 16> companion(dim * dim);
  SmallVector<Constant *, 16> identity(dim * dim);
  for (unsigned i = 0; i < dim; ++i)
    for (unsigned j = 0; j < dim; ++j) {
      identity[i * dim + j] =
          ConstantInt::get(type, i == j ? 1 : 0);
      APInt entry = APInt::getZero(type->getBitWidth());
      if (i < stateSize) {
        if (j < stateSize) {
          auto it = candidate.updates[i].coeffs.find(candidate.state[j]);
          if (it != candidate.updates[i].coeffs.end())
            entry = it->second;
        } else {
          entry = candidate.updates[i].constant;
        }
      } else if (j == dim - 1) {
        entry = APInt(type->getBitWidth(), 1);
      }
      companion[i * dim + j] = ConstantInt::get(type, entry);
    }

  BasicBlock *check =
      BasicBlock::Create(ctx, "reussir.matexp.check", function, exitBlock);
  BasicBlock *body =
      BasicBlock::Create(ctx, "reussir.matexp.body", function, exitBlock);
  BasicBlock *done =
      BasicBlock::Create(ctx, "reussir.matexp.done", function, exitBlock);

  auto *preBranch = cast<BranchInst>(preheader->getTerminator());
  preBranch->setSuccessor(0, check);

  Value *exponent = candidate.exponent;
  auto *exponentTy = cast<IntegerType>(exponent->getType());

  // check: while (e != 0)
  IRBuilder<> builder(check);
  PHINode *ePhi = builder.CreatePHI(exponentTy, 2, "reussir.matexp.e");
  SmallVector<PHINode *, 16> accPhis(dim * dim);
  SmallVector<PHINode *, 16> basePhis(dim * dim);
  for (unsigned i = 0; i < dim * dim; ++i)
    accPhis[i] = builder.CreatePHI(type, 2, "reussir.matexp.acc");
  for (unsigned i = 0; i < dim * dim; ++i)
    basePhis[i] = builder.CreatePHI(type, 2, "reussir.matexp.base");
  Value *eIsZero =
      builder.CreateICmpEQ(ePhi, ConstantInt::get(exponentTy, 0));
  builder.CreateCondBr(eIsZero, done, body);

  // body: if (e & 1) acc *= base; base *= base; e >>= 1
  builder.SetInsertPoint(body);
  Value *odd = builder.CreateTrunc(ePhi, Type::getInt1Ty(ctx),
                                   "reussir.matexp.odd");
  SmallVector<Value *, 16> accValues(accPhis.begin(), accPhis.end());
  SmallVector<Value *, 16> baseValues(basePhis.begin(), basePhis.end());
  SmallVector<Value *, 16> accTimesBase =
      emitMatMul(builder, accValues, baseValues, dim);
  SmallVector<Value *, 16> accNext(dim * dim);
  for (unsigned i = 0; i < dim * dim; ++i)
    accNext[i] = builder.CreateSelect(odd, accTimesBase[i], accPhis[i]);
  SmallVector<Value *, 16> baseNext =
      emitMatMul(builder, baseValues, baseValues, dim);
  Value *eNext =
      builder.CreateLShr(ePhi, ConstantInt::get(exponentTy, 1));
  builder.CreateBr(check);

  ePhi->addIncoming(exponent, preheader);
  ePhi->addIncoming(eNext, body);
  for (unsigned i = 0; i < dim * dim; ++i) {
    accPhis[i]->addIncoming(identity[i], preheader);
    accPhis[i]->addIncoming(accNext[i], body);
    basePhis[i]->addIncoming(companion[i], preheader);
    basePhis[i]->addIncoming(baseNext[i], body);
  }

  // done: final = acc * s_0, then rebuild the live-outs from it.
  builder.SetInsertPoint(done);
  SmallVector<Value *, 8> initial(dim);
  for (unsigned j = 0; j < stateSize; ++j)
    initial[j] = candidate.state[j]->getIncomingValueForBlock(preheader);
  if (candidate.needsAugment)
    initial[dim - 1] = ConstantInt::get(type, 1);
  SmallVector<Value *, 8> finalState(stateSize);
  for (unsigned i = 0; i < stateSize; ++i) {
    Value *acc = nullptr;
    for (unsigned j = 0; j < dim; ++j) {
      Value *term = builder.CreateMul(accPhis[i * dim + j], initial[j]);
      acc = acc ? builder.CreateAdd(acc, term) : term;
    }
    finalState[i] = acc;
  }
  for (auto &[inst, expr] : liveOuts) {
    Value *replacement =
        emitAffine(builder, expr, candidate.state, finalState, type);
    inst->replaceUsesWithIf(replacement, [&](Use &use) {
      return cast<Instruction>(use.getUser())->getParent() != header;
    });
  }
  builder.CreateBr(exitBlock);
  exitBlock->replacePhiUsesWith(header, done);

  header->dropAllReferences();
  header->eraseFromParent();
  return true;
}

} // namespace

llvm::PreservedAnalyses
LinearRecurrenceMatExpPass::run(llvm::Function &function,
                                llvm::FunctionAnalysisManager &fam) {
  using namespace llvm;
  auto &loopInfo = fam.getResult<LoopAnalysis>(function);
  auto &se = fam.getResult<ScalarEvolutionAnalysis>(function);

  // Phase 1: analysis over the intact IR.
  SmallVector<LoopCandidate, 4> candidates;
  for (Loop *loop : loopInfo.getLoopsInPreorder())
    if (auto candidate = analyzeLoop(loop, se))
      candidates.push_back(std::move(*candidate));
  if (candidates.empty())
    return llvm::PreservedAnalyses::all();

  // Phase 2: materialize every trip count while ScalarEvolution still
  // matches the IR.
  const llvm::DataLayout &dataLayout = function.getParent()->getDataLayout();
  SCEVExpander expander(se, dataLayout, "reussir.matexp");
  SmallVector<LoopCandidate, 4> expandable;
  for (LoopCandidate &candidate : candidates) {
    llvm::Instruction *insertion = candidate.preheader->getTerminator();
    if (!expander.isSafeToExpandAt(candidate.backedgeCount, insertion))
      continue;
    candidate.exponent = expander.expandCodeFor(
        candidate.backedgeCount, candidate.backedgeCount->getType(),
        insertion);
    expandable.push_back(std::move(candidate));
  }

  // Phase 3: structural rewrites (no analysis use past this point). Even if
  // every rewrite backs out, materialized trip counts already changed the IR.
  bool changed = !expandable.empty();
  for (LoopCandidate &candidate : expandable)
    changed |= rewriteLoop(candidate);
  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace reussir::llvmpass
