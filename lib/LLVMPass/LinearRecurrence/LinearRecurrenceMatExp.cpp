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
/// This file implements loop-to-logarithmic strength reduction for linear
/// recurrences over machine integers, via the Kitamasa method.
///
/// A single-block loop whose loop-carried integer state evolves as an affine
/// map with compile-time constant coefficients,
///
///   s_{i+1} = M * s_i + c        (all arithmetic wrapping, i.e. in Z/2^N)
///
/// computes `s_n = A^n * s_0` for the augmented companion matrix
/// `A = [[M, c], [0, 1]]` of dimension D. Rather than exponentiating the
/// matrix directly (O(D^3) per squaring), the pass evaluates `z^n mod
/// chi_A(z)` in the quotient ring (Z/2^N)[z]/chi_A by square-and-multiply —
/// O(D^2) ring operations per squaring — and then combines the resulting
/// polynomial `p` with the Krylov vectors of the initial state:
///
///   A^n s_0 = p(A) s_0 = sum_i p_i (A^i s_0)     (Cayley–Hamilton)
///
/// Cayley–Hamilton is a polynomial identity over Z, so it holds in any
/// commutative ring, including Z/2^N; the characteristic polynomial is
/// computed at compile time by division-free cofactor expansion mod 2^N and
/// is monic by construction, so the reductions need no division either. The
/// Krylov vectors `A^i s_0` for `i < D` are emitted once as straight-line
/// code, with the (typically very sparse, 0/±1) constant matrix entries
/// folded away. Overall: O(D^2 log n) emitted ring operations, the
/// FFT-free Kitamasa bound — an FFT-based O(D log D log n) variant only
/// wins for D far beyond the size cap here.
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

#include <cassert>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/bit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/ScalarEvolutionExpander.h>

namespace reussir::llvmpass {
namespace {

using namespace llvm;

/// Maximum number of state PHIs (excluding the augmentation row). The emitted
/// exponentiation body is O(D^2) instructions for D = state + 1.
constexpr unsigned kMaxStateSize = 6;
/// Cap on expression-walk depth during affine decomposition.
constexpr unsigned kMaxDecomposeDepth = 512;

/// An affine expression `sum coeff_i * phi_i + constant` over the header
/// PHIs, with all arithmetic in Z/2^N.
struct AffineExpr {
  DenseMap<PHINode *, APInt> coeffs;
  APInt constant;
};

/// A polynomial over Z/2^N, coefficients low degree first.
using Poly = SmallVector<APInt, 8>;

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
  /// Ordered state PHIs; row i of the companion matrix updates state[i].
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

//===----------------------------------------------------------------------===//
// Compile-time polynomial arithmetic over Z/2^N
//===----------------------------------------------------------------------===//

Poly polyMul(const Poly &lhs, const Poly &rhs, unsigned bits) {
  Poly out(lhs.size() + rhs.size() - 1, APInt::getZero(bits));
  for (unsigned i = 0; i < lhs.size(); ++i)
    for (unsigned j = 0; j < rhs.size(); ++j)
      out[i + j] += lhs[i] * rhs[j];
  return out;
}

void polyAddInPlace(Poly &lhs, const Poly &rhs, bool negate, unsigned bits) {
  if (lhs.size() < rhs.size())
    lhs.resize(rhs.size(), APInt::getZero(bits));
  for (unsigned i = 0; i < rhs.size(); ++i)
    lhs[i] += negate ? -rhs[i] : rhs[i];
}

/// Characteristic polynomial `det(zI - A) mod 2^N` of the row-major
/// `dim x dim` matrix, by cofactor expansion memoized on the column subset —
/// division-free, so valid over Z/2^N; monic of degree `dim` by construction.
Poly charPoly(ArrayRef<APInt> matrix, unsigned dim, unsigned bits) {
  DenseMap<uint32_t, Poly> memo;
  // det over rows [row..dim) and the columns in `mask`, where
  // row = dim - popcount(mask).
  auto det = [&](auto &&self, uint32_t mask) -> Poly {
    if (mask == 0)
      return Poly{APInt(bits, 1)};
    auto cached = memo.find(mask);
    if (cached != memo.end())
      return cached->second;
    unsigned row = dim - llvm::popcount(mask);
    Poly result{APInt::getZero(bits)};
    bool negate = false;
    for (unsigned col = 0; col < dim; ++col) {
      if (!(mask & (1U << col)))
        continue;
      // Entry (zI - A)[row][col]: degree <= 1.
      Poly entry{-matrix[size_t(row) * dim + col]};
      if (row == col)
        entry.push_back(APInt(bits, 1));
      Poly sub = self(self, mask & ~(1U << col));
      polyAddInPlace(result, polyMul(entry, sub, bits), negate, bits);
      negate = !negate;
    }
    memo.try_emplace(mask, result);
    return result;
  };
  Poly chi = det(det, (1U << dim) - 1);
  chi.resize(dim + 1, APInt::getZero(bits));
  assert(chi[dim].isOne() && "characteristic polynomial must be monic");
  return chi;
}

//===----------------------------------------------------------------------===//
// IR emission helpers
//===----------------------------------------------------------------------===//

/// Emits `coeff * value` folding the trivial coefficients (0, 1, -1).
Value *emitScaled(IRBuilder<> &builder, const APInt &coeff, Value *value,
                  IntegerType *type) {
  if (coeff.isZero())
    return nullptr;
  if (coeff.isOne())
    return value;
  if (coeff.isAllOnes())
    return builder.CreateNeg(value);
  return builder.CreateMul(ConstantInt::get(type, coeff), value);
}

/// Emits `acc + term`, treating null as zero.
Value *emitAccumulate(IRBuilder<> &builder, Value *acc, Value *term) {
  if (!term)
    return acc;
  return acc ? builder.CreateAdd(acc, term) : term;
}

Value *emitAffine(IRBuilder<> &builder, const AffineExpr &expr,
                  ArrayRef<PHINode *> state, ArrayRef<Value *> stateValues,
                  IntegerType *type) {
  Value *acc = nullptr;
  for (unsigned i = 0; i < state.size(); ++i) {
    auto it = expr.coeffs.find(state[i]);
    if (it == expr.coeffs.end())
      continue;
    acc = emitAccumulate(builder, acc,
                         emitScaled(builder, it->second, stateValues[i], type));
  }
  if (!expr.constant.isZero() || !acc) {
    Constant *c = ConstantInt::get(type, expr.constant);
    acc = acc ? builder.CreateAdd(acc, c) : static_cast<Value *>(c);
  }
  return acc;
}

/// Emits `lhs * rhs mod chi` for polynomials of degree < D given as SSA
/// coefficient vectors, `chi` monic of degree D: O(D^2) ring operations.
SmallVector<Value *, 8> emitPolyModMul(IRBuilder<> &builder,
                                       ArrayRef<Value *> lhs,
                                       ArrayRef<Value *> rhs, const Poly &chi,
                                       IntegerType *type) {
  unsigned dim = lhs.size();
  // Full product, degree <= 2D-2.
  SmallVector<Value *, 16> prod(2 * size_t(dim) - 1, nullptr);
  for (unsigned i = 0; i < dim; ++i)
    for (unsigned j = 0; j < dim; ++j)
      prod[i + j] = emitAccumulate(builder, prod[i + j],
                                   builder.CreateMul(lhs[i], rhs[j]));
  // Reduce top-down: z^k = z^(k-D) z^D = -sum_j chi_j z^(k-D+j) (mod chi).
  for (unsigned k = 2 * dim - 2; k >= dim; --k) {
    Value *top = prod[k];
    for (unsigned j = 0; j < dim; ++j) {
      Value *term = emitScaled(builder, chi[j], top, type);
      if (!term)
        continue;
      unsigned at = k - dim + j;
      prod[at] = prod[at] ? builder.CreateSub(prod[at], term)
                          : builder.CreateNeg(term);
    }
  }
  prod.truncate(dim);
  return prod;
}

/// Rewrites one candidate loop into Kitamasa exponentiation: `z^btc mod
/// chi_A` by square-and-multiply, then `A^btc s_0 = sum_i p_i (A^i s_0)`.
/// Returns false (leaving the function untouched) if new live-outs appeared
/// since analysis that are not affine — e.g. uses introduced by expanding
/// another candidate's trip count.
bool rewriteLoop(LoopCandidate &candidate) {
  BasicBlock *header = candidate.header;
  BasicBlock *preheader = candidate.preheader;
  BasicBlock *exitBlock = candidate.exitBlock;
  IntegerType *type = candidate.type;
  Function *function = header->getParent();
  LLVMContext &ctx = function->getContext();
  unsigned bits = type->getBitWidth();
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
  // augmentation row [0 ... 0 1] carrying the affine constants) and its
  // characteristic polynomial.
  SmallVector<APInt, 16> companion(size_t(dim) * dim, APInt::getZero(bits));
  for (unsigned i = 0; i < stateSize; ++i) {
    for (unsigned j = 0; j < stateSize; ++j) {
      auto it = candidate.updates[i].coeffs.find(candidate.state[j]);
      if (it != candidate.updates[i].coeffs.end())
        companion[size_t(i) * dim + j] = it->second;
    }
    if (candidate.needsAugment)
      companion[size_t(i) * dim + (dim - 1)] = candidate.updates[i].constant;
  }
  if (candidate.needsAugment)
    companion[size_t(dim - 1) * dim + (dim - 1)] = APInt(bits, 1);
  Poly chi = charPoly(companion, dim, bits);

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

  // check: while (e != 0). acc/base are polynomials of degree < D in
  // (Z/2^N)[z]/chi; acc starts at 1, base at `z mod chi`.
  IRBuilder<> builder(check);
  PHINode *ePhi = builder.CreatePHI(exponentTy, 2, "reussir.matexp.e");
  SmallVector<PHINode *, 8> accPhis(dim);
  SmallVector<PHINode *, 8> basePhis(dim);
  for (unsigned i = 0; i < dim; ++i)
    accPhis[i] = builder.CreatePHI(type, 2, "reussir.matexp.acc");
  for (unsigned i = 0; i < dim; ++i)
    basePhis[i] = builder.CreatePHI(type, 2, "reussir.matexp.base");
  Value *eIsZero = builder.CreateICmpEQ(ePhi, ConstantInt::get(exponentTy, 0));
  builder.CreateCondBr(eIsZero, done, body);

  // body: if (e & 1) acc = acc*base mod chi; base = base^2 mod chi; e >>= 1
  builder.SetInsertPoint(body);
  Value *odd = builder.CreateTrunc(ePhi, Type::getInt1Ty(ctx),
                                   "reussir.matexp.odd");
  SmallVector<Value *, 8> accValues(accPhis.begin(), accPhis.end());
  SmallVector<Value *, 8> baseValues(basePhis.begin(), basePhis.end());
  SmallVector<Value *, 8> accTimesBase =
      emitPolyModMul(builder, accValues, baseValues, chi, type);
  SmallVector<Value *, 8> accNext(dim);
  for (unsigned i = 0; i < dim; ++i)
    accNext[i] = builder.CreateSelect(odd, accTimesBase[i], accPhis[i]);
  SmallVector<Value *, 8> baseNext =
      emitPolyModMul(builder, baseValues, baseValues, chi, type);
  Value *eNext = builder.CreateLShr(ePhi, ConstantInt::get(exponentTy, 1));
  builder.CreateBr(check);

  // Initial polynomials: acc = 1; base = z mod chi (which is -chi_0 when
  // D == 1, else just z).
  SmallVector<Constant *, 8> accInit(dim), baseInit(dim);
  for (unsigned i = 0; i < dim; ++i) {
    accInit[i] = ConstantInt::get(type, i == 0 ? 1 : 0);
    APInt baseCoeff = APInt::getZero(bits);
    if (dim == 1)
      baseCoeff = -chi[0];
    else if (i == 1)
      baseCoeff = APInt(bits, 1);
    baseInit[i] = ConstantInt::get(type, baseCoeff);
  }
  ePhi->addIncoming(exponent, preheader);
  ePhi->addIncoming(eNext, body);
  for (unsigned i = 0; i < dim; ++i) {
    accPhis[i]->addIncoming(accInit[i], preheader);
    accPhis[i]->addIncoming(accNext[i], body);
    basePhis[i]->addIncoming(baseInit[i], preheader);
    basePhis[i]->addIncoming(baseNext[i], body);
  }

  // done: s_n = sum_i p_i * (A^i s_0). The Krylov vectors A^i s_0 are
  // straight-line constant-matrix/vector products, with the (typically
  // sparse, 0/±1) companion entries folded away.
  builder.SetInsertPoint(done);
  SmallVector<Value *, 8> krylov(dim);
  for (unsigned j = 0; j < stateSize; ++j)
    krylov[j] = candidate.state[j]->getIncomingValueForBlock(preheader);
  if (candidate.needsAugment)
    krylov[dim - 1] = ConstantInt::get(type, 1);
  SmallVector<Value *, 8> result(dim, nullptr);
  for (unsigned i = 0; i < dim; ++i) {
    for (unsigned r = 0; r < dim; ++r)
      result[r] = emitAccumulate(
          builder, result[r], builder.CreateMul(accPhis[i], krylov[r]));
    if (i + 1 == dim)
      break;
    SmallVector<Value *, 8> next(dim, nullptr);
    for (unsigned r = 0; r < dim; ++r) {
      for (unsigned c = 0; c < dim; ++c)
        next[r] = emitAccumulate(
            builder, next[r],
            emitScaled(builder, companion[size_t(r) * dim + c], krylov[c], type));
      if (!next[r])
        next[r] = ConstantInt::get(type, 0);
    }
    krylov = std::move(next);
  }
  SmallVector<Value *, 8> finalState(result.begin(),
                                     result.begin() + stateSize);
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
#if LLVM_VERSION_MAJOR >= 22
  SCEVExpander expander(se, "reussir.matexp");
#else
  SCEVExpander expander(se, function.getParent()->getDataLayout(),
                        "reussir.matexp");
#endif
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
