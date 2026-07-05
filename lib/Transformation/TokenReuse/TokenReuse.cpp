//===-- TokenReuse.cpp - Reussir token reuse pass impl ----------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 With LLVM Exceptions OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/TokenReuse.h"
#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/Conversion/RcDecrementExpansion.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirInterfaces.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <bit>
#include <optional>
#include <tuple>
#include <cstddef>
#include <functional>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <immer/set.hpp>
#pragma GCC diagnostic pop
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/xxhash.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

namespace reussir {
#define GEN_PASS_DEF_REUSSIRTOKENREUSEPASS
#include "Reussir/Transformation/Passes.h.inc"

// This Pass implements token reuse optimization for Reussir dialect.
// This is currently a one-shot token assignment pass that does not rely on
// iterative data-flow analysis.
// Assumptions and Current Limitations:
// - IR passed in this pass is still in structured control flow form.
// - Currently, if there is exceptional control flow or loop, we do not proceed
//   the analysis. Before entering such region, we free all pending tokens and
//   give up the optimization.
// - This pass assumes all token producer operations (basically drop operations)
//   are inserted already.
// - IPA/IPO is not performed, so we give up optimization across function.
//   + if target function is marked with "leaf/no_alloc", we consider keeping
//     tokens across function call.
//   + otherwise, we free all pending tokens before function call to avoid
//     non-deterministic heap growth.
// - We do not go into regions that are `IsolatedFromAbove`, e.g., region of
//   `reussir.closure.create` operation. So, for closure to work properly,
//   the closure must the outlined first. We `signalPassFailure` if we encounter
//   closure operation with inlined region.
// This pass works as the follows:
// 1. We start with an empty set of available tokens. We use immer's persistent
//    data structure to reduce the overhead of set maintenance.
// 2. We traverse the IR following the DAG, maintaining the fact that available
//    tokens always dominate the current position. So, on exiting a region (via
//    `Terminator`), we return a set of token that are still available intersect
//    with the token that dominates the region parent.
// 3. Encountering a token producer operation (e.g., `reussir.token.dec`), we
//    add it into the pool if the token has no user. (This makes the analysis
//    compatible with pre-deallocated tokens in special cases.)
// 4. Encountering a token consumer operation (e.g., `reussir.token.inc`), we
//    use heuristic to determine the best token to use from the available pool.
//    Remove the selected token from the pool. Currently, we just iterate all
//    the tokens to find the most suitable one.
// 5. At `CallOpInterface` or `LoopLikeOpInterface`, we free all pending tokens.
//    (TODO: Exceptional/Early-Exit Regions do not exist in MLIR yet but
//    maintainers are actively working on it. We need to monitor the progress.)
// 6. At `RegionBranchInterface`, we continue on each nested branch and
//    intersect the result from each of them as remaining available tokens.
//
// The above process only describes how we maintain the available token set.
// Each time the token set change, we actually need to emit corresponding
// operations.
// 1. If we assign a token to a consumer operation, we need to insert:
//    - a `reussir.token.ensure` operation if token is nullable and the original
//      layout is compatible with target layout.
//    - a `reussir.token.realloc` operation if the original layout is not
//      compatible with target layout. (`realloc` operation also supports
//      nullable token.)
// 2. At terminator, if token befores unusable at parent, either because of
//    dominance or because token is consumed in other branches, we need to
//    insert `reussir.token.free` operation to free the token.
// 3. Similarly, at function call or loop, we need to free all pending tokens
//    with `reussir.token.free` operation.
// The change is applied lazily. During the first traversal, we maintain a
// vector of operations where such changes are needed.

namespace {
static inline constexpr size_t MIN_ALLOC_STEP_SIZE = 2 * sizeof(void *);
static inline constexpr size_t MIN_ALLOC_STEP_BITS =
    std::countr_zero(MIN_ALLOC_STEP_SIZE);
static inline constexpr size_t INTERMEDIATE_BITS = 2;
size_t toExpMand(size_t value) {
  auto oneAtBit = [](size_t bit) { return 1 << bit; };
  constexpr size_t LEADING_BIT =
      oneAtBit(INTERMEDIATE_BITS + MIN_ALLOC_STEP_BITS) >> 1;
  constexpr size_t MANTISSA_MASK = oneAtBit(INTERMEDIATE_BITS) - 1;
  constexpr size_t BITS = sizeof(size_t) * CHAR_BIT;

  value = value - 1;

  size_t e = BITS - INTERMEDIATE_BITS - MIN_ALLOC_STEP_BITS -
             __builtin_clz(value | LEADING_BIT);
  size_t b = (e == 0) ? 0 : 1;
  size_t m = (value >> (MIN_ALLOC_STEP_BITS + e - b)) & MANTISSA_MASK;

  return (e << INTERMEDIATE_BITS) + m;
}
// Guess if two tokens are in the same size class.
// Computation logic is from SnMalloc, may not apply to other allocators.
bool possiblyInplaceReallocable(size_t oldAlign, size_t oldSize,
                                size_t newAlign, size_t newSize) {
  if (oldAlign != newAlign)
    return false;
  auto alignedSize = [](size_t alignment, size_t size) {
    return ((alignment - 1) | (size - 1)) + 1;
  };
  // Do not attempt reuse if data is likely managed via superslab.
  constexpr size_t GB = 1024 * 1024 * 1024;
  auto oldAlignedSize = alignedSize(oldAlign, oldSize);
  auto newAlignedSize = alignedSize(newAlign, newSize);
  if (oldAlignedSize >= GB || newAlignedSize >= GB)
    return false;
  auto oldExpMand = toExpMand(oldAlignedSize >> MIN_ALLOC_STEP_BITS);
  auto newExpMand = toExpMand(newAlignedSize >> MIN_ALLOC_STEP_BITS);
  return newExpMand == oldExpMand;
}
// heuristic < 0 : do not reuse at all
// heuristic == 0: can reuse via realloc
// heuristic > 0 : can reuse via ensure, larger is better
// TODO: consider appreantly non-exclusive cases.
int hueristic(TokenType producedType, mlir::TypedValue<RcType> producerRc,
              TokenAcceptor consumer, mlir::AliasAnalysis &aliasAnalyzer) {
  // Under perfect match, we measure the locality score.
  if (producedType == consumer.getTokenType()) {
    ReussirRcCreateOp create =
        dyn_cast<ReussirRcCreateOp>(consumer.getOperation());
    int localityScore = 1;
    // First, we do a very coarse grained copy avoidance analysis.
    if (producerRc && create &&
        mlir::isa<RecordType>(create.getRcPtr().getType().getElementType())) {
      // Check if any record field is assembled from a projection whose source
      // is aliased with the producer.
      mlir::ValueRange fields{};
      mlir::Operation *op = create.getRcPtr().getDefiningOp();
      while (auto variant =
                 llvm::dyn_cast_if_present<ReussirRecordVariantOp>(op))
        op = variant.getValue().getDefiningOp();
      if (auto compound =
              llvm::dyn_cast_if_present<ReussirRecordCompoundOp>(op))
        fields = compound.getFields();
      for (mlir::Value field : fields) {
        if (auto loaded = llvm::dyn_cast_if_present<ReussirRefLoadOp>(
                field.getDefiningOp())) {
          mlir::Value root = loaded.getRef();
          while (true) {
            if (auto projection =
                    llvm::dyn_cast<ReussirRefProjectOp>(root.getDefiningOp())) {
              root = projection.getRef();
              continue;
            }
            if (auto blkArg = llvm::dyn_cast<mlir::BlockArgument>(root)) {
              if (auto dispatch =
                      llvm::dyn_cast_if_present<ReussirRecordDispatchOp>(
                          blkArg.getOwner()->getParentOp())) {
                root = dispatch.getValue();
                continue;
              }
            }
            break;
          }
          if (auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
                  root.getDefiningOp()))
            localityScore +=
                aliasAnalyzer.alias(borrow.getRcPtr(), producerRc) ==
                mlir::AliasResult::MustAlias;
        }
      }
    }
    return localityScore;
  }
  TokenType consumerType = consumer.getTokenType();
  size_t oldSize = producedType.getSize();
  size_t newSize = consumerType.getSize();
  size_t oldAlign = producedType.getAlign();
  size_t newAlign = consumerType.getAlign();
  return possiblyInplaceReallocable(oldAlign, oldSize, newAlign, newSize) ? 0
                                                                          : -1;
}

// The (token type, producing rc value) pair behind an available token, for
// both producer shapes the matcher understands: a `reussir.rc.dec` that
// yields its box as a nullable token, and an *expanded* decrement (an
// `scf.if` tagged kExpandedDecrementAttr whose condition tests the fetched
// refcount of the decremented box).
struct ProducedToken {
  TokenType type{};
  mlir::TypedValue<RcType> rc{};
};
std::optional<ProducedToken> describeToken(mlir::Value tokenVal) {
  if (auto producer =
          dyn_cast_or_null<TokenProducer>(tokenVal.getDefiningOp())) {
    ReussirRcDecOp dec = dyn_cast<ReussirRcDecOp>(producer.getOperation());
    return ProducedToken{producer.getTokenType(),
                         dec ? dec.getRcPtr() : nullptr};
  }
  if (auto scfIf =
          dyn_cast_or_null<mlir::scf::IfOp>(tokenVal.getDefiningOp())) {
    if (!scfIf->hasAttr(kExpandedDecrementAttr))
      return std::nullopt;
    auto nullableType = dyn_cast<NullableType>(scfIf.getResult(0).getType());
    if (!nullableType)
      return std::nullopt;
    auto producedType = dyn_cast<TokenType>(nullableType.getPtrTy());
    if (!producedType)
      return std::nullopt;
    auto expectOp = dyn_cast_or_null<ReussirExpectOp>(
        scfIf.getCondition().getDefiningOp());
    if (!expectOp)
      return std::nullopt;
    auto cmp = dyn_cast_or_null<mlir::arith::CmpIOp>(
        expectOp.getCondition().getDefiningOp());
    if (!cmp)
      return std::nullopt;
    auto rcFetch = llvm::dyn_cast_if_present<ReussirRcFetchOp>(
        cmp.getLhs().getDefiningOp());
    return ProducedToken{producedType, rcFetch ? rcFetch.getRcPtr() : nullptr};
  }
  return std::nullopt;
}

// The nominal type domain of a producer/consumer: the rc element type. Two
// dead boxes of the same record are interchangeable in every respect; a
// same-domain reuse is exactly the in-place update Koka's reuse analysis
// aims for. Null when the shape is unknown.
mlir::Type producerDomain(const ProducedToken &produced) {
  return produced.rc ? produced.rc.getType().getElementType() : mlir::Type{};
}
mlir::Type acceptorDomain(TokenAcceptor acceptor) {
  if (auto create = dyn_cast<ReussirRcCreateOp>(acceptor.getOperation()))
    return create.getRcPtr().getType().getElementType();
  return {};
}

struct ValueHash {
  uint64_t operator()(mlir::Value v) const {
    void *ptr = v.getAsOpaquePointer();
    auto bytes = std::bit_cast<std::array<uint8_t, sizeof(void *)>>(ptr);
    return llvm::xxHash64(bytes);
  }
};
using ValueSet = immer::set<mlir::Value, ValueHash, std::equal_to<mlir::Value>>;

ValueSet intersect(const ValueSet &lhs, const ValueSet &rhs) {
  if (lhs.empty())
    return lhs;
  if (rhs.empty())
    return rhs;

  ValueSet res = lhs;
  for (auto val : lhs)
    if (!rhs.count(val))
      res = res.erase(val);
  return res;
}

unsigned
getDfsOrder(const mlir::DenseMap<mlir::Operation *, unsigned> &dfsOrder,
            mlir::Value token) {
  if (auto *defOp = token.getDefiningOp())
    return dfsOrder.lookup(defOp);
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(token))
    return dfsOrder.lookup(blockArg.getOwner()->getParentOp());
  return 0;
}

struct Reuse {
  mlir::Value token;
  bool realloc;
  TokenAcceptor anchor;
};
struct Free {
  mlir::Value token;
  mlir::Operation *anchor;
};
struct TokenReusePass : public impl::ReussirTokenReusePassBase<TokenReusePass> {
  using Base::Base;

  // Same-nominal-type bias strategy (the `domain-bias` option).
  enum class DomainBias { None, Lookahead, TwoPhase };
  // Which matches the current walk may form: everything admissible, or only
  // same-domain pairs (the first walk of the two-phase strategy).
  enum class MatchMode { Full, DomainOnly };

  DomainBias biasKind = DomainBias::None;
  MatchMode mode = MatchMode::Full;
  // Cross-walk state for the two-phase strategy: pairs formed in the
  // domain-only walk stay fixed and are *replayed* by the fallback walk when
  // it reaches each assigned acceptor. Consumption must stay path-local:
  // sibling branches of an scf.if each see the token as available (they are
  // mutually exclusive), so phase one may assign one token to an acceptor in
  // every sibling arm, and the replay erases it from each arm's own
  // availability copy. `reservedTokens` keeps the fallback matcher from
  // handing a phase-one token to an *earlier* unassigned acceptor, which
  // would double-consume it along the path to its reserved consumer.
  llvm::DenseMap<mlir::Operation *, mlir::Value> phaseOneAssignments;
  llvm::DenseSet<mlir::Value> reservedTokens;

  struct Candidate {
    mlir::Value token{};
    int score = -1;
    bool realloc = false;
    bool sameDomain = false;
  };

  // The best admissible token for `acceptor` under the current mode and bias,
  // ignoring `exclude`. Ranking: with a domain bias, same-domain first, then
  // the locality/size score, then recency (DFS order); without, the original
  // score-then-recency ranking.
  Candidate pickBest(const ValueSet &availableTokens, TokenAcceptor acceptor,
                     mlir::AliasAnalysis &aliasAnalyzer,
                     const mlir::DenseMap<mlir::Operation *, unsigned> &dfsOrder,
                     MatchMode mode, mlir::Value exclude) {
    mlir::Type consumerDomain = acceptorDomain(acceptor);
    Candidate best;
    for (auto tokenVal : availableTokens) {
      if (tokenVal == exclude)
        continue;
      // A token reserved by the domain-only walk belongs to its downstream
      // consumer; the fallback walk must not hand it out again.
      if (mode == MatchMode::Full && reservedTokens.contains(tokenVal))
        continue;
      auto produced = describeToken(tokenVal);
      if (!produced)
        continue;
      int score =
          hueristic(produced->type, produced->rc, acceptor, aliasAnalyzer);
      if (score < 0)
        continue;
      mlir::Type domain = producerDomain(*produced);
      bool sameDomain = domain && consumerDomain && domain == consumerDomain;
      if (mode == MatchMode::DomainOnly && !sameDomain)
        continue;
      bool better;
      if (!best.token) {
        better = true;
      } else if (biasKind == DomainBias::None) {
        better = score > best.score ||
                 (score == best.score &&
                  getDfsOrder(dfsOrder, tokenVal) >
                      getDfsOrder(dfsOrder, best.token));
      } else {
        better = std::tuple(sameDomain, score,
                            getDfsOrder(dfsOrder, tokenVal)) >
                 std::tuple(best.sameDomain, best.score,
                            getDfsOrder(dfsOrder, best.token));
      }
      if (better)
        best = Candidate{tokenVal, score, score == 0, sameDomain};
    }
    return best;
  }

  // Whether a later acceptor in the same block wants a token of `domain`.
  // Conservative and block-local: stops at anything that kills or may
  // conditionally consume the token (loops, region-bearing ops, and calls
  // when reuse does not cross calls) — mirroring the walk's own barriers.
  bool blockHasLaterSameDomainConsumer(mlir::Operation *from,
                                       mlir::Type domain) {
    for (mlir::Operation *op = from->getNextNode(); op;
         op = op->getNextNode()) {
      if (isa<mlir::LoopLikeOpInterface>(*op) || op->getNumRegions() > 0)
        return false;
      if (isa<mlir::CallOpInterface>(*op) && !reuseAcrossCall) {
        mlir::func::CallOp funcCall = llvm::dyn_cast<mlir::func::CallOp>(*op);
        if (!funcCall ||
            !funcCall.getCallee().starts_with("core::intrinsic::"))
          return false;
      }
      if (auto acceptor = dyn_cast<TokenAcceptor>(op)) {
        auto alloc = llvm::dyn_cast_if_present<ReussirTokenAllocOp>(
            acceptor.getToken().getDefiningOp());
        if (alloc && alloc.getToken().hasOneUse() &&
            !phaseOneAssignments.contains(op) &&
            acceptorDomain(acceptor) == domain)
          return true;
      }
    }
    return false;
  }

  ValueSet oneShotTokenReuse(
      mlir::Region &region, ValueSet availableTokens,
      llvm::SmallVectorImpl<Reuse> &reuses, llvm::SmallVectorImpl<Free> &frees,
      mlir::AliasAnalysis &aliasAnalyzer, mlir::DominanceInfo &domInfo,
      const mlir::DenseMap<mlir::Operation *, unsigned> &dfsOrder) {
    if (region.empty())
      return availableTokens;

    if (!region.hasOneBlock()) {
      region.getParentOp()->emitOpError()
          << "Token reuse pass only supports single block regions (SCF)";
      signalPassFailure();
      return {};
    }

    for (auto &op : region.front()) {
      if (isa<mlir::LoopLikeOpInterface>(op) ||
          (!reuseAcrossCall && isa<mlir::CallOpInterface>(op))) {
        mlir::func::CallOp funcCall = llvm::dyn_cast<mlir::func::CallOp>(op);
        // skip intrinsic calls
        if (!funcCall ||
            !funcCall.getCallee().starts_with("core::intrinsic::")) {
          for (auto token : availableTokens)
            frees.push_back({token, &op});
          availableTokens = {};
          for (auto &nestedRegion : op.getRegions())
            oneShotTokenReuse(nestedRegion, {}, reuses, frees, aliasAnalyzer,
                              domInfo, dfsOrder);
        }
      } else if (auto branchOp = dyn_cast<mlir::RegionBranchOpInterface>(op)) {
        llvm::SmallVector<ValueSet> branchResults;
        for (auto &nestedRegion : op.getRegions())
          branchResults.push_back(
              oneShotTokenReuse(nestedRegion, availableTokens, reuses, frees,
                                aliasAnalyzer, domInfo, dfsOrder));
        if (branchResults.empty()) {
          llvm::errs() << "[WARN] RegionBranch with no regions?\n";
        } else {
          // effective intersect with available token at parent
          // this rule out inner-scope created tokens from escaping parent
          // scope.
          ValueSet intersection = availableTokens;
          for (size_t i = 0; i < branchResults.size(); ++i)
            intersection = intersect(intersection, branchResults[i]);

          for (size_t i = 0; i < branchResults.size(); ++i) {
            for (auto val : branchResults[i]) {
              if (!intersection.count(val)) {
                mlir::Block &block = op.getRegion(i).front();
                frees.push_back({val, block.getTerminator()});
              }
            }
          }
          availableTokens = intersection;
        }
        if (auto scfIf = dyn_cast<mlir::scf::IfOp>(op))
          if (scfIf->hasAttr(kExpandedDecrementAttr) && scfIf->use_empty())
            availableTokens = availableTokens.insert(scfIf.getResult(0));
      }

      if (auto producer = dyn_cast<TokenProducer>(op)) {
        if (producer.shouldProduceToken()) {
          mlir::Value token = producer.getProducedValue();
          if (!token) {
            producer->emitOpError() << " should have produced a token before "
                                       "passing into reuse analysis";
            signalPassFailure();
            return {};
          }
          if (token.use_empty())
            availableTokens = availableTokens.insert(token);
        }
      }

      if (auto acceptor = dyn_cast<TokenAcceptor>(op)) {

        auto allocOp = llvm::dyn_cast_if_present<ReussirTokenAllocOp>(
            acceptor.getToken().getDefiningOp());
        auto preAssigned = phaseOneAssignments.find(acceptor.getOperation());
        if (allocOp && allocOp.getToken().hasOneUse() &&
            preAssigned != phaseOneAssignments.end() &&
            mode == MatchMode::Full) {
          // Replay a phase-one pair: consume the reserved token along this
          // path (the reuse itself was already recorded by phase one).
          availableTokens = availableTokens.erase(preAssigned->second);
        } else if (allocOp && allocOp.getToken().hasOneUse() &&
                   preAssigned == phaseOneAssignments.end()) {
          Candidate best =
              pickBest(availableTokens, acceptor, aliasAnalyzer, dfsOrder,
                       mode, /*exclude=*/{});

          // Lookahead bias: giving this token to a cross-domain consumer
          // starves any same-domain consumer further down the block, and a
          // same-domain reuse is strictly better (in-place update, no
          // realloc, hot cache line). Skip the assignment once and reselect
          // without the token; the downstream consumer picks it up when the
          // walk reaches it.
          if (best.token && !best.sameDomain &&
              biasKind == DomainBias::Lookahead) {
            mlir::Type tokenDomain;
            if (auto produced = describeToken(best.token))
              tokenDomain = producerDomain(*produced);
            if (tokenDomain && blockHasLaterSameDomainConsumer(&op, tokenDomain))
              best = pickBest(availableTokens, acceptor, aliasAnalyzer,
                              dfsOrder, mode, /*exclude=*/best.token);
          }

          if (best.token) {
            availableTokens = availableTokens.erase(best.token);
            reuses.push_back({best.token, best.realloc, acceptor});
            if (mode == MatchMode::DomainOnly) {
              phaseOneAssignments[acceptor.getOperation()] = best.token;
              reservedTokens.insert(best.token);
            }
          }
        }
        // ReussirClosureCreateOp is a kind of acceptor.
        if (auto closure = dyn_cast<ReussirClosureCreateOp>(op)) {
          if (closure.isInlined()) {
            closure->emitOpError()
                << " with inlined region found in token reuse pass";
            signalPassFailure();
            return {};
          }
        }
      }
    }

    mlir::Operation *terminator = region.front().getTerminator();
    // Collect tokens to free first, then erase them. Modifying the immer::set
    // during iteration would invalidate iterators (use-after-free).
    llvm::SmallVector<mlir::Value> tokensToFree;
    for (auto token : availableTokens) {
      if (!region.getParentOp() ||
          !domInfo.properlyDominates(token, region.getParentOp())) {
        frees.push_back({token, terminator});
        tokensToFree.push_back(token);
      }
    }
    for (auto token : tokensToFree)
      availableTokens = availableTokens.erase(token);
    return availableTokens;
  }

  void runOnOperation() override {
    if (domainBias != "none" && domainBias != "lookahead" &&
        domainBias != "two-phase") {
      getOperation()->emitError()
          << "invalid domain-bias '" << domainBias
          << "'; expected 'none', 'lookahead' or 'two-phase'";
      return signalPassFailure();
    }
    biasKind = domainBias == "lookahead"    ? DomainBias::Lookahead
               : domainBias == "two-phase" ? DomainBias::TwoPhase
                                           : DomainBias::None;
    mode = MatchMode::Full;
    phaseOneAssignments.clear();
    reservedTokens.clear();

    llvm::SmallVector<Reuse> reuses;
    llvm::SmallVector<Free> frees;
    mlir::AliasAnalysis aliasAnalyzer(getOperation());
    reussir::registerAliasAnalysisImplementations(aliasAnalyzer);
    mlir::DominanceInfo domInfo(getOperation());

    // Compute DFS pre-visit order for tiebreaking.
    mlir::DenseMap<mlir::Operation *, unsigned> dfsOrder;
    unsigned counter = 0;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation *op) { dfsOrder[op] = counter++; });

    // Two-phase bias: a first walk forms only same-domain pairs — the
    // matches worth protecting — so a cross-domain consumer earlier in
    // program order can no longer starve a same-domain consumer behind it.
    // The second (normal) walk matches what remains around the fixed pairs;
    // the first walk's frees are discarded, the fallback walk re-derives
    // the full free set with phase-one consumptions out of availability.
    if (biasKind == DomainBias::TwoPhase) {
      mode = MatchMode::DomainOnly;
      llvm::SmallVector<Free> discardedFrees;
      for (auto &region : getOperation()->getRegions())
        oneShotTokenReuse(region, {}, reuses, discardedFrees, aliasAnalyzer,
                          domInfo, dfsOrder);
      mode = MatchMode::Full;
    }

    for (auto &region : getOperation()->getRegions()) {
      oneShotTokenReuse(region, {}, reuses, frees, aliasAnalyzer, domInfo,
                        dfsOrder);
    }

    mlir::IRRewriter rewriter(getOperation());

    for (auto &reuse : reuses) {
      rewriter.setInsertionPoint(reuse.anchor);
      TokenType targetType = reuse.anchor.getTokenType();

      mlir::Value newToken;
      mlir::Value oldToken = reuse.anchor.getToken();
      if (reuse.realloc)
        newToken = ReussirTokenReallocOp::create(rewriter, 
            reuse.anchor->getLoc(), targetType, reuse.token);
      else
        newToken = ReussirTokenEnsureOp::create(rewriter, 
            reuse.anchor->getLoc(), targetType, reuse.token);
      reuse.anchor.assignToken(newToken);
      auto allocOp = llvm::cast<ReussirTokenAllocOp>(oldToken.getDefiningOp());
      rewriter.eraseOp(allocOp);
    }

    // A free of an *expanded decrement*'s token pays for a nullable nobody
    // consumes: the unique (rc==1) branch wraps the reinterpreted pointer
    // only so a later `token.free` can test it for null again — and on the
    // common shared branch the token is null, so the whole dance produces a
    // runtime call that does nothing. When the decrement's result is used by
    // nothing but frees (a consumed token would carry an ensure/realloc use
    // by now — reuses were rewritten above), sink one statically non-null
    // free into the unique branch instead: the shared path stops touching
    // the allocator entirely and the nullable result dies as a dead phi.
    // Sibling-exit records of the same token collapse into the single sunk
    // free — the producer dominates every path the records covered.
    llvm::DenseSet<mlir::Value> sunkTokens;
    for (const auto &free : frees) {
      auto scfIf = llvm::dyn_cast_if_present<mlir::scf::IfOp>(
          free.token.getDefiningOp());
      if (scfIf && scfIf->hasAttr(kExpandedDecrementAttr) &&
          free.token.use_empty()) {
        if (!sunkTokens.insert(free.token).second)
          continue; // already sunk via an earlier record of this token
        mlir::Operation *thenYield =
            scfIf.getThenRegion().back().getTerminator();
        auto nonNull = llvm::dyn_cast_if_present<ReussirNullableCreateOp>(
            thenYield->getOperand(0).getDefiningOp());
        if (nonNull && nonNull.getPtr()) {
          rewriter.setInsertionPoint(thenYield);
          ReussirTokenFreeOp::create(rewriter, thenYield->getLoc(),
                                     nonNull.getPtr());
          continue;
        }
        // Unexpected branch shape: fall back to the guarded top-level free.
        sunkTokens.erase(free.token);
      }
      rewriter.setInsertionPoint(free.anchor);
      ReussirTokenFreeOp::create(rewriter, free.anchor->getLoc(), free.token);
    }
  }
};
} // namespace
} // namespace reussir
