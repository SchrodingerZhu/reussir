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
/// This file implements the Reussir default inliner pass: MLIR's SCC inliner
/// driven directly through `mlir::Inliner` with a programmatic
/// `InlinerConfig`, rather than through `createInlinerPass` +
/// `initializeOptions` string parsing. The textual route resolves its
/// `default-pipeline` through the global pass registry — forcing the driver
/// to register upstream passes it otherwise never names — and re-parses the
/// policy from strings on every construction; building the config in C++
/// keeps the policy typed, registry-free, and in one place.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Transformation/Passes.h"

#include <mlir/Analysis/CallGraph.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Inliner.h>
#include <mlir/Transforms/Passes.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRDEFAULTINLINERPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

// The callable optimizer between inlining rounds: canonicalize with region
// simplification disabled, matching the rest of the pipeline's
// canonicalization policy.
void addCanonicalizerWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addPass(mlir::createCanonicalizerPass(config));
}

// Whether the callable region holds at most `limit` operations, counted
// recursively through nested regions. The walk stops as soon as the limit
// is crossed, so a query costs O(limit) however large the callee is.
bool calleeIsSmall(mlir::Region *callable, unsigned limit) {
  if (!callable)
    return false;
  unsigned count = 0;
  auto result = callable->walk([&](mlir::Operation *) {
    return ++count > limit ? mlir::WalkResult::interrupt()
                           : mlir::WalkResult::advance();
  });
  return !result.wasInterrupted();
}

struct DefaultInlinerPass
    : public impl::ReussirDefaultInlinerPassBase<DefaultInlinerPass> {
  using Base::Base;
  void runOnOperation() override {
    // A single inlining iteration: within a recursive SCC the inliner
    // re-inlines the call sites each round just created, so more iterations
    // grow a mutually recursive call graph geometrically in its fanout (see
    // the pass description). One round still inlines every non-recursive
    // call bottom-up across SCCs.
    const mlir::InlinerConfig config(addCanonicalizerWithoutRegionSimplification,
                                     /*maxInliningIterations=*/1);
    mlir::CallGraph &callGraph = getAnalysis<mlir::CallGraph>();
    // Nested (callable-optimizer) pipelines run through the pass's own
    // instrumented pipeline runner. Profitability caps the callee's size:
    // bottom-up processing means an uncapped policy re-copies each level's
    // fully-inlined bodies into every caller, growing the module
    // geometrically in call-tree depth (see the pass description).
    mlir::Inliner inliner(
        getOperation(), callGraph, *this, getAnalysisManager(),
        [this](mlir::Pass &, mlir::OpPassManager &pipeline,
               mlir::Operation *op) { return runPipeline(pipeline, op); },
        config, [this](const mlir::Inliner::ResolvedCall &call) {
          return calleeIsSmall(call.targetNode->getCallableRegion(),
                               maxCalleeOps);
        });
    if (mlir::failed(inliner.doInlining()))
      signalPassFailure();
  }
};

} // namespace
} // namespace reussir
