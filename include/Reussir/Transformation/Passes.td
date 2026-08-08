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
/// This file defines the transformation and analysis passes for the Reussir
/// dialect.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_TRANSFORMATION_PASSES_TD
#define REUSSIR_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SpecialPointerTag
//===----------------------------------------------------------------------===//
def ReussirSpecialPointerTagPass
    : Pass<"reussir-special-pointer-tag", "::mlir::ModuleOp"> {
  let summary = "encode nullary variants as tagged pointer immediates";
  let description = [{
    `reussir-special-pointer-tag` rewrites nullary variant constructions of
    shared rc-boxed enums into `reussir.rc.tagged` immediates pointing at
    per-tag dummy boxes (no allocation) and stamps the module with the
    chosen encoding so the LLVM lowering steers refcount stores away from
    the dummy boxes on taggable types. Two encodings exist: `tbi` (top byte
    = tag + 1; hard-depends on hardware top-byte-ignore, aarch64) and
    `immortal` (plain dummy address with an immortal refcount; no
    architectural dependency).
  }];
  let options = [Option<"encoding", "encoding", "std::string",
                        /*default=*/"\"tbi\"",
                        "Immediate encoding: 'tbi' or 'immortal'">];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// ClosureBetaReduction
//===----------------------------------------------------------------------===//
def ReussirClosureBetaReductionPass
    : Pass<"reussir-closure-beta-reduction", "::mlir::func::FuncOp"> {
  let summary = "beta-reduce closure apply/eval chains";
  let description = [{
    `reussir-closure-beta-reduction` walks each `closure.eval`'s operand up
    the def-use chain, folding single-use `closure.uniqify`/`closure.apply`
    links into the eval's fused `with` pack. A chain bottoming at a
    single-use inlined `closure.create` is a visible beta redex: the body is
    spliced in place of the eval with the pack substituted for its
    parameters, and every intermediate operation (chain, create, dead token)
    is erased. Any other root leaves the fused eval, which the SCF-ops
    lowering expands as ONE uniqueness resolution + unchecked applies +
    plain eval — the intermediate uniqueness checks of the folded chain
    collapse into one. The same reduction applies when the closure is carried
    through a nest containing only LoopLike operations and `cell.rmw`.

    Runs after the inliner (so cross-function create→eval pairs are visible)
    and before token instantiation / closure outlining (bodies are still
    inline regions). Intended for aggressive optimization only: inlining
    closure bodies detaches the code from the closure abstraction the
    programmer wrote, which hurts debuggability.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// RcCreateSink
//===----------------------------------------------------------------------===//
def ReussirRcCreateSinkPass
    : Pass<"reussir-rc-create-sink", "::mlir::func::FuncOp"> {
  let summary = "sink rc.create operations";
  let description = [{
    `reussir-rc-create-sink` sinks `reussir.rc.create` operations into ensure
    scf for better copy avoidance.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// RcCreateFusion
//===----------------------------------------------------------------------===//
def ReussirRcCreateFusionPass
    : Pass<"reussir-rc-create-fusion", "::mlir::func::FuncOp"> {
  let summary = "fuse record materialization into rc.create operations";
  let description = [{
    `reussir-rc-create-fusion` fuses `reussir.record.variant` and
    `reussir.record.compound` producers into dedicated RC creation operations.
    For `variant(compound(...))`, it performs a total fusion into
    `reussir.rc.create_variant`.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// PartialMove
//===----------------------------------------------------------------------===//
def ReussirEarlyPartialMovePass
    : Pass<"reussir-early-partial-move", "::mlir::func::FuncOp"> {
  let summary = "schedule fresh owned constructions before inlining";
  let description = [{
    `reussir-early-partial-move` recognizes a closed fresh-owned construction
    placed immediately after a call despite all inputs being available before
    it. If the chain consumes a retained projected owner, it moves the
    compound, variant, and allocation before the call. This changes placement
    rather than RC volume and recovers the ordering available to
    hand-scheduled source.

    This pass deliberately omits complete-owner scheduling and adjacent
    retain/decrement fusion. Those rewrites remain in
    `reussir-partial-move`, after dispatch fusion has exposed their canonical
    late forms.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

def ReussirPartialMovePass
    : Pass<"reussir-partial-move", "::mlir::func::FuncOp"> {
  let summary = "recognize and materialize partial moves";
  let description = [{
    `reussir-partial-move` recognizes a projection-only compound owner whose
    last projection appears after an earlier extracted member is consumed.
    When all nontrivial top-level members transfer, it gathers the canonical
    extraction chains at the first consuming use and advances the root
    decrement to that ownership cut. It rejects control flow and any crossed
    count observation that may alias the root. Complete transfer is required
    because an unbound member's early drop may invoke observable cleanup glue.

    For shared boxed owners, the already-adjacent retain/decrement pattern is
    the degenerate case. Both forms materialize as a tagless destructuring
    decrement whose `boundMembers` transfer on the unique path and
    rematerialize their retains on the shared path.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// RcDispatchFusion
//===----------------------------------------------------------------------===//
def ReussirRcDispatchFusionPass
    : Pass<"reussir-rc-dispatch-fusion", "::mlir::func::FuncOp"> {
  let summary = "fuse pattern-match consumption into destructuring decrements";
  let description = [{
    `reussir-rc-dispatch-fusion` recognizes a borrowing `record.dispatch`
    whose arm retains its bound members and then decrements the scrutinee,
    erases the bound retains, and tags the decrement with the arm's tag and
    bound member indices (Koka's `dropn_reuse` shape). The decrement
    expansion then releases the box *shallowly*: on the unique path bound
    members transfer with the arm and only unbound rc members are released;
    on the shared path the bound members are retained. A preceding increment
    cancels against the destructuring decrement in
    `reussir-inc-dec-cancellation` by rematerializing the bound retains.

  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// InferVariantTag
//===----------------------------------------------------------------------===//
def ReussirInferVariantTagPass
    : Pass<"reussir-infer-variant-tag", "::mlir::func::FuncOp"> {
  let summary = "infer variant tags for drop operations";
  let description = [{
    `reussir-infer-variant-tag` infers variant tags for drop operations.
  }];

  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// IncDecCancellation
//===----------------------------------------------------------------------===//
def ReussirIncDecCancellationPass
    : Pass<"reussir-inc-dec-cancellation", "::mlir::func::FuncOp"> {
  let summary = "cancel out adjacent increment and decrement operations";
  let description = [{
    `reussir-inc-dec-cancellation` cancels out adjacent increment and decrement
    operations.
  }];

  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// TokenReuse
//===----------------------------------------------------------------------===//
def ReussirTokenReusePass
    : Pass<"reussir-token-reuse", "::mlir::func::FuncOp"> {
  let summary = "reuse tokens for operations";
  let description = [{
    `reussir-token-reuse` reuses tokens for operations.
  }];

  let options = [Option<"reuseAcrossCall", "reuse-across-call", "bool",
                        /*default=*/"false",
                        "allow token reuse across function calls (may increase "
                        "peak heap usage)">,
  ];

  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// InvariantGroupAnalysis
//===----------------------------------------------------------------------===//
def ReussirInvariantGroupAnalysisPass
    : Pass<"reussir-invariant-group-analysis", "::mlir::func::FuncOp"> {
  let summary = "mark ref.load operations with invariant_group attribute";
  let description = [{
    `reussir-invariant-group-analysis` uses sparse forward dataflow analysis
    to determine which `ref.load` operations can be marked with invariant_group.
    A load is marked when its ref operand provably originates from `rc.borrow`
    or `closure.inspect_payload` through a chain of `ref.project` operations,
    and no projection in the chain goes through a flex reference's mutable field.
  }];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

def ReussirUniqueCarryingRecursionAnalysisPass
    : Pass<"reussir-unique-carrying-recursion-analysis", "::mlir::ModuleOp"> {
  let summary =
      "analyze uniqueness-carrying recursion and outline unique clones";
  let description = [{
    `reussir-unique-carrying-recursion-analysis` computes, per rc-typed
    function result, whether the returned value is on every path either
    fresh (an `rc.create*` with an initialized count, never shared
    afterwards) or one of the function's own rc arguments passed through
    unshared — a Kleene iteration over demand-driven provenance summaries.
    Functions and structured-control-flow ops whose rc results all qualify
    are marked `reussir.carrying_uniqueness`. When a carrying function is
    directly self-recursive and a self call's carried arguments are provably
    unique, the pass creates internal `.unique` clones (marked
    `reussir.unique_clone`) that assert argument uniqueness at entry and
    rewrites the provable self calls to use them, one clone per provable
    assumption set. Provenance is traced only through region ops whose
    results are exhaustively yield-defined (`scf.if`, `scf.index_switch`,
    `scf.execute_region`, `reussir.record.dispatch`,
    `reussir.array.with_unique_view`); values with retaining or multiple
    consuming users are disqualified — see the pass source header for the
    soundness argument.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

def ReussirTRMCRecursionAnalysisPass
    : Pass<"reussir-trmc-recursion-analysis", "::mlir::ModuleOp"> {
  let summary =
      "tail recursion modulo constructors via first-class constructor contexts";
  let description = [{
    `reussir-trmc-recursion-analysis` splits every directly self-recursive
    function returning an rc-boxed value whose recursion feeds a fused
    `reussir.rc.create_compound`/`reussir.rc.create_variant` field into a thin
    wrapper and an internal `.trmc` helper threading a `!reussir.cctx`
    accumulator. Constructor leaves allocate with holes, `cctx.extend` the
    context, and tail-call the helper (a loop after lowering); plain self tail
    calls thread the context through; every other leaf completes the context
    with `cctx.apply`. Multiple recursive fields in one create recurse through
    fresh single-node contexts, with the last field carrying the loop.
    Because every return leaf has a sound rewrite, mixed functions — where
    some arms pass the recursive result through another call first — keep
    full coverage of their constructor arms.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect", "::mlir::ub::UBDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

def ReussirDefaultInlinerPass
    : Pass<"reussir-default-inliner", "::mlir::ModuleOp"> {
  let summary = "the pipeline's inliner: one iteration, canonicalize between";
  let description = [{
    `reussir-default-inliner` runs MLIR's SCC inliner with the pipeline's
    fixed policy, configured programmatically (no textual pass options, no
    pass-registry resolution): a single inlining iteration, and a callable
    optimizer that canonicalizes with region simplification disabled.

    One iteration is the load-bearing part: within a recursive SCC the
    inliner re-inlines the call sites each round just created, so the
    upstream default of four iterations grows a mutually recursive call
    graph geometrically in its fanout — an interpreter-shaped program
    inflates a few thousand ops into millions. A single iteration keeps
    ordinary bottom-up inlining across SCCs (every non-recursive call still
    inlines) and unrolls recursion one level, the posture LLVM's own inliner
    takes. Region simplification stays off in the callable optimizer to
    match the rest of the pipeline's canonicalization policy.
  }];
}

#endif // REUSSIR_TRANSFORMATION_PASSES_TD
