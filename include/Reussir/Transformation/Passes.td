//===-- Passes.td - Transformation pass definition file ----*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the transformation and analysis passes for the Reussir
// dialect.
//
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
    the dummy boxes on taggable types. Three encodings exist: `tbi` (top byte
    = tag + 1; hard-depends on hardware top-byte-ignore, aarch64), `immortal`
    (plain dummy address with an immortal refcount; no architectural
    dependency), and `taggedbox` (PoC Koka/Lean-style low-bit immediate
    `(tag << 1) | 1`; no dummy box, rc ops guard on the pointer's low bit).
  }];
  let options = [Option<"encoding", "encoding", "std::string",
                        /*default=*/"\"tbi\"",
                        "Immediate encoding: 'tbi', 'immortal', or "
                        "'taggedbox'">];
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

  let options = [
    Option<"reuseAcrossCall", "reuse-across-call", "bool", /*default=*/"false",
        "allow token reuse across function calls (may increase peak heap usage)">,
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
  let summary = "analyze uniqueness-carrying recursion and outline unique clones";
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
  let summary = "tail recursion modulo constructors via first-class constructor contexts";
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
                           "::mlir::scf::SCFDialect",
                           "::mlir::ub::UBDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

#endif // REUSSIR_TRANSFORMATION_PASSES_TD
