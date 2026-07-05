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
    `reussir-unique-carrying-recursion-analysis` uses sparse forward dataflow
    with interprocedural propagation to mark RC-returning function and SCF
    operations carrying uniqueness. When a carrying function is directly
    self-recursive, the pass creates an internal `.unique` clone that asserts
    RC argument uniqueness at entry and rewrites recursive calls to use the
    specialized clone.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

def ReussirTRMCRecursionAnalysisPass
    : Pass<"reussir-trmc-recursion-analysis", "::mlir::ModuleOp"> {
  let summary = "rewrite fused tail-recursive modulo cons patterns to hole-based helpers";
  let description = [{
    `reussir-trmc-recursion-analysis` recognizes directly self-recursive
    functions whose recursive result is embedded in a fused
    `reussir.rc.create_compound` or `reussir.rc.create_variant` site.
    It outlines an internal `.trmc` helper with an explicit `reussir.hole`
    out-parameter and rewrites the recursive site to allocate the outer RC box
    first, expose pending-field holes, and redirect the recursive fields
    through the helper. When a fused create site contains multiple direct
    self-calls, the pass forwards all of them through the generated holes while
    still falling back to the original function for non-linear recursive sites.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::ub::UBDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

#endif // REUSSIR_TRANSFORMATION_PASSES_TD
