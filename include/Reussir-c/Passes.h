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
/// C API for assembling the Reussir lowering pipeline from Rust. Each Reussir
/// (and required upstream) pass is exposed as an MlirPass factory so the Rust
/// backend can build the pipeline step by step with melior's pass manager, plus
/// the standalone helpers the pipeline depends on.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_PASSES_H
#define REUSSIR_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Reussir passes
//===----------------------------------------------------------------------===//

MlirPass reussirCreateUniqueCarryingRecursionAnalysisPass(void);
MlirPass reussirCreateTokenInstantiationPass(void);
MlirPass reussirCreateClosureOutliningPass(void);
MlirPass reussirCreateRegionPatternsPass(void);
MlirPass reussirCreateIncDecCancellationPass(void);
MlirPass reussirCreateRcDecrementExpansionPass(void);
MlirPass reussirCreateInferVariantTagPass(void);
MlirPass reussirCreateConvertToSTDPass(void);
// Beta-reduces closure apply/eval chains (inlines fully visible closures,
// collapses chained uniqueness checks). Aggressive-opt only: inlining closure
// bodies hurts debuggability.
MlirPass reussirCreateClosureBetaReductionPass(void);
MlirPass reussirCreateRcCreateSinkPass(void);

/// Encodes nullary variants of shared rc-boxed enums as tagged pointer
/// immediates (top byte = tag + 1) and stamps the module so the LLVM
/// lowering steers refcount stores away from the dummy boxes. Add only when
/// the scheme is enabled; `archIndependent` selects the encoding: false =
/// `tbi` (top-byte tag, requires hardware top-byte-ignore, aarch64), true =
/// `immortal` (plain dummy address with an immortal refcount, any target).
MlirPass reussirCreateSpecialPointerTagPass(bool archIndependent);

/// Attaches a stderr per-pass progress logger to `pm`: every pass execution
/// prints `[mlir-pass] begin/end <pass> on <op>` lines, so a wedged or
/// pathological pass can be located from a captured log alone (the last
/// `begin` without an `end` names it). Intended for verbose/diagnostic runs.
void reussirPassManagerAttachPhaseLogger(MlirPassManager pm);
MlirPass reussirCreateRcDispatchFusionPass(void);
MlirPass reussirCreatePartialMovePass(void);
MlirPass reussirCreateRcCreateFusionPass(void);
MlirPass reussirCreateTRMCRecursionAnalysisPass(void);
MlirPass reussirCreateCompilePolymorphicFFIPass(bool optimized);
MlirPass reussirCreateInvariantGroupAnalysisPass(void);
MlirPass reussirCreateBasicOpsLoweringPass(bool closureWpd);
MlirPass reussirCreateDebugInfoConversionPass(void);

// Acquire/drop expansion has two phases controlled by these options; the
// pipeline runs it once with both disabled and once with both enabled.
MlirPass reussirCreateAcquireDropExpansionPass(bool expandDecrement,
                                               bool outlineRecord);
// Token reuse can optionally reuse tokens across function calls and emit one
// structured optimization remark for every allocation/reuse decision.
MlirPass reussirCreateTokenReusePass(bool reuseAcrossCall, bool emitRemarks);

//===----------------------------------------------------------------------===//
// Upstream passes used by the pipeline
//===----------------------------------------------------------------------===//

// The default inliner configured with a canonicalizer that does not simplify
// regions, matching the C++ backend pipeline.
MlirPass reussirCreateDefaultInlinerPass(void);

// The upstream transform-dialect interpreter, applying the transform named
// sequence `entryPoint` to the payload module. The entry point is resolved
// first in the payload module itself, then in the context's transform
// library (populated by the preload pass below); the pass fails if it is
// found in neither. The pipeline runs one interpreter per anchor, each with
// the anchor's `__reussir_anchor_<name>` entry point.
MlirPass reussirCreateTransformInterpreterPass(MlirStringRef entryPoint);

// The upstream transform-library preload pass: parses each of the `nPaths`
// script files in `paths` and merges its transform named sequences into the
// context-wide transform library consulted by the interpreter. Merging
// reports a symbol clash, so distinct scripts must use distinct entry
// points (one script per anchor).
MlirPass reussirCreateTransformPreloadLibraryPass(MlirStringRef const *paths,
                                                  intptr_t nPaths);
MlirPass reussirCreateCanonicalizerPass(void);
MlirPass reussirCreateCSEPass(void);
MlirPass reussirCreateControlFlowSinkPass(void);
MlirPass reussirCreateSCFToControlFlowPass(void);
MlirPass reussirCreateConvertToLLVMPass(void);
MlirPass reussirCreateReconcileUnrealizedCastsPass(void);

//===----------------------------------------------------------------------===//
// Standalone helpers
//===----------------------------------------------------------------------===//

// Installs a context-wide token-reuse remark streamer that writes a
// deterministic, location-deduplicated JSON report to `outputPath` when the
// context is destroyed. Returns false when the path cannot be opened or a
// remark engine is already installed on the context.
bool reussirContextEnableTokenReuseRemarks(MlirContext context,
                                           MlirStringRef outputPath);

// Monomorphizes and compiles polymorphic FFI operations in the module. Returns
// true on success. `rustPath` and the `libDirs` array (of `nLibDirs` entries)
// name the rustc executable and the Rust package search directories
// explicitly; a non-empty `targetTriple` is passed to `rustc --target`. Pass
// an empty string ref / an empty array to fall back to the REUSSIR_RUSTC /
// REUSSIR_RUSTC_DEPS environment variables and the built-in probe list.
bool reussirCompilePolymorphicFFI(MlirModule module, bool optimized,
                                  MlirStringRef rustPath,
                                  const MlirStringRef *libDirs,
                                  intptr_t nLibDirs,
                                  MlirStringRef targetTriple);

// Gathers the LLVM bitcode modules attached to compiled operations into a
// single LLVM module owned by `context`. Returns NULL on failure; otherwise the
// caller owns the returned module.
//
// `dataLayout` and `targetTriple` describe the machine the gathered module is
// destined for and are stamped on it. A non-empty `targetTriple` also settles
// the spelling: rustc-produced bitcode may name the same target differently
// (`wasm32-wasip1-threads` arrives as `wasm32-unknown-wasi`), which the IR
// linker reports as a mismatch. Pass an empty string to leave it as parsed.
LLVMModuleRef reussirGatherCompiledModules(MlirModule module,
                                           LLVMContextRef context,
                                           const char *dataLayout,
                                           const char *targetTriple);

// Rewrites the `{ tag, payload-union }` debug type emitted for each enum into a
// real DWARF `DW_TAG_variant_part`, so a debugger shows only the active case.
// Operates in place on the (already debug-info-bearing) LLVM module; a no-op
// when there is no debug info.
void reussirFixupVariantDebugInfo(LLVMModuleRef module);

// On COFF targets, attaches a `comdat any` (keyed by the symbol's own name, as
// COFF requires) to every function and global variable defined with
// weak-for-linker linkage (`weak`/`weak_odr`/`linkonce`/`linkonce_odr`) that
// does not already carry one. COFF has no weak symbol binding: without a
// COMDAT section LLVM lowers such definitions to the fragile
// `.weak.<sym>.default` weak-external fallback and identical definitions in
// two objects fail the link. Run it on the final llvm::Module, after every
// definition (monomorphized instances, drop/acquire glue, linked-in FFI
// bitcode) exists. A no-op for every other object format — Mach-O rejects
// comdats outright, ELF weak binding dedups without them.
void reussirAttachCoffComdats(LLVMModuleRef module);

// Reports whether TPDE support was compiled into the backend.
int reussirHasTPDE(void);

// Stamps `module` with the target's layout facts before the lowering pipeline
// runs: the LLVM data layout string (`llvm.data_layout`), the target triple
// (`llvm.target_triple`), and the translated DLTI spec (`dlti.dl_spec`) that
// MLIR `DataLayout` queries read. Without the spec MLIR falls back to its
// conservative defaults (e.g. `i64` at ABI alignment 4), so every
// size/alignment the pipeline computes — allocation sizes, spill alignments,
// load/store alignment annotations — understates the target. Returns false if
// `dataLayout` does not parse.
bool reussirModuleAttachTargetSpec(MlirModule module, const char *dataLayout,
                                   const char *triple);

// Sets whether compound record members are laid out in packed physical order
// (descending storage alignment) rather than declaration order. On by default.
// This is a whole-compilation layout contract held on the context-loaded
// Reussir dialect, so set it once before the lowering pipeline computes any
// layout. Loads the Reussir dialect into `context` if not already loaded.
void reussirContextSetPackRecordMembers(MlirContext context, bool enable);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_PASSES_H
