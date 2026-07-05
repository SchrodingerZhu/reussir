//! The Reussir lowering pipeline.
//!
//! Assembles the sequence of Reussir and upstream passes that lowers a Reussir
//! module down to the LLVM dialect, mirroring the backend's C++ pipeline. Each
//! pass is created through the Reussir C API and added to a melior pass manager,
//! with `func.func`-nested passes placed in their original order so the
//! interleaving with module-level passes is preserved.

use melior::Context;
use melior::ir::Module;
use melior::pass::{Pass, PassManager};

use reussir_backend_sys as sys;

/// Optimization level for the lowering pipeline, matching the backend's
/// `ReussirOptOption` C enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization; the optimization-only prologue passes are skipped.
    None,
    /// Default optimization.
    Default,
    /// Aggressive optimization; also runs the unique-carrying recursion analysis.
    Aggressive,
    /// Optimize for size.
    Size,
    /// Lower for the TPDE fast back end.
    Tpde,
}

impl OptLevel {
    /// Whether the optimization-only prologue (inliner and friends) runs.
    fn runs_optimization(self) -> bool {
        !matches!(self, OptLevel::None)
    }

    /// This level as its matching `ReussirOptOption` C enum value — the contract
    /// the LLVM-IR-level backend pipeline (`reussirRunBackendLLVMPipeline`)
    /// expects. Owned here so callers (the JIT, the AOT compiler) need not
    /// re-encode it.
    pub fn as_reussir_opt_option(self) -> core::ffi::c_int {
        match self {
            OptLevel::None => 0,
            OptLevel::Default => 1,
            OptLevel::Aggressive => 2,
            OptLevel::Size => 3,
            OptLevel::Tpde => 4,
        }
    }
}

/// How nullary variants of shared rc-boxed enums are represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullaryVariantEncoding {
    /// Legacy layout: every construction heap-allocates a refcounted box.
    Boxed,
    /// Unboxed immediate whose top byte is `tag + 1` and whose low bits
    /// point at a per-tag dummy box. Hard-depends on the hardware ignoring
    /// the pointer's top byte on data accesses (aarch64 TBI); the top byte
    /// doubles as a dereference-free FFI tag decode.
    Tbi,
    /// Unboxed immediate that *is* the per-tag dummy box address, with an
    /// immortal refcount recognized by magnitude. No architectural
    /// dependency (works on any target, including wasm32); foreign code
    /// sees a layout-compatible box.
    Immortal,
}

/// Knobs for [`run_lowering_pipeline`].
#[derive(Clone, Copy, Debug)]
pub struct LoweringOptions {
    /// Optimization level.
    pub opt: OptLevel,
    /// Allow the token-reuse pass to reuse tokens across function calls.
    pub reuse_token_across_call: bool,
    /// How nullary variants of shared rc-boxed enums are encoded. `rrc`
    /// exposes this as `--nullary-variant-encoding`; embedders (REPL/JIT,
    /// tests) default to [`NullaryVariantEncoding::Boxed`].
    pub nullary_variant_encoding: NullaryVariantEncoding,
    /// Run the invariant-group analysis pass.
    pub enable_invariant_analysis: bool,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            opt: OptLevel::Default,
            reuse_token_across_call: false,
            // Boxed by default: only target-aware drivers (rrc) pick an
            // immediate encoding, so embedders (REPL/JIT, tests) keep the
            // boxed layout untouched.
            nullary_variant_encoding: NullaryVariantEncoding::Boxed,
            // Off by default, mirroring the C++ `createLoweringPipeline`
            // (`enableInvariantAnalysis = false`).
            enable_invariant_analysis: false,
        }
    }
}

/// Stamps `module` with the target's layout facts — the `llvm.data_layout`
/// string, the `llvm.target_triple`, and the translated `dlti.dl_spec` that
/// MLIR `DataLayout` queries read. Call it before [`run_lowering_pipeline`]:
/// without the spec MLIR falls back to conservative defaults (e.g. `i64` at
/// ABI alignment 4), so every size and alignment the pipeline computes —
/// allocation sizes, spill alignments, load/store alignment annotations —
/// understates the target.
pub fn attach_target_spec(module: &Module, data_layout: &str, triple: &str) -> Result<(), String> {
    let data_layout = std::ffi::CString::new(data_layout)
        .map_err(|_| "data layout contains a NUL byte".to_string())?;
    let triple = std::ffi::CString::new(triple)
        .map_err(|_| "target triple contains a NUL byte".to_string())?;
    // SAFETY: `module` is live for the call, and both strings are
    // NUL-terminated; the CAPI only reads them to build attributes.
    let ok = unsafe {
        sys::reussirModuleAttachTargetSpec(module.to_raw(), data_layout.as_ptr(), triple.as_ptr())
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid data layout string `{}`",
            data_layout.to_string_lossy()
        ))
    }
}

// Wraps a Reussir C API pass factory result as a melior pass.
fn pass(raw: sys::mlir_sys::MlirPass) -> Pass {
    // SAFETY: the C API returns a freshly created, owned MlirPass.
    unsafe { Pass::from_raw(raw) }
}

/// A small DSL describing the lowering pipeline as a declarative list of steps,
/// so the layout reads top-to-bottom instead of being buried in builder calls.
///
/// Each step is one of:
/// * `module: <factory>;` — a module-level pass;
/// * `func: <factory>;` — a pass nested under `func.func`;
/// * `if <cond> => { <steps> }` — a group run only when `<cond>` holds.
///
/// `<factory>` is a Reussir C API pass factory (an `unsafe` call); the macro
/// wraps each one and hands the resulting [`Pass`] to the pass manager. The
/// first argument names the [`PassManager`] to build into.
macro_rules! lowering_pipeline {
    // Done.
    ($pm:ident) => {};
    ($pm:ident,) => {};
    // Module-level pass.
    ($pm:ident, module: $factory:expr; $($rest:tt)*) => {{
        // SAFETY: each factory returns a freshly created, owned MlirPass.
        $pm.add_pass(pass(unsafe { $factory }));
        lowering_pipeline!($pm, $($rest)*);
    }};
    // Pass nested under `func.func`.
    ($pm:ident, func: $factory:expr; $($rest:tt)*) => {{
        // SAFETY: each factory returns a freshly created, owned MlirPass.
        $pm.nested_under("func.func").add_pass(pass(unsafe { $factory }));
        lowering_pipeline!($pm, $($rest)*);
    }};
    // Group of steps gated on a condition.
    ($pm:ident, if $cond:expr => { $($body:tt)* } $($rest:tt)*) => {{
        if $cond {
            lowering_pipeline!($pm, $($body)*);
        }
        lowering_pipeline!($pm, $($rest)*);
    }};
}

/// Runs the full Reussir lowering pipeline on `module`, leaving it in the LLVM
/// dialect ready for translation to LLVM IR.
pub fn run_lowering_pipeline(
    context: &Context,
    module: &mut Module,
    options: &LoweringOptions,
) -> Result<(), melior::Error> {
    let _span = tracing::debug_span!(
        "reussir_lowering",
        opt = ?options.opt,
        reuse_token_across_call = options.reuse_token_across_call,
        enable_invariant_analysis = options.enable_invariant_analysis,
        nullary_variant_encoding = ?options.nullary_variant_encoding,
    )
    .entered();

    let manager = PassManager::new(context);

    lowering_pipeline!(manager,
        // Nullary variants of shared rc-boxed enums become tagged pointer
        // immediates. Must run before any uniqueness/token pass so no
        // provenance or allocation token is ever attached to a rewritten
        // construction.
        if options.nullary_variant_encoding != NullaryVariantEncoding::Boxed => {
            module: sys::reussirCreateSpecialPointerTagPass(
                options.nullary_variant_encoding == NullaryVariantEncoding::Immortal,
            );
        }

        // Optimization-only prologue.
        if options.opt.runs_optimization() => {
            if options.opt == OptLevel::Aggressive => {
                module: sys::reussirCreateUniqueCarryingRecursionAnalysisPass();
            }
            module: sys::reussirCreateDefaultInlinerPass();
        }

        // Reussir-level transformation and analysis.
        func:   sys::reussirCreateTokenInstantiationPass();
        module: sys::reussirCreateClosureOutliningPass();
        module: sys::reussirCreateRegionPatternsPass();
        // Fuse pattern-match consumption into destructuring decrements before
        // the cancellation pass (which cancels `inc; destructuring dec` into
        // borrow semantics) and the decrement expansion (which expands the
        // tagged decs shallowly).
        func:   sys::reussirCreateRcDispatchFusionPass();
        func:   sys::reussirCreateIncDecCancellationPass();
        module: sys::reussirCreateRcDecrementExpansionPass();
        func:   sys::reussirCreateInferVariantTagPass();
        module: sys::reussirCreateAcquireDropExpansionPass(false, false);
        module: sys::reussirCreateSCFOpsLoweringPass();
        func:   sys::reussirCreateIncDecCancellationPass();

        // Second acquire/drop expansion phase: expand decrements and outline
        // record drops.
        module: sys::reussirCreateAcquireDropExpansionPass(true, true);
        func:   sys::reussirCreateTokenReusePass(options.reuse_token_across_call);
        module: sys::reussirCreateSCFOpsLoweringPass();
        func:   sys::reussirCreateRcCreateSinkPass();
        func:   sys::reussirCreateRcCreateFusionPass();
        module: sys::reussirCreateTRMCRecursionAnalysisPass();
        module: sys::reussirCreateCompilePolymorphicFFIPass(false);

        if options.enable_invariant_analysis => {
            func: sys::reussirCreateInvariantGroupAnalysisPass();
        }

        // Lower to the LLVM dialect.
        module: sys::reussirCreateCanonicalizerPass();
        module: sys::reussirCreateControlFlowSinkPass();
        module: sys::reussirCreateSCFToControlFlowPass();
        module: sys::reussirCreateBasicOpsLoweringPass();
        module: sys::reussirCreateConvertToLLVMPass();
        module: sys::reussirCreateReconcileUnrealizedCastsPass();
        // Convert fused Reussir debug-info attributes to LLVM DI now that
        // functions are `llvm.func`; a function's DI only survives translation
        // once its `llvm.func` carries a `DISubprogram`.
        module: sys::reussirCreateDebugInfoConversionPass();
        module: sys::reussirCreateCSEPass();
        module: sys::reussirCreateCanonicalizerPass();
    );

    tracing::trace!("running Reussir lowering pipeline");
    let result = manager.run(module);
    match &result {
        Ok(()) => tracing::debug!("lowered module to the LLVM dialect"),
        Err(error) => tracing::error!(?error, "lowering pipeline failed"),
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::operation::OperationLike;

    #[test]
    fn lowers_simple_module_to_llvm_dialect() {
        let context = crate::context();

        let source = r#"
            module {
              func.func @answer() -> i32 {
                %0 = arith.constant 42 : i32
                func.return %0 : i32
              }
            }
        "#;
        let mut module = Module::parse(&context, source).expect("module should parse");

        run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
            .expect("pipeline should succeed");

        assert!(module.as_operation().verify());
        // After lowering, the function is an llvm.func rather than a func.func.
        let rendered = module.as_operation().to_string();
        assert!(rendered.contains("llvm.func @answer"), "got:\n{rendered}");
    }
}
