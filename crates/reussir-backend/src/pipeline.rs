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
}

/// Knobs for [`run_lowering_pipeline`].
#[derive(Clone, Copy, Debug)]
pub struct LoweringOptions {
    /// Optimization level.
    pub opt: OptLevel,
    /// Allow the token-reuse pass to reuse tokens across function calls.
    pub reuse_token_across_call: bool,
    /// Run the invariant-group analysis pass.
    pub enable_invariant_analysis: bool,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            opt: OptLevel::Default,
            reuse_token_across_call: false,
            // Off by default, mirroring the C++ `createLoweringPipeline`
            // (`enableInvariantAnalysis = false`).
            enable_invariant_analysis: false,
        }
    }
}

// Wraps a Reussir C API pass factory result as a melior pass.
fn pass(raw: sys::mlir_sys::MlirPass) -> Pass {
    // SAFETY: the C API returns a freshly created, owned MlirPass.
    unsafe { Pass::from_raw(raw) }
}

/// Runs the full Reussir lowering pipeline on `module`, leaving it in the LLVM
/// dialect ready for translation to LLVM IR.
pub fn run_lowering_pipeline(
    context: &Context,
    module: &mut Module,
    options: &LoweringOptions,
) -> Result<(), melior::Error> {
    let manager = PassManager::new(context);

    // SAFETY: every factory returns an owned MlirPass handed to the manager.
    unsafe {
        if options.opt.runs_optimization() {
            if options.opt == OptLevel::Aggressive {
                manager.add_pass(pass(sys::reussirCreateUniqueCarryingRecursionAnalysisPass()));
            }
            manager.add_pass(pass(sys::reussirCreateDefaultInlinerPass()));
        }

        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateTokenInstantiationPass()));
        manager.add_pass(pass(sys::reussirCreateClosureOutliningPass()));
        manager.add_pass(pass(sys::reussirCreateRegionPatternsPass()));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateIncDecCancellationPass()));
        manager.add_pass(pass(sys::reussirCreateRcDecrementExpansionPass()));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateInferVariantTagPass()));
        manager.add_pass(pass(sys::reussirCreateAcquireDropExpansionPass(
            false, false,
        )));
        manager.add_pass(pass(sys::reussirCreateSCFOpsLoweringPass()));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateIncDecCancellationPass()));

        // Second acquire/drop expansion phase: expand decrements and outline
        // record drops.
        manager.add_pass(pass(sys::reussirCreateAcquireDropExpansionPass(true, true)));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateTokenReusePass(
                options.reuse_token_across_call,
            )));
        manager.add_pass(pass(sys::reussirCreateSCFOpsLoweringPass()));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateRcCreateSinkPass()));
        manager
            .nested_under("func.func")
            .add_pass(pass(sys::reussirCreateRcCreateFusionPass()));
        manager.add_pass(pass(sys::reussirCreateTRMCRecursionAnalysisPass()));
        manager.add_pass(pass(sys::reussirCreateCompilePolymorphicFFIPass(false)));

        if options.enable_invariant_analysis {
            manager
                .nested_under("func.func")
                .add_pass(pass(sys::reussirCreateInvariantGroupAnalysisPass()));
        }

        // Lower to the LLVM dialect.
        manager.add_pass(pass(sys::reussirCreateCanonicalizerPass()));
        manager.add_pass(pass(sys::reussirCreateControlFlowSinkPass()));
        manager.add_pass(pass(sys::reussirCreateSCFToControlFlowPass()));
        manager.add_pass(pass(sys::reussirCreateBasicOpsLoweringPass()));
        manager.add_pass(pass(sys::reussirCreateConvertToLLVMPass()));
        manager.add_pass(pass(sys::reussirCreateReconcileUnrealizedCastsPass()));
        manager.add_pass(pass(sys::reussirCreateCSEPass()));
        manager.add_pass(pass(sys::reussirCreateCanonicalizerPass()));
    }

    manager.run(module)
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
