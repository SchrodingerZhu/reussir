//! Raw FFI bindings to the Reussir C API.
//!
//! This crate exposes the C entry points provided by `libReussirCAPI` together
//! with a re-export of [`mlir_sys`], the raw MLIR C API the bindings are built
//! on. Safe wrappers live in the `reussir-backend` crate; everything here is
//! `unsafe` and mirrors the C signatures one-to-one.

#![allow(non_snake_case)]

pub use mlir_sys;

use core::ffi::c_int;

use mlir_sys::{MlirContext, MlirDialectHandle, MlirDialectRegistry, MlirModule, MlirPass};

unsafe extern "C" {
    //==-- Dialect registration --==//

    /// Returns the dialect handle for the Reussir dialect. The handle can be
    /// inserted into a dialect registry or registered into a context.
    pub fn mlirGetDialectHandle__reussir__() -> MlirDialectHandle;

    /// Registers the Reussir dialect together with every upstream dialect,
    /// extension and LLVM/builtin translation it relies on into `context`, then
    /// loads all available dialects.
    pub fn reussirRegisterAllDialects(context: MlirContext);

    /// Populates a dialect registry with the Reussir dialect and everything the
    /// backend pipeline depends on. Build a context from the registry so dialect
    /// extensions are applied as dialects load.
    pub fn reussirPopulateRegistry(registry: MlirDialectRegistry);

    //==-- Reussir passes --==//

    pub fn reussirCreateUniqueCarryingRecursionAnalysisPass() -> MlirPass;
    pub fn reussirCreateTokenInstantiationPass() -> MlirPass;
    pub fn reussirCreateClosureOutliningPass() -> MlirPass;
    pub fn reussirCreateRegionPatternsPass() -> MlirPass;
    pub fn reussirCreateIncDecCancellationPass() -> MlirPass;
    pub fn reussirCreateRcDecrementExpansionPass() -> MlirPass;
    pub fn reussirCreateInferVariantTagPass() -> MlirPass;
    pub fn reussirCreateSCFOpsLoweringPass() -> MlirPass;
    pub fn reussirCreateRcCreateSinkPass() -> MlirPass;
    pub fn reussirCreateRcCreateFusionPass() -> MlirPass;
    pub fn reussirCreateTRMCRecursionAnalysisPass() -> MlirPass;
    pub fn reussirCreateCompilePolymorphicFFIPass(optimized: bool) -> MlirPass;
    pub fn reussirCreateInvariantGroupAnalysisPass() -> MlirPass;
    pub fn reussirCreateBasicOpsLoweringPass() -> MlirPass;
    pub fn reussirCreateAcquireDropExpansionPass(
        expand_decrement: bool,
        outline_record: bool,
    ) -> MlirPass;
    pub fn reussirCreateTokenReusePass(reuse_across_call: bool) -> MlirPass;

    //==-- Upstream passes used by the pipeline --==//

    pub fn reussirCreateDefaultInlinerPass() -> MlirPass;
    pub fn reussirCreateCanonicalizerPass() -> MlirPass;
    pub fn reussirCreateCSEPass() -> MlirPass;
    pub fn reussirCreateControlFlowSinkPass() -> MlirPass;
    pub fn reussirCreateSCFToControlFlowPass() -> MlirPass;
    pub fn reussirCreateConvertToLLVMPass() -> MlirPass;
    pub fn reussirCreateReconcileUnrealizedCastsPass() -> MlirPass;

    //==-- Standalone helpers --==//

    /// Monomorphizes and compiles polymorphic FFI operations in the module.
    /// Returns true on success.
    pub fn reussirCompilePolymorphicFFI(module: MlirModule, optimized: bool) -> bool;

    /// Reports whether TPDE support was compiled into the backend.
    pub fn reussirHasTPDE() -> c_int;
}
