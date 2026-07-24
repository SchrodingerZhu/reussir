//! The LLVM-IR boundary shared by the JIT (`reussir-jit`) and the AOT compiler
//! (`reussir-compiler`): translating a lowered MLIR module to LLVM IR, running
//! the backend LLVM pass pipeline, and the polymorphic-FFI gather/link.
//!
//! Polymorphic FFI straddles the MLIR lowering pipeline. The `reussir.polyffi`
//! templates are compiled to LLVM bitcode and gathered *before* the pipeline
//! erases them, but the gathered module is linked into the main module *after*
//! it is translated to LLVM IR. [`LlvmLowering`] is the handle carried across
//! the caller's [`run_lowering_pipeline`](crate::pipeline::run_lowering_pipeline)
//! call to bracket those two halves; it also owns the `LLVMContext` both modules
//! live in, so they can be linked.
//!
//! These helpers operate on raw `llvm-sys` handles (and pass them to the Reussir
//! CAPI, whose mirror types are pointer-cast across) so callers keep driving the
//! ORC session / target-machine emission through `llvm-sys` directly. All LLVM
//! handles here resolve to the one statically linked LLVM, so the casts are
//! sound.

use std::ffi::CString;

use llvm_sys::core::{LLVMContextCreate, LLVMContextDispose, LLVMDisposeModule, LLVMSetDataLayout};
use llvm_sys::linker::LLVMLinkModules2;
use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef};

use melior::ir::Module;

use reussir_backend_sys as sys;

use crate::pipeline::OptLevel;

/// Explicit locations for the polymorphic-FFI texture compile. Empty fields
/// fall back to the `REUSSIR_RUSTC` / `REUSSIR_RUSTC_DEPS` environment
/// variables and the built-in probe list (see `RustCompiler.cpp`).
#[derive(Clone, Debug, Default)]
pub struct PolyffiPaths {
    /// The `rustc` executable used to compile textures.
    pub rust_path: Option<String>,
    /// The directories searched for Rust packages (one `rustc -L` each) —
    /// `libreussir_rt` and friends.
    pub libdirs: Vec<String>,
}

// Views an optional string as the borrowed MlirStringRef the C API takes; the
// referent must outlive every use of the result. `None` becomes an empty ref,
// which the C side treats as "fall back to discovery".
fn opt_string_ref(s: &Option<String>) -> sys::mlir_sys::MlirStringRef {
    let s = s.as_deref().unwrap_or("");
    sys::mlir_sys::MlirStringRef {
        data: s.as_ptr().cast(),
        length: s.len(),
    }
}

// Views a slice of strings as borrowed MlirStringRefs; the referents must
// outlive every use of the result. An empty slice becomes an empty array,
// which the C side treats as "fall back to discovery".
fn string_refs(strings: &[String]) -> Vec<sys::mlir_sys::MlirStringRef> {
    strings
        .iter()
        .map(|s| sys::mlir_sys::MlirStringRef {
            data: s.as_ptr().cast(),
            length: s.len(),
        })
        .collect()
}

/// Translates a lowered MLIR `module` to an LLVM IR module created in `context`.
///
/// The returned module is owned alongside `context`. Returns an error if MLIR
/// translation fails.
///
/// # Safety
/// `context` must be a valid `LLVMContextRef`, and `module` must already be
/// lowered to the LLVM dialect.
pub unsafe fn translate_to_llvm_ir(
    module: &Module,
    context: LLVMContextRef,
) -> Result<LLVMModuleRef, String> {
    unsafe {
        let operation = mlir_sys::mlirModuleGetOperation(module.to_raw());
        let llvm_module =
            mlir_sys::mlirTranslateModuleToLLVMIR(operation, context as mlir_sys::LLVMContextRef)
                as LLVMModuleRef;
        if llvm_module.is_null() {
            return Err("failed to translate MLIR module to LLVM IR".into());
        }
        Ok(llvm_module)
    }
}

/// Runs the Reussir backend LLVM pass pipeline on `module` in place (a no-op for
/// [`OptLevel::None`]/[`OptLevel::Tpde`], matching the C++ backend).
///
/// `machine` supplies the target model the cost-driven passes (loop/SLP
/// vectorization) work against; without one they would run on the base
/// TargetTransformInfo, which has no vector registers, and never fire. Pass
/// null to have the pipeline build a host TargetMachine internally (the JIT
/// case).
///
/// # Safety
/// `module` must be a valid `LLVMModuleRef`, and `machine` a valid
/// `LLVMTargetMachineRef` or null.
pub unsafe fn run_backend_llvm_pipeline(
    module: LLVMModuleRef,
    opt: OptLevel,
    machine: llvm_sys::target_machine::LLVMTargetMachineRef,
) {
    unsafe {
        sys::reussirRunBackendLLVMPipeline(
            module as sys::LLVMModuleRef,
            opt.as_reussir_opt_option(),
            machine as sys::LLVMTargetMachineRef,
        );
    }
}

/// Polymorphic-FFI lowering bracketing the MLIR lowering pipeline.
///
/// Construct it with [`prepare`](Self::prepare) before running the lowering
/// pipeline, then [`finish`](Self::finish) afterwards. It owns the `LLVMContext`
/// the gathered and (eventually) main modules share.
pub struct LlvmLowering {
    context: LLVMContextRef,
    /// The gathered FFI bitcode (an empty module when there is no polyffi).
    /// Null after [`finish`](Self::finish) consumes it via the linker.
    gathered: LLVMModuleRef,
    data_layout: CString,
}

impl LlvmLowering {
    /// Phase 1 — run *before* the MLIR lowering pipeline. Creates the LLVM
    /// context, compiles the module's `reussir.polyffi` templates, and gathers
    /// their bitcode (an empty module when there is none, so the scalar path is
    /// unaffected). `data_layout` stamps the gathered module — and later the main
    /// module — so the eventual link is layout-consistent.
    ///
    /// Gathering erases the polyffi ops, so the lowering pipeline's own
    /// `CompilePolymorphicFFIPass` then sees nothing to do.
    ///
    /// `paths` pins the rustc executable and package directories the texture
    /// compile uses; default it to keep the environment/probe discovery.
    pub fn prepare(
        module: &Module,
        data_layout: &str,
        optimized: bool,
        paths: &PolyffiPaths,
    ) -> Result<Self, String> {
        let data_layout =
            CString::new(data_layout).map_err(|_| "data layout contains a NUL byte".to_string())?;
        let libdirs = string_refs(&paths.libdirs);
        unsafe {
            if !sys::reussirCompilePolymorphicFFI(
                module.to_raw(),
                optimized,
                opt_string_ref(&paths.rust_path),
                libdirs.as_ptr(),
                libdirs.len() as isize,
            ) {
                return Err("failed to compile polymorphic FFI".into());
            }
            let context = LLVMContextCreate();
            let gathered = sys::reussirGatherCompiledModules(
                module.to_raw(),
                context as sys::LLVMContextRef,
                data_layout.as_ptr(),
            ) as LLVMModuleRef;
            if gathered.is_null() {
                LLVMContextDispose(context);
                return Err("failed to gather compiled polymorphic-FFI modules".into());
            }
            Ok(Self {
                context,
                gathered,
                data_layout,
            })
        }
    }

    /// Phase 3 — run *after* the MLIR lowering pipeline. Translates the lowered
    /// `module` into the prepared context, stamps the shared data layout, and
    /// links the gathered FFI bitcode into it. Returns the finalized LLVM IR
    /// module (and the context that owns it) for the caller to optimize and
    /// either JIT or emit.
    pub fn finish(mut self, module: &Module) -> Result<Finalized, String> {
        unsafe {
            let main = translate_to_llvm_ir(module, self.context)?;
            LLVMSetDataLayout(main, self.data_layout.as_ptr());
            // Promote each enum's `{ tag, payload-union }` debug type to a real
            // DWARF `DW_TAG_variant_part` now that we are at the LLVM-DI level,
            // where the discriminator operand exists (MLIR's attribute has none).
            sys::reussirFixupVariantDebugInfo(main as sys::LLVMModuleRef);
            // `LLVMLinkModules2` consumes (disposes) the source module, even on
            // failure — so null our handle either way to keep `Drop` correct.
            let gathered = std::mem::replace(&mut self.gathered, std::ptr::null_mut());
            if LLVMLinkModules2(main, gathered) != 0 {
                LLVMDisposeModule(main);
                return Err("failed to link polymorphic-FFI modules into the main module".into());
            }
            // Hand the context off to `Finalized`; null it so our `Drop` is inert.
            let context = std::mem::replace(&mut self.context, std::ptr::null_mut());
            Ok(Finalized {
                context,
                module: main,
            })
        }
    }
}

impl Drop for LlvmLowering {
    fn drop(&mut self) {
        // Only disposes anything when `finish` was *not* called (e.g. an error
        // between `prepare` and `finish`); `finish` nulls both handles.
        unsafe {
            if !self.gathered.is_null() {
                LLVMDisposeModule(self.gathered);
            }
            if !self.context.is_null() {
                LLVMContextDispose(self.context);
            }
        }
    }
}

/// A finalized LLVM IR module and the `LLVMContext` that owns it. The caller is
/// responsible for the handles: run [`run_backend_llvm_pipeline`] over `module`,
/// then either hand `module`/`context` to the ORC JIT or emit `module` and
/// dispose both.
pub struct Finalized {
    pub context: LLVMContextRef,
    pub module: LLVMModuleRef,
}
