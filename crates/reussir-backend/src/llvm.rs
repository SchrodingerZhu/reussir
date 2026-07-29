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
    /// The machine target passed to the texture's `rustc --target`. `None`
    /// leaves rustc on its native host, matching an omitted rrc
    /// `--target-triple`.
    pub target_triple: Option<String>,
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

/// How a compilation is packaged for link-time optimization — the Rust face
/// of the C `ReussirLtoMode`.
///
/// It selects which module pipeline runs: the per-module one, or an LTO
/// pre-link pipeline that leaves the whole-program work to the link step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LtoMode {
    /// No LTO: the full per-module pipeline, native code out.
    #[default]
    None,
    /// ThinLTO: pre-link pipeline, bitcode carrying a module summary index.
    Thin,
    /// Full ("fat") LTO: pre-link pipeline, plain bitcode.
    Fat,
}

impl LtoMode {
    /// The `ReussirLtoMode` discriminant the C API expects.
    pub fn as_c_int(self) -> std::ffi::c_int {
        match self {
            LtoMode::None => 0,
            LtoMode::Thin => 1,
            LtoMode::Fat => 2,
        }
    }

    pub fn is_lto(self) -> bool {
        self != LtoMode::None
    }
}

/// Runs the Reussir backend LLVM pass pipeline on `module` in place (a no-op for
/// [`OptLevel::None`]/[`OptLevel::Tpde`], matching the C++ backend).
///
/// `machine` supplies the target model the cost-driven passes (loop/SLP
/// vectorization) work against; without one they would run on the base
/// TargetTransformInfo, which has no vector registers, and never fire. Pass
/// null to have the pipeline build a host TargetMachine internally (the JIT
/// case). `lto` picks the per-module or the matching pre-link pipeline.
///
/// # Safety
/// `module` must be a valid `LLVMModuleRef`, and `machine` a valid
/// `LLVMTargetMachineRef` or null.
pub unsafe fn run_backend_llvm_pipeline(
    module: LLVMModuleRef,
    opt: OptLevel,
    machine: llvm_sys::target_machine::LLVMTargetMachineRef,
    lto: LtoMode,
) {
    unsafe {
        sys::reussirRunBackendLLVMPipeline(
            module as sys::LLVMModuleRef,
            opt.as_reussir_opt_option(),
            machine as sys::LLVMTargetMachineRef,
            lto.as_c_int(),
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
                opt_string_ref(&paths.target_triple),
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
            // Every ODR definition now exists in one module (codegen'd
            // instances, the dialect's drop/acquire glue, linked FFI bitcode),
            // so this is the one seam where the COFF comdat invariant can be
            // enforced for all of them: COFF has no weak symbol binding, and a
            // weak-for-linker definition without a comdat lowers to the
            // fragile `.weak.<sym>.default` fallback that fails the link when
            // two objects carry the same instance. No-op off COFF (Mach-O
            // rejects comdats; ELF weak binding needs none).
            sys::reussirAttachCoffComdats(main as sys::LLVMModuleRef);
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

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use llvm_sys::LLVMLinkage;
    use llvm_sys::core::{
        LLVMAddFunction, LLVMAppendBasicBlockInContext, LLVMBuildRetVoid, LLVMContextCreate,
        LLVMContextDispose, LLVMCreateBuilderInContext, LLVMDisposeBuilder, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMFunctionType, LLVMModuleCreateWithNameInContext,
        LLVMPositionBuilderAtEnd, LLVMPrintModuleToString, LLVMSetLinkage, LLVMSetTarget,
        LLVMVoidTypeInContext,
    };

    use reussir_backend_sys as sys;

    /// Builds a module for `triple` holding one defined function per linkage,
    /// runs the comdat attachment over it, and returns the printed IR.
    unsafe fn attach_and_print(triple: &CStr, linkages: &[(&CStr, LLVMLinkage)]) -> String {
        unsafe {
            let context = LLVMContextCreate();
            // Not named after the feature: the assertions substring-match the
            // printed IR for `comdat`, and the module name is printed too.
            let module = LLVMModuleCreateWithNameInContext(c"seam-test".as_ptr(), context);
            LLVMSetTarget(module, triple.as_ptr());
            let void = LLVMVoidTypeInContext(context);
            let fn_ty = LLVMFunctionType(void, std::ptr::null_mut(), 0, 0);
            let builder = LLVMCreateBuilderInContext(context);
            for &(name, linkage) in linkages {
                let f = LLVMAddFunction(module, name.as_ptr(), fn_ty);
                LLVMSetLinkage(f, linkage);
                // Comdats are only legal on definitions, so give each a body.
                let bb = LLVMAppendBasicBlockInContext(context, f, c"entry".as_ptr());
                LLVMPositionBuilderAtEnd(builder, bb);
                LLVMBuildRetVoid(builder);
            }
            LLVMDisposeBuilder(builder);
            sys::reussirAttachCoffComdats(module as sys::LLVMModuleRef);
            let printed = LLVMPrintModuleToString(module);
            let ir = CStr::from_ptr(printed).to_string_lossy().into_owned();
            LLVMDisposeMessage(printed);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
            ir
        }
    }

    const WEAK_FOR_LINKER: [(&CStr, LLVMLinkage); 4] = [
        (c"wo", LLVMLinkage::LLVMWeakODRLinkage),
        (c"lo", LLVMLinkage::LLVMLinkOnceODRLinkage),
        (c"wa", LLVMLinkage::LLVMWeakAnyLinkage),
        (c"la", LLVMLinkage::LLVMLinkOnceAnyLinkage),
    ];

    #[test]
    fn coff_weak_for_linker_definitions_get_comdat_any() {
        let mut cases = WEAK_FOR_LINKER.to_vec();
        cases.push((c"strong", LLVMLinkage::LLVMExternalLinkage));
        cases.push((c"local", LLVMLinkage::LLVMInternalLinkage));
        let ir = unsafe { attach_and_print(c"x86_64-pc-windows-msvc", &cases) };
        for (name, _) in &WEAK_FOR_LINKER {
            let name = name.to_str().unwrap();
            // The comdat is keyed by the symbol's own name (a COFF
            // requirement), which LLVM prints as a bare `comdat` token.
            assert!(
                ir.contains(&format!("${name} = comdat any")),
                "missing comdat for {name}: {ir}"
            );
            assert!(
                ir.contains(&format!("@{name}() comdat")),
                "definition of {name} not attached to its comdat: {ir}"
            );
        }
        // Strong and internal definitions are not ODR-mergeable; comdat'ing
        // them would change their link semantics.
        assert!(!ir.contains("$strong"), "strong external comdat'd: {ir}");
        assert!(!ir.contains("$local"), "internal comdat'd: {ir}");
    }

    #[test]
    fn non_coff_formats_stay_comdat_free() {
        // Mach-O rejects comdats outright (hard LLVM error at emission); ELF
        // binds weak symbols natively and must stay byte-identical.
        for triple in [c"aarch64-apple-darwin", c"x86_64-unknown-linux-gnu"] {
            let ir = unsafe { attach_and_print(triple, &WEAK_FOR_LINKER) };
            assert!(
                !ir.contains("comdat"),
                "comdat leaked into {}: {ir}",
                triple.to_string_lossy()
            );
        }
    }
}
