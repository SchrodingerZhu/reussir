//! Ahead-of-time compilation: a lowered MLIR module → an object file, assembly,
//! or LLVM IR on disk.
//!
//! This is the AOT counterpart to `reussir-jit`: instead of adding the
//! translated LLVM IR to an ORC JIT session, it runs the backend LLVM pass
//! pipeline and emits a file with a host (or cross) `TargetMachine`. It links neither the
//! JIT engine nor the Reussir runtime — an emitted object is linked against
//! `reussir-rt` later by the C toolchain. The `reussir-compiler` binary drives
//! the whole frontend → lowering → [`compile_to_file`] path.

use std::ffi::{CString, c_char, c_int};
use std::path::Path;
use std::sync::Once;

use llvm_sys::core::{
    LLVMContextCreate, LLVMContextDispose, LLVMDisposeMessage, LLVMDisposeModule,
    LLVMPrintModuleToString, LLVMSetTarget,
};
use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef};
use llvm_sys::target::{
    LLVM_InitializeNativeAsmParser, LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget,
    LLVMDisposeTargetData, LLVMSetModuleDataLayout,
};
use llvm_sys::target_machine::{
    LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout,
    LLVMCreateTargetMachine, LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple,
    LLVMGetHostCPUFeatures, LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
    LLVMTargetMachineEmitToFile, LLVMTargetRef,
};

use melior::ir::Module;

use reussir_backend::pipeline::OptLevel;

// LLVM-side codegen helper from libReussirCAPI: the custom Reussir LLVM passes,
// the same pipeline the JIT runs.
unsafe extern "C" {
    fn reussirRunBackendLLVMPipeline(module: LLVMModuleRef, opt: c_int);
}

/// What `reussir-compiler` writes out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    /// A relocatable object file (`.o`).
    Object,
    /// Target assembly (`.s`).
    Assembly,
    /// LLVM IR text (`.ll`).
    LlvmIr,
}

/// Position-independence of the emitted code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RelocMode {
    /// The target default.
    #[default]
    Default,
    /// Position-independent code.
    Pic,
    /// Non-PIC.
    Static,
}

/// Which machine to emit for. A `None` field falls back to the native host —
/// except that, for a custom `triple`, `cpu`/`features` fall back to *empty*
/// (LLVM's architecture defaults) rather than the host's: feeding the host CPU
/// or feature string to a foreign triple makes LLVM crash.
#[derive(Clone, Debug, Default)]
pub struct TargetSpec {
    /// Target triple; `None` = the native host triple.
    pub triple: Option<String>,
    /// Target CPU; `None` = the native host CPU (generic for a custom triple).
    pub cpu: Option<String>,
    /// Target features; `None` = the native host features (none for a custom triple).
    pub features: Option<String>,
}

fn init_native() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        // A non-zero return means the target is unavailable; codegen then fails
        // loudly at `LLVMGetTargetFromTriple`, so there is nothing to recover here.
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();
        LLVM_InitializeNativeAsmParser();
    });
}

fn codegen_level(opt: OptLevel) -> LLVMCodeGenOptLevel {
    match opt {
        OptLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
        OptLevel::Size | OptLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
        OptLevel::Aggressive | OptLevel::Tpde => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
    }
}

fn reloc(mode: RelocMode) -> LLVMRelocMode {
    match mode {
        RelocMode::Default => LLVMRelocMode::LLVMRelocDefault,
        RelocMode::Pic => LLVMRelocMode::LLVMRelocPIC,
        RelocMode::Static => LLVMRelocMode::LLVMRelocStatic,
    }
}

unsafe fn c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Takes ownership of an LLVM-allocated C string (e.g. from
/// `LLVMGetDefaultTargetTriple`), copying it into an owned [`CString`] and
/// freeing the original with `LLVMDisposeMessage`. A null pointer yields an
/// empty string.
unsafe fn take_llvm_string(ptr: *mut c_char) -> CString {
    if ptr.is_null() {
        return CString::default();
    }
    unsafe {
        let owned = std::ffi::CStr::from_ptr(ptr).to_owned();
        LLVMDisposeMessage(ptr);
        owned
    }
}

/// Compile a lowered (LLVM-dialect) MLIR `module` to `out_path`.
///
/// The module must already have been through
/// [`crate::pipeline`](reussir_backend::pipeline)-equivalent lowering to the LLVM
/// dialect; this translates it to LLVM IR, runs the backend LLVM pipeline, and
/// emits the requested artifact for `target` (the native host by default).
pub fn compile_to_file(
    module: &Module,
    opt: OptLevel,
    kind: OutputKind,
    reloc_mode: RelocMode,
    target: &TargetSpec,
    out_path: &Path,
) -> Result<(), String> {
    init_native();
    unsafe {
        let context: LLVMContextRef = LLVMContextCreate();
        let operation = mlir_sys::mlirModuleGetOperation(module.to_raw());
        let llvm_module =
            mlir_sys::mlirTranslateModuleToLLVMIR(operation, context as mlir_sys::LLVMContextRef)
                as LLVMModuleRef;
        if llvm_module.is_null() {
            LLVMContextDispose(context);
            return Err("failed to translate MLIR module to LLVM IR".into());
        }

        // The custom Reussir LLVM passes + standard optimization.
        reussirRunBackendLLVMPipeline(llvm_module, opt.as_reussir_opt_option());

        let result = emit(llvm_module, opt, kind, reloc_mode, target, out_path);

        LLVMDisposeModule(llvm_module);
        LLVMContextDispose(context);
        result
    }
}

unsafe fn emit(
    llvm_module: LLVMModuleRef,
    opt: OptLevel,
    kind: OutputKind,
    reloc_mode: RelocMode,
    target_spec: &TargetSpec,
    out_path: &Path,
) -> Result<(), String> {
    unsafe {
        // LLVM IR text needs no target machine.
        if kind == OutputKind::LlvmIr {
            let s = LLVMPrintModuleToString(llvm_module);
            let text = c_str(s);
            LLVMDisposeMessage(s);
            return std::fs::write(out_path, text)
                .map_err(|e| format!("failed to write {}: {e}", out_path.display()));
        }

        // Resolve the triple/CPU/features into owned C strings. A custom triple
        // makes CPU/features default to empty (LLVM picks architecture defaults)
        // rather than the host's, which would crash for a foreign triple.
        let foreign = target_spec.triple.is_some();
        let triple = match target_spec.triple.as_deref() {
            Some(t) => {
                CString::new(t).map_err(|_| "target triple contains a NUL byte".to_string())?
            }
            None => take_llvm_string(LLVMGetDefaultTargetTriple()),
        };
        let cpu = match target_spec.cpu.as_deref() {
            Some(c) => CString::new(c).map_err(|_| "target CPU contains a NUL byte".to_string())?,
            None if foreign => CString::default(),
            None => take_llvm_string(LLVMGetHostCPUName()),
        };
        let features = match target_spec.features.as_deref() {
            Some(f) => {
                CString::new(f).map_err(|_| "target features contain a NUL byte".to_string())?
            }
            None if foreign => CString::default(),
            None => take_llvm_string(LLVMGetHostCPUFeatures()),
        };

        let mut target: LLVMTargetRef = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) != 0 {
            let msg = c_str(err);
            LLVMDisposeMessage(err);
            return Err(format!(
                "no target for triple `{}`: {msg}",
                triple.to_string_lossy()
            ));
        }
        let machine = LLVMCreateTargetMachine(
            target,
            triple.as_ptr(),
            cpu.as_ptr(),
            features.as_ptr(),
            codegen_level(opt),
            reloc(reloc_mode),
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        // Stamp the module with the target triple and data layout so the emitted
        // object's ABI matches the target (and so `%cc` can link it).
        LLVMSetTarget(llvm_module, triple.as_ptr());
        let data_layout = LLVMCreateTargetDataLayout(machine);
        // `LLVMSetModuleDataLayout` copies the layout into the module, so the
        // target-data handle is ours to dispose.
        LLVMSetModuleDataLayout(llvm_module, data_layout);
        LLVMDisposeTargetData(data_layout);

        let file_type = match kind {
            OutputKind::Object => LLVMCodeGenFileType::LLVMObjectFile,
            OutputKind::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputKind::LlvmIr => unreachable!("handled above"),
        };
        let path = CString::new(out_path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| "output path contains a NUL byte".to_string())?;
        let mut emit_err: *mut c_char = std::ptr::null_mut();
        let failed = LLVMTargetMachineEmitToFile(
            machine,
            llvm_module,
            path.as_ptr() as *mut c_char,
            file_type,
            &mut emit_err,
        );
        LLVMDisposeTargetMachine(machine);
        if failed != 0 {
            let msg = c_str(emit_err);
            LLVMDisposeMessage(emit_err);
            return Err(format!("failed to emit {}: {msg}", out_path.display()));
        }
        Ok(())
    }
}
