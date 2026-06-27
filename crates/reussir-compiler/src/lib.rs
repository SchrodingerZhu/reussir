//! Ahead-of-time compilation: a finalized LLVM IR module → an object file,
//! assembly, or LLVM IR on disk.
//!
//! This is the AOT counterpart to `reussir-jit`: instead of adding the finalized
//! LLVM IR to an ORC JIT session, it emits a file with a host (or cross)
//! `TargetMachine`. It links neither the JIT engine nor the Reussir runtime — an
//! emitted object is linked against `reussir-rt` later by the C toolchain.
//!
//! The `reussir-compiler` binary drives the shared lowering in
//! [`reussir_backend::llvm`] — build a [`TargetMachine`] (its data layout feeds
//! the polymorphic-FFI gather), run the lowering pipeline, finalize, then hand
//! the [`Finalized`] module to [`emit_to_file`].

use std::ffi::{CString, c_char};
use std::path::Path;

use llvm_sys::core::{
    LLVMContextDispose, LLVMDisposeMessage, LLVMDisposeModule, LLVMPrintModuleToString,
    LLVMSetTarget,
};
use llvm_sys::prelude::LLVMModuleRef;
use llvm_sys::target::{
    LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos,
    LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets, LLVM_InitializeNativeAsmParser,
    LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget, LLVMCopyStringRepOfTargetData,
    LLVMDisposeTargetData, LLVMSetModuleDataLayout,
};
use llvm_sys::target_machine::{
    LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout,
    LLVMCreateTargetMachine, LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple,
    LLVMGetHostCPUFeatures, LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
    LLVMTargetMachineEmitToFile, LLVMTargetMachineRef, LLVMTargetRef,
};

use reussir_backend::llvm::{Finalized, run_backend_llvm_pipeline};
use reussir_backend::pipeline::OptLevel;

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

/// Registers LLVM targets before codegen. A host compile needs only the native
/// target; a custom triple may name any architecture, so register them all —
/// otherwise `LLVMGetTargetFromTriple` would not find a foreign target's backend.
///
/// No `Once` guard: LLVM's `TargetRegistry::RegisterTarget` early-returns once a
/// target is registered (`if (T.Name) return;`), and `TargetSelect.h` documents
/// that "it is legal for a client to make multiple calls to this function", so
/// these are idempotent on every compile.
fn init_targets(cross: bool) {
    unsafe {
        if cross {
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmPrinters();
            LLVM_InitializeAllAsmParsers();
        } else {
            // A non-zero return means the target is unavailable; codegen then
            // fails loudly at `LLVMGetTargetFromTriple`, so nothing to recover.
            LLVM_InitializeNativeTarget();
            LLVM_InitializeNativeAsmPrinter();
            LLVM_InitializeNativeAsmParser();
        }
    }
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

/// A host or cross `TargetMachine` plus the triple and data layout it implies.
///
/// Built up front (before the lowering pipeline) because the data layout must be
/// known before polymorphic-FFI gathering; the same machine then emits the file.
pub struct TargetMachine {
    machine: LLVMTargetMachineRef,
    triple: CString,
    data_layout: CString,
}

impl TargetMachine {
    /// Resolves `spec` (host by default), creates the target machine at the given
    /// optimization and relocation model, and queries its data layout.
    pub fn new(spec: &TargetSpec, opt: OptLevel, reloc_mode: RelocMode) -> Result<Self, String> {
        init_targets(spec.triple.is_some());
        unsafe {
            // A custom triple defaults CPU/features to empty (architecture
            // defaults); the host's would crash LLVM against a foreign triple.
            let foreign = spec.triple.is_some();
            let triple = match spec.triple.as_deref() {
                Some(t) => {
                    CString::new(t).map_err(|_| "target triple contains a NUL byte".to_string())?
                }
                None => take_llvm_string(LLVMGetDefaultTargetTriple()),
            };
            let cpu = match spec.cpu.as_deref() {
                Some(c) => {
                    CString::new(c).map_err(|_| "target CPU contains a NUL byte".to_string())?
                }
                None if foreign => CString::default(),
                None => take_llvm_string(LLVMGetHostCPUName()),
            };
            let features = match spec.features.as_deref() {
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
            if machine.is_null() {
                return Err("failed to create the target machine".to_string());
            }
            let target_data = LLVMCreateTargetDataLayout(machine);
            let data_layout = take_llvm_string(LLVMCopyStringRepOfTargetData(target_data));
            LLVMDisposeTargetData(target_data);
            Ok(TargetMachine {
                machine,
                triple,
                data_layout,
            })
        }
    }

    /// The target's data layout string — feed this to
    /// [`reussir_backend::llvm::LlvmLowering::prepare`].
    pub fn data_layout(&self) -> &str {
        self.data_layout.to_str().unwrap_or("")
    }
}

impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetMachine(self.machine) }
    }
}

/// Runs the backend LLVM pass pipeline over a [`Finalized`] module, emits the
/// requested artifact for `machine`, and disposes the module and its context.
pub fn emit_to_file(
    finalized: Finalized,
    machine: &TargetMachine,
    opt: OptLevel,
    kind: OutputKind,
    out_path: &Path,
) -> Result<(), String> {
    unsafe {
        run_backend_llvm_pipeline(finalized.module, opt);
        let result = emit(finalized.module, machine, kind, out_path);
        LLVMDisposeModule(finalized.module);
        LLVMContextDispose(finalized.context);
        result
    }
}

unsafe fn emit(
    llvm_module: LLVMModuleRef,
    machine: &TargetMachine,
    kind: OutputKind,
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

        // Stamp the module with the target triple and data layout so the emitted
        // object's ABI matches the target (and so `%cc` can link it).
        LLVMSetTarget(llvm_module, machine.triple.as_ptr());
        let target_data = LLVMCreateTargetDataLayout(machine.machine);
        // `LLVMSetModuleDataLayout` copies the layout into the module, so the
        // target-data handle is ours to dispose.
        LLVMSetModuleDataLayout(llvm_module, target_data);
        LLVMDisposeTargetData(target_data);

        let file_type = match kind {
            OutputKind::Object => LLVMCodeGenFileType::LLVMObjectFile,
            OutputKind::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputKind::LlvmIr => unreachable!("handled above"),
        };
        let path = CString::new(out_path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| "output path contains a NUL byte".to_string())?;
        let mut emit_err: *mut c_char = std::ptr::null_mut();
        let failed = LLVMTargetMachineEmitToFile(
            machine.machine,
            llvm_module,
            path.as_ptr() as *mut c_char,
            file_type,
            &mut emit_err,
        );
        if failed != 0 {
            let msg = c_str(emit_err);
            LLVMDisposeMessage(emit_err);
            return Err(format!("failed to emit {}: {msg}", out_path.display()));
        }
        Ok(())
    }
}
