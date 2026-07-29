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

pub mod driver;
pub mod externs;
pub mod package;

use std::ffi::{CString, c_char};
use std::path::{Path, PathBuf};

use llvm_sys::core::{
    LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
    LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMDisposeModule, LLVMPrintModuleToString,
    LLVMSetTarget,
};
use llvm_sys::ir_reader::LLVMParseIRInContext2;
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
    LLVMNormalizeTargetTriple, LLVMTargetMachineEmitToFile, LLVMTargetMachineRef, LLVMTargetRef,
};

use reussir_backend::llvm::{Finalized, LtoMode, run_backend_llvm_pipeline};
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
    /// LLVM bitcode for the given LTO mode (`.bc`) — what a static library's
    /// members are under `--lto`, so the link step can optimize across them.
    Bitcode(LtoMode),
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
    /// Dynamic, but not position-independent (LLVM's `dynamic-no-pic`).
    DynamicNoPic,
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
        RelocMode::DynamicNoPic => LLVMRelocMode::LLVMRelocDynamicNoPic,
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

/// The target features a triple implies on its own — what the machine must
/// have for code compiled *for* that target to mean what it says, regardless
/// of whether the user passed `--target-features`.
///
/// Only WebAssembly needs this today, and it needs it badly. Every other
/// architecture Reussir targets has atomics in its baseline, so an
/// `atomicrmw` in the IR is an atomic instruction in the object. WebAssembly
/// does not: without the `atomics` feature the backend is entitled to assume
/// the module is single-threaded and lowers the same IR to a plain load and
/// store. A threaded target (`wasm32-wasip1-threads`, whose environment
/// normalizes to `-threads`) is exactly the case where that assumption is
/// wrong — the atomic reference counts and the lock-guarded cells are the
/// program's shared state — and the failure is silent: the module builds,
/// links, runs, and loses updates under contention.
///
/// So the triple decides, as it does for rustc: the same target name gives
/// rustc `+atomics,+bulk-memory,+mutable-globals` (and a `--shared-memory`
/// link) for the standard library and every polyffi texture, and this keeps
/// the objects `rrc` emits on the same footing. An explicit
/// `--target-features` is appended after these, so `-atomics` still turns
/// them back off: LLVM takes the last setting of a feature.
fn implied_target_features(triple: &str) -> &'static str {
    let mut components = triple.split('-');
    let arch = components.next().unwrap_or_default();
    let threaded = components.next_back() == Some("threads");
    if matches!(arch, "wasm32" | "wasm64") && threaded {
        "+atomics,+bulk-memory,+mutable-globals"
    } else {
        ""
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
                    let requested = CString::new(t)
                        .map_err(|_| "target triple contains a NUL byte".to_string())?;
                    // Rust target names may use LLVM aliases (notably
                    // `wasm32-wasip1`). Stamp LLVM's canonical spelling so
                    // rustc-produced polyffi bitcode links without a
                    // different-target-triples warning.
                    take_llvm_string(LLVMNormalizeTargetTriple(requested.as_ptr()))
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
            // What the triple implies (see `implied_target_features`) comes
            // first, so an explicit `--target-features` can still countermand
            // it; the host's own feature string is the baseline when no
            // triple was given, and nothing is implied on top of it.
            let implied = implied_target_features(triple.to_str().unwrap_or_default());
            let features = match spec.features.as_deref() {
                Some(f) if implied.is_empty() => {
                    CString::new(f).map_err(|_| "target features contain a NUL byte".to_string())?
                }
                Some(f) => CString::new(format!("{implied},{f}"))
                    .map_err(|_| "target features contain a NUL byte".to_string())?,
                None if foreign => {
                    CString::new(implied).expect("implied features are NUL-free literals")
                }
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

    /// The resolved target triple — with [`data_layout`](Self::data_layout),
    /// what [`reussir_backend::pipeline::attach_target_spec`] stamps on the
    /// module before the lowering pipeline.
    pub fn triple(&self) -> &str {
        self.triple.to_str().unwrap_or("")
    }
}

impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetMachine(self.machine) }
    }
}

/// Parses LLVM IR text into a fresh context, yielding a [`Finalized`] ready for
/// [`emit_to_file`] — the entry point for a `.ll` input, which skips the whole
/// MLIR front of the pipeline. `name` labels parse diagnostics.
pub fn parse_llvm_ir(name: &str, source: &str) -> Result<Finalized, String> {
    init_targets(false);
    unsafe {
        let context = LLVMContextCreate();
        // `source` is borrowed, so copy it into an LLVM-owned buffer we control.
        let name_c =
            CString::new(name).unwrap_or_else(|_| CString::new("<input>").expect("valid cstr"));
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            source.as_ptr() as *const c_char,
            source.len(),
            name_c.as_ptr(),
        );
        let mut module: LLVMModuleRef = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        // `LLVMParseIRInContext2` copies what it needs and does *not* take the
        // buffer, so we free it ourselves either way.
        let failed = LLVMParseIRInContext2(context, buffer, &mut module, &mut err) != 0;
        LLVMDisposeMemoryBuffer(buffer);
        if failed {
            let msg = c_str(err);
            LLVMDisposeMessage(err);
            LLVMContextDispose(context);
            return Err(format!("{name}: failed to parse LLVM IR: {msg}"));
        }
        Ok(Finalized { context, module })
    }
}

/// Runs the backend LLVM pass pipeline over a [`Finalized`] module, emits the
/// requested artifact for `machine`, and disposes the module and its context.
///
/// The artifact decides the pipeline: emitting bitcode for an LTO mode runs
/// that mode's pre-link pipeline, since the link step will do the rest.
pub fn emit_to_file(
    finalized: Finalized,
    machine: &TargetMachine,
    opt: OptLevel,
    kind: OutputKind,
    out_path: &Path,
) -> Result<(), String> {
    let lto = match kind {
        OutputKind::Bitcode(mode) => mode,
        _ => LtoMode::None,
    };
    unsafe {
        // Stamp the target before the pass pipeline runs: the module's triple
        // and data layout feed the pipeline's queries, and the machine feeds
        // the TargetTransformInfo the cost-driven passes (vectorization)
        // decide against.
        stamp_target(finalized.module, machine);
        tracing::debug!(?opt, "running the LLVM pass pipeline");
        run_backend_llvm_pipeline(finalized.module, opt, machine.machine, lto);
        tracing::debug!("running LLVM codegen");
        let result = emit(finalized.module, machine, kind, out_path);
        LLVMDisposeModule(finalized.module);
        LLVMContextDispose(finalized.context);
        result
    }
}

/// Writes a static library at `path` over `members`, in order, with a symbol
/// table indexing each — native objects and LLVM bitcode alike, so an
/// LTO-mode library resolves through the linker's plugin. `triple` selects
/// the archive flavor so a cross-compiled library is written in the target's
/// format.
pub fn write_archive(path: &Path, members: &[PathBuf], triple: &str) -> Result<(), String> {
    let member_paths = members
        .iter()
        .map(|member| {
            CString::new(member.as_os_str().to_string_lossy().as_bytes())
                .map_err(|_| format!("member path `{}` contains a NUL byte", member.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pointers: Vec<*const c_char> = member_paths.iter().map(|m| m.as_ptr()).collect();
    let out = CString::new(path.as_os_str().to_string_lossy().as_bytes())
        .map_err(|_| "output path contains a NUL byte".to_string())?;
    let triple = CString::new(triple).map_err(|_| "target triple contains a NUL byte")?;
    unsafe {
        let err = reussir_backend_sys::reussirWriteArchive(
            out.as_ptr(),
            pointers.as_ptr(),
            pointers.len(),
            triple.as_ptr(),
        );
        if err.is_null() {
            return Ok(());
        }
        let msg = c_str(err);
        LLVMDisposeMessage(err);
        Err(format!("failed to write {}: {msg}", path.display()))
    }
}

/// Stamps `machine`'s target triple and data layout onto the module, so the
/// emitted artifact's ABI matches the target (and so `%cc` and downstream
/// tools can consume the IR-text form).
unsafe fn stamp_target(llvm_module: LLVMModuleRef, machine: &TargetMachine) {
    unsafe {
        LLVMSetTarget(llvm_module, machine.triple.as_ptr());
        let target_data = LLVMCreateTargetDataLayout(machine.machine);
        // `LLVMSetModuleDataLayout` copies the layout into the module, so the
        // target-data handle is ours to dispose.
        LLVMSetModuleDataLayout(llvm_module, target_data);
        LLVMDisposeTargetData(target_data);
    }
}

unsafe fn emit(
    llvm_module: LLVMModuleRef,
    machine: &TargetMachine,
    kind: OutputKind,
    out_path: &Path,
) -> Result<(), String> {
    unsafe {
        // LLVM IR text needs no target-machine emission — just print the module.
        if kind == OutputKind::LlvmIr {
            let s = LLVMPrintModuleToString(llvm_module);
            let text = c_str(s);
            LLVMDisposeMessage(s);
            return std::fs::write(out_path, text)
                .map_err(|e| format!("failed to write {}: {e}", out_path.display()));
        }

        // Bitcode likewise bypasses the target machine: the *linker* runs
        // instruction selection, once, over the merged module.
        if let OutputKind::Bitcode(mode) = kind {
            let path = CString::new(out_path.as_os_str().to_string_lossy().as_bytes())
                .map_err(|_| "output path contains a NUL byte".to_string())?;
            let err = reussir_backend_sys::reussirWriteLtoBitcode(
                llvm_module as reussir_backend_sys::LLVMModuleRef,
                path.as_ptr(),
                mode.as_c_int(),
            );
            if err.is_null() {
                return Ok(());
            }
            let msg = c_str(err);
            LLVMDisposeMessage(err);
            return Err(msg);
        }

        let file_type = match kind {
            OutputKind::Object => LLVMCodeGenFileType::LLVMObjectFile,
            OutputKind::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputKind::LlvmIr | OutputKind::Bitcode(_) => unreachable!("handled above"),
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

#[cfg(test)]
mod tests {
    use super::implied_target_features;

    /// The threaded WebAssembly targets — in either spelling, since the
    /// caller works from LLVM's normalized triple but the flag the user
    /// types is rustc's name.
    #[test]
    fn threaded_wasm_implies_atomics() {
        for triple in [
            "wasm32-wasip1-threads",
            "wasm32-unknown-wasip1-threads",
            "wasm64-unknown-wasip1-threads",
        ] {
            assert_eq!(
                implied_target_features(triple),
                "+atomics,+bulk-memory,+mutable-globals",
                "{triple}"
            );
        }
    }

    /// Everything else keeps its architectural default: single-threaded wasm
    /// stays single-threaded (its reference counts are plain loads and
    /// stores, correctly), and no native target needs a feature to make
    /// `atomicrmw` atomic.
    #[test]
    fn other_targets_imply_nothing() {
        for triple in [
            "wasm32-wasip1",
            "wasm32-unknown-unknown",
            "wasm32-unknown-emscripten",
            "x86_64-unknown-linux-gnu",
            "aarch64-apple-darwin",
            // Not a wasm target, whatever it calls its environment.
            "x86_64-unknown-linux-threads",
            "",
        ] {
            assert_eq!(implied_target_features(triple), "", "{triple}");
        }
    }
}
