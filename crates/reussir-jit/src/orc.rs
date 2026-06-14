//! ORC-based JIT engine built directly on LLVM's `LLJIT` C API (`llvm-sys`).
//!
//! [`OrcJit`] owns a persistent `LLJIT`: a single execution session with a main
//! `JITDylib` into which modules are added incrementally and resolved against one
//! another. This replaces MLIR's one-shot `ExecutionEngine` (one module per
//! engine, no compiler hooks) with the hand-built ORC stack the Reussir JIT needs
//! — REPL-style incremental use, runtime-library symbol resolution, and (layered
//! on later) the TPDE compiler, custom LLVM passes and lazy materialization.
//!
//! Modules enter either as LLVM IR text ([`OrcJit::add_ir_module`]) or as a
//! melior [`Module`] already lowered to the LLVM dialect
//! ([`OrcJit::add_module`]), which is translated to LLVM IR via mlir-sys.

use std::ffi::{CString, c_char, c_int};
use std::sync::Once;

use llvm_sys::core::{
    LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
    LLVMDisposeMessage, LLVMDisposeModule,
};
use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMErrorRef, LLVMGetErrorMessage};
#[allow(deprecated)]
use llvm_sys::ir_reader::LLVMParseIRInContext;
use llvm_sys::orc2::lljit::{
    LLVMOrcCreateLLJIT, LLVMOrcDisposeLLJIT, LLVMOrcLLJITAddLLVMIRModule,
    LLVMOrcLLJITAddObjectFile, LLVMOrcLLJITGetDataLayoutStr, LLVMOrcLLJITGetGlobalPrefix,
    LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup, LLVMOrcLLJITRef,
};
use llvm_sys::orc2::{
    LLVMOrcCreateDynamicLibrarySearchGeneratorForPath,
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess,
    LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
    LLVMOrcDefinitionGeneratorRef, LLVMOrcDisposeThreadSafeContext, LLVMOrcExecutorAddress,
    LLVMOrcJITDylibAddGenerator,
};
use llvm_sys::prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef};
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::LLVMGetDefaultTargetTriple;

use melior::ir::Module;

use crate::runtime_library_path;

// LLVM-side codegen helpers provided by libReussirCAPI (lib/CAPI/Jit.cpp): the
// custom Reussir LLVM passes and TPDE, which cannot be reached through the
// LLVM-C ORC API. Linked via the reussir-backend-sys dependency.
unsafe extern "C" {
    fn reussirRunBackendLLVMPipeline(module: LLVMModuleRef, opt: c_int);
    fn reussirTpdeCompileToObject(
        module: LLVMModuleRef,
        data_layout: *const c_char,
        triple: *const c_char,
    ) -> LLVMMemoryBufferRef;
    fn reussirHasTPDE() -> c_int;
}

/// Optimization / codegen level for [`OrcJit::add_module`], mirroring the
/// backend's `ReussirOptOption`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No LLVM optimization.
    None,
    /// Default optimization (`-O2`).
    #[default]
    Default,
    /// Aggressive optimization (`-O3`).
    Aggressive,
    /// Optimize for size (`-Os`).
    Size,
    /// Compile with the TPDE fast back end instead of the LLVM code generator.
    Tpde,
}

impl OptLevel {
    fn as_c(self) -> c_int {
        match self {
            OptLevel::None => 0,
            OptLevel::Default => 1,
            OptLevel::Aggressive => 2,
            OptLevel::Size => 3,
            OptLevel::Tpde => 4,
        }
    }
}

/// Reports whether TPDE support was compiled into the backend. [`OptLevel::Tpde`]
/// falls back to an error when this is false.
pub fn has_tpde() -> bool {
    unsafe { reussirHasTPDE() != 0 }
}

// LLJIT codegen requires the native target to be initialized, exactly once.
fn init_native_target() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();
    });
}

// Copies an LLVM-owned C string into a Rust `String` (the caller still disposes
// the original via the matching LLVM dispose function).
unsafe fn copy_c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

// Turns an `LLVMErrorRef` into a `Result`; a null error is success, otherwise the
// error is consumed and its message returned.
fn check_error(error: LLVMErrorRef) -> Result<(), String> {
    if error.is_null() {
        return Ok(());
    }
    unsafe {
        let message = LLVMGetErrorMessage(error);
        let owned = copy_c_string(message);
        LLVMDisposeErrorMessage(message);
        Err(owned)
    }
}

/// A persistent ORC `LLJIT` engine for Reussir.
pub struct OrcJit {
    jit: LLVMOrcLLJITRef,
}

// The LLJIT session is internally synchronized and may be used across threads.
unsafe impl Send for OrcJit {}
unsafe impl Sync for OrcJit {}

impl OrcJit {
    /// Creates a JIT with a default host `LLJIT` and a generator that resolves
    /// symbols already present in the current process (libc, the runtime if it is
    /// linked in, etc.).
    pub fn new() -> Result<Self, String> {
        init_native_target();
        let mut jit: LLVMOrcLLJITRef = std::ptr::null_mut();
        // A null builder selects host defaults.
        check_error(unsafe { LLVMOrcCreateLLJIT(&mut jit, std::ptr::null_mut()) })?;
        let engine = Self { jit };
        engine.add_process_symbols()?;
        Ok(engine)
    }

    /// Creates a JIT and, if the Reussir runtime shared library can be located
    /// (see [`runtime_library_path`]), makes its symbols (`__reussir_allocate`,
    /// …) resolvable. Without it, JIT'd code calling the runtime will fail to
    /// resolve those symbols.
    pub fn with_runtime() -> Result<Self, String> {
        let engine = Self::new()?;
        match runtime_library_path() {
            Some(path) => engine.add_library(&path.to_string_lossy())?,
            None => tracing::warn!(
                "Reussir runtime library not found; JIT'd code referencing \
                 runtime symbols will fail to resolve"
            ),
        }
        Ok(engine)
    }

    // The data-layout global symbol prefix (e.g. '_' on macOS, '\0' elsewhere).
    fn global_prefix(&self) -> c_char {
        unsafe { LLVMOrcLLJITGetGlobalPrefix(self.jit) }
    }

    // Attaches a definition generator to the main JITDylib.
    fn add_generator(&self, generator: LLVMOrcDefinitionGeneratorRef) {
        unsafe {
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            LLVMOrcJITDylibAddGenerator(dylib, generator);
        }
    }

    fn add_process_symbols(&self) -> Result<(), String> {
        let mut generator: LLVMOrcDefinitionGeneratorRef = std::ptr::null_mut();
        check_error(unsafe {
            LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                self.global_prefix(),
                None,
                std::ptr::null_mut(),
            )
        })?;
        self.add_generator(generator);
        Ok(())
    }

    /// Makes the symbols exported by the shared library at `path` resolvable from
    /// JIT'd code.
    pub fn add_library(&self, path: &str) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|_| "library path has a NUL byte")?;
        let mut generator: LLVMOrcDefinitionGeneratorRef = std::ptr::null_mut();
        check_error(unsafe {
            LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
                &mut generator,
                c_path.as_ptr(),
                self.global_prefix(),
                None,
                std::ptr::null_mut(),
            )
        })?;
        self.add_generator(generator);
        Ok(())
    }

    /// Parses LLVM IR text and adds it to the main `JITDylib`. `name` only labels
    /// the buffer for diagnostics.
    pub fn add_ir_module(&self, name: &str, ir: &str) -> Result<(), String> {
        unsafe {
            let context = LLVMContextCreate();

            let buffer_name = CString::new(name).map_err(|_| "module name has a NUL byte")?;
            let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                ir.as_ptr() as *const c_char,
                ir.len(),
                buffer_name.as_ptr(),
            );

            // Parses into `context` and takes ownership of `buffer`.
            let mut module: LLVMModuleRef = std::ptr::null_mut();
            let mut error_message: *mut c_char = std::ptr::null_mut();
            #[allow(deprecated)]
            let failed =
                LLVMParseIRInContext(context, buffer, &mut module, &mut error_message) != 0;
            if failed {
                let message = copy_c_string(error_message);
                LLVMDisposeMessage(error_message);
                LLVMContextDispose(context);
                return Err(format!("failed to parse LLVM IR: {message}"));
            }

            self.add_owned_module(context, module)
        }
    }

    /// Translates a melior [`Module`] already lowered to the LLVM dialect into
    /// LLVM IR, optimizes/compiles it at `opt`, and adds it to the main
    /// `JITDylib`. The module's MLIR context must have the LLVM/builtin
    /// translations registered (as `reussir_backend::context` provides).
    ///
    /// For [`OptLevel::Tpde`] the module is compiled to an object file by TPDE
    /// and added directly; otherwise the Reussir LLVM pass pipeline runs and the
    /// IR module is handed to the default `LLJIT` compiler.
    pub fn add_module(&self, module: &Module, opt: OptLevel) -> Result<(), String> {
        unsafe {
            let context = LLVMContextCreate();
            let operation = mlir_sys::mlirModuleGetOperation(module.to_raw());
            // The translated module is created in (and owned alongside) `context`.
            let llvm_module = mlir_sys::mlirTranslateModuleToLLVMIR(
                operation,
                context as mlir_sys::LLVMContextRef,
            ) as LLVMModuleRef;
            if llvm_module.is_null() {
                LLVMContextDispose(context);
                return Err("failed to translate MLIR module to LLVM IR".into());
            }

            if opt == OptLevel::Tpde {
                // TPDE compiles the IR to an object directly; the IR module and
                // its context are not handed to the JIT, so we free them here.
                let result = self.add_tpde_object(llvm_module);
                LLVMDisposeModule(llvm_module);
                LLVMContextDispose(context);
                return result;
            }

            // Run the Reussir LLVM passes in place, then add the IR module.
            reussirRunBackendLLVMPipeline(llvm_module, opt.as_c());
            self.add_owned_module(context, llvm_module)
        }
    }

    // Compiles a module to an object with TPDE and adds the object to the main
    // JITDylib. Does not take ownership of `module`.
    fn add_tpde_object(&self, module: LLVMModuleRef) -> Result<(), String> {
        if !has_tpde() {
            return Err("TPDE support is not compiled into the backend".into());
        }
        unsafe {
            // The JIT's data layout and the host triple, which TPDE stamps onto
            // the module before compiling.
            let data_layout = LLVMOrcLLJITGetDataLayoutStr(self.jit);
            let triple = LLVMGetDefaultTargetTriple();
            let buffer = reussirTpdeCompileToObject(module, data_layout, triple);
            LLVMDisposeMessage(triple);
            if buffer.is_null() {
                return Err("TPDE compilation failed".into());
            }
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            // Consumes `buffer`.
            check_error(LLVMOrcLLJITAddObjectFile(self.jit, dylib, buffer))
        }
    }

    // Wraps an owned (context, module) pair in a thread-safe module and adds it to
    // the main JITDylib. Consumes ownership of both.
    unsafe fn add_owned_module(
        &self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
    ) -> Result<(), String> {
        unsafe {
            // The thread-safe context takes ownership of `context`; the
            // thread-safe module then takes ownership of `module` and shares the
            // context (ref-counted), so our TSC handle can be disposed here.
            let ts_context = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(context);
            let ts_module = LLVMOrcCreateNewThreadSafeModule(module, ts_context);
            LLVMOrcDisposeThreadSafeContext(ts_context);

            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            check_error(LLVMOrcLLJITAddLLVMIRModule(self.jit, dylib, ts_module))
        }
    }

    /// Looks up a symbol by name, materializing (compiling) it if needed, and
    /// returns its executable address (0 is never returned for a found symbol).
    pub fn lookup(&self, name: &str) -> Result<u64, String> {
        let symbol = CString::new(name).map_err(|_| "symbol name has a NUL byte")?;
        let mut address: LLVMOrcExecutorAddress = 0;
        check_error(unsafe { LLVMOrcLLJITLookup(self.jit, &mut address, symbol.as_ptr()) })?;
        Ok(address)
    }
}

impl Drop for OrcJit {
    fn drop(&mut self) {
        // Ends the session and frees the engine; ignore the teardown error.
        unsafe {
            let _ = LLVMOrcDisposeLLJIT(self.jit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_llvm_ir_through_lljit() {
        let jit = OrcJit::new().expect("create LLJIT");
        let ir = r#"
            define i32 @addi(i32 %a, i32 %b) {
              %r = add i32 %a, %b
              ret i32 %r
            }
        "#;
        jit.add_ir_module("smoke", ir).expect("add module");

        let address = jit.lookup("addi").expect("lookup addi");
        let addi: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(address as usize) };
        assert_eq!(addi(2, 3), 5);
    }

    #[test]
    fn resolves_symbols_across_modules() {
        let jit = OrcJit::new().expect("create LLJIT");
        jit.add_ir_module("callee", "define i32 @answer() { ret i32 42 }")
            .expect("add callee");
        jit.add_ir_module(
            "caller",
            "declare i32 @answer()\n\
             define i32 @call_answer() { %r = call i32 @answer() ret i32 %r }",
        )
        .expect("add caller");

        let address = jit.lookup("call_answer").expect("lookup");
        let call: extern "C" fn() -> i32 = unsafe { std::mem::transmute(address as usize) };
        assert_eq!(call(), 42);
    }

    // Lowers a tiny `@add` module and runs it through the JIT at `opt`,
    // returning `add(20, 22)`.
    fn jit_add(opt: OptLevel) -> i32 {
        let context = reussir_backend::context();
        let source = r#"
            module {
              func.func @add(%a: i32, %b: i32) -> i32 {
                %0 = arith.addi %a, %b : i32
                func.return %0 : i32
              }
            }
        "#;
        let mut module = Module::parse(&context, source).expect("module should parse");
        reussir_backend::pipeline::run_lowering_pipeline(
            &context,
            &mut module,
            &reussir_backend::pipeline::LoweringOptions::default(),
        )
        .expect("lowering should succeed");

        let jit = OrcJit::new().expect("create LLJIT");
        jit.add_module(&module, opt).expect("add lowered module");

        let address = jit.lookup("add").expect("lookup add");
        let add: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(address as usize) };
        add(20, 22)
    }

    #[test]
    fn runs_a_lowered_mlir_module() {
        assert_eq!(jit_add(OptLevel::Default), 42);
    }

    #[test]
    fn runs_a_lowered_mlir_module_with_tpde() {
        if !has_tpde() {
            eprintln!("skipping: TPDE support not compiled in");
            return;
        }
        assert_eq!(jit_add(OptLevel::Tpde), 42);
    }
}
