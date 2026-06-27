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

use std::ffi::{CString, c_char, c_int, c_void};
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
    LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup, LLVMOrcLLJITMangleAndIntern, LLVMOrcLLJITRef,
};
use llvm_sys::orc2::{
    LLVMJITEvaluatedSymbol, LLVMJITSymbolFlags, LLVMJITSymbolGenericFlags, LLVMOrcAbsoluteSymbols,
    LLVMOrcCSymbolMapPair, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath,
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess,
    LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
    LLVMOrcDefinitionGeneratorRef, LLVMOrcDisposeThreadSafeContext, LLVMOrcExecutorAddress,
    LLVMOrcJITDylibAddGenerator, LLVMOrcJITDylibDefine,
};
use llvm_sys::prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef};
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::LLVMGetDefaultTargetTriple;

use melior::ir::Module;

// The MLIR→LLVM-IR translation and the backend LLVM pass pipeline are shared with
// the AOT compiler; `OptLevel` is the backend's pipeline level.
use reussir_backend::llvm::{run_backend_llvm_pipeline, translate_to_llvm_ir};
pub use reussir_backend::pipeline::OptLevel;

// TPDE object compilation is JIT-specific (libReussirCAPI / lib/CAPI/Jit.cpp); it
// cannot be reached through the LLVM-C ORC API. Linked via reussir-backend-sys.
unsafe extern "C" {
    fn reussirTpdeCompileToObject(
        module: LLVMModuleRef,
        data_layout: *const c_char,
        triple: *const c_char,
    ) -> LLVMMemoryBufferRef;
    fn reussirHasTPDE() -> c_int;
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
        tracing::debug!(tpde = has_tpde(), "created ORC LLJIT engine");
        Ok(engine)
    }

    /// Creates a JIT with the Reussir runtime's `__reussir_*` symbols defined
    /// from the linked-in `reussir-rt` (see [`OrcJit::define_runtime_symbols`]),
    /// so JIT'd code calling the runtime resolves against the in-process runtime
    /// — no runtime shared library to locate or load.
    pub fn with_runtime() -> Result<Self, String> {
        let engine = Self::new()?;
        engine.define_runtime_symbols()?;
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
        tracing::debug!(path, "loaded shared library for symbol resolution");
        Ok(())
    }

    /// Defines each `(name, address)` pair in the main `JITDylib` as an absolute
    /// symbol, so JIT'd code resolves those names to the given in-process
    /// addresses. This exposes statically linked functions to JIT'd code without
    /// routing through a shared library or the process symbol table.
    pub fn define_symbols(&self, symbols: &[(&str, *const c_void)]) -> Result<(), String> {
        if symbols.is_empty() {
            return Ok(());
        }
        // Reject NUL bytes before interning anything, so a bad name cannot leak
        // already-interned string-pool references.
        let names = symbols
            .iter()
            .map(|(name, _)| CString::new(*name))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "symbol name has a NUL byte")?;

        // Exported + callable: a normal function definition visible to lookups.
        let generic_flags = LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8
            | LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8;

        unsafe {
            let mut pairs: Vec<LLVMOrcCSymbolMapPair> = names
                .iter()
                .zip(symbols)
                .map(|(name, (_, address))| LLVMOrcCSymbolMapPair {
                    // Interns the (mangled) name; `LLVMOrcAbsoluteSymbols` takes
                    // ownership of this string-pool reference.
                    Name: LLVMOrcLLJITMangleAndIntern(self.jit, name.as_ptr()),
                    Sym: LLVMJITEvaluatedSymbol {
                        Address: *address as LLVMOrcExecutorAddress,
                        Flags: LLVMJITSymbolFlags {
                            GenericFlags: generic_flags,
                            TargetFlags: 0,
                        },
                    },
                })
                .collect();

            // Builds a materialization unit owning the interned names, then hands
            // it to the dylib; a successful `Define` consumes the unit.
            let unit = LLVMOrcAbsoluteSymbols(pairs.as_mut_ptr(), pairs.len());
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            check_error(LLVMOrcJITDylibDefine(dylib, unit))?;
        }
        Ok(())
    }

    /// Defines the Reussir runtime's `__reussir_*` symbols, taken from the
    /// linked-in [`reussir_rt`], in the main `JITDylib`. JIT'd code then resolves
    /// runtime calls against the in-process runtime, so no runtime shared library
    /// needs to be located or loaded.
    pub fn define_runtime_symbols(&self) -> Result<(), String> {
        let symbols: Vec<(&str, *const c_void)> = reussir_rt::symbols::exported_symbols()
            .iter()
            .map(|symbol| (symbol.name, symbol.address))
            .collect();
        self.define_symbols(&symbols)?;
        tracing::debug!(count = symbols.len(), "defined in-process runtime symbols");
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
    ///
    /// TODO(polyffi): this path does not gather/link polymorphic FFI. The module
    /// arrives already lowered, but `reussir_backend::llvm::LlvmLowering::prepare`
    /// (compile + gather) must run *before* the lowering pipeline erases the
    /// `reussir.polyffi` ops, and `finish` (translate + link) after — see the AOT
    /// compiler's flow. A polyffi-capable JIT entry would orchestrate
    /// `prepare → run_lowering_pipeline → finish` and add the resulting
    /// [`reussir_backend::llvm::Finalized`] here. Until then a module containing
    /// `reussir.polyffi` fails loudly at `run_lowering_pipeline`, not silently.
    pub fn add_module(&self, module: &Module, opt: OptLevel) -> Result<(), String> {
        let _span = tracing::debug_span!("orcjit_add_module", opt = ?opt).entered();
        unsafe {
            // The translated module is created in (and owned alongside) `context`.
            let context = LLVMContextCreate();
            let llvm_module = match translate_to_llvm_ir(module, context) {
                Ok(llvm_module) => llvm_module,
                Err(message) => {
                    LLVMContextDispose(context);
                    tracing::error!("{message}");
                    return Err(message);
                }
            };
            tracing::trace!("translated MLIR module to LLVM IR");

            if opt == OptLevel::Tpde {
                // TPDE compiles the IR to an object directly; the IR module and
                // its context are not handed to the JIT, so we free them here.
                let result = self.add_tpde_object(llvm_module);
                LLVMDisposeModule(llvm_module);
                LLVMContextDispose(context);
                return result;
            }

            // Run the Reussir LLVM passes in place, then add the IR module.
            tracing::trace!("running backend LLVM pass pipeline");
            run_backend_llvm_pipeline(llvm_module, opt);
            self.add_owned_module(context, llvm_module)?;
            tracing::debug!("added LLVM-IR module to the JIT");
            Ok(())
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
            tracing::debug!("compiling module to an object with TPDE");
            let buffer = reussirTpdeCompileToObject(module, data_layout, triple);
            LLVMDisposeMessage(triple);
            if buffer.is_null() {
                tracing::error!("TPDE compilation failed");
                return Err("TPDE compilation failed".into());
            }
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            // Consumes `buffer`.
            check_error(LLVMOrcLLJITAddObjectFile(self.jit, dylib, buffer))?;
            tracing::debug!("added TPDE object to the JIT");
            Ok(())
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
        check_error(unsafe { LLVMOrcLLJITLookup(self.jit, &mut address, symbol.as_ptr()) })
            .inspect_err(|error| tracing::debug!(name, %error, "symbol lookup failed"))?;
        tracing::trace!(name, address, "resolved symbol");
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

    // A host function defined as an absolute symbol and called from JIT'd code.
    extern "C" fn host_meaning() -> i32 {
        42
    }

    #[test]
    fn define_symbols_exposes_a_host_function() {
        let jit = OrcJit::new().expect("create LLJIT");
        jit.define_symbols(&[("host_meaning", host_meaning as *const c_void)])
            .expect("define host symbol");
        jit.add_ir_module(
            "caller",
            "declare i32 @host_meaning()\n\
             define i32 @ask() { %r = call i32 @host_meaning() ret i32 %r }",
        )
        .expect("add caller");

        let address = jit.lookup("ask").expect("lookup ask");
        let ask: extern "C" fn() -> i32 = unsafe { std::mem::transmute(address as usize) };
        assert_eq!(ask(), 42);
    }

    #[test]
    fn with_runtime_resolves_runtime_symbols_in_process() {
        // The runtime is linked into this test binary via the `reussir-rt`
        // dependency; `with_runtime` defines its `__reussir_*` symbols directly,
        // so JIT'd code calling the allocator resolves without any shared library.
        let jit = OrcJit::with_runtime().expect("create JIT with runtime");
        jit.add_ir_module(
            "alloc_roundtrip",
            "declare ptr @__reussir_allocate(i64, i64)\n\
             declare void @__reussir_deallocate(ptr, i64, i64)\n\
             define i64 @roundtrip() {\n\
               %p = call ptr @__reussir_allocate(i64 8, i64 16)\n\
               %is_null = icmp eq ptr %p, null\n\
               call void @__reussir_deallocate(ptr %p, i64 8, i64 16)\n\
               %z = zext i1 %is_null to i64\n\
               ret i64 %z\n\
             }",
        )
        .expect("add module");

        let address = jit.lookup("roundtrip").expect("lookup roundtrip");
        let roundtrip: extern "C" fn() -> i64 = unsafe { std::mem::transmute(address as usize) };
        // Returns 0 when the allocator handed back a non-null pointer.
        assert_eq!(roundtrip(), 0);
    }
}
