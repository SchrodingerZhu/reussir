//! Just-in-time execution of lowered Reussir modules.
//!
//! [`Jit`] wraps melior's [`ExecutionEngine`] (the MLIR ORC-backed JIT) and
//! takes a module already lowered to the LLVM dialect — typically the output of
//! [`run_lowering_pipeline`](crate::pipeline::run_lowering_pipeline). It loads
//! the Reussir runtime shared library so JIT'd code can resolve runtime symbols,
//! and exposes symbol lookup and invocation.
//!
//! This is the eager, in-process engine. The lazy AST-callback materialization
//! the C++ bridge used to drive the REPL is intentionally left out; it was REPL
//! plumbing rather than part of the core JIT.

use std::path::{Path, PathBuf};

use melior::ExecutionEngine;
use melior::ir::Module;

/// A JIT engine over a module lowered to the LLVM dialect.
pub struct Jit {
    engine: ExecutionEngine,
}

impl Jit {
    /// Creates a JIT engine for `module`, optimizing at `optimization_level`
    /// (0–3, as for `clang -O`) and making the given shared libraries available
    /// for symbol resolution.
    ///
    /// `module` must already be lowered to the LLVM dialect and its context must
    /// have the LLVM/builtin translations registered (as [`crate::context`]
    /// provides).
    pub fn new(module: &Module, optimization_level: usize, shared_library_paths: &[&str]) -> Self {
        Self::with_options(
            module,
            optimization_level,
            shared_library_paths,
            /* enable_object_dump */ false,
            /* enable_pic */ false,
        )
    }

    /// Creates a JIT engine for `module` with the Reussir runtime library loaded
    /// when it can be located (see [`runtime_library_path`]). Without it, JIT'd
    /// code that calls runtime functions will fail to resolve those symbols.
    pub fn with_runtime(module: &Module, optimization_level: usize) -> Self {
        match runtime_library_path() {
            Some(path) => {
                let path = path.to_string_lossy().into_owned();
                Self::new(module, optimization_level, &[&path])
            }
            None => {
                tracing::warn!(
                    "Reussir runtime library not found; JIT'd code referencing \
                     runtime symbols will fail to resolve"
                );
                Self::new(module, optimization_level, &[])
            }
        }
    }

    /// Creates a JIT engine with full control over object dumping and
    /// position-independent code. `enable_object_dump` must be set to later call
    /// [`Jit::dump_to_object_file`].
    pub fn with_options(
        module: &Module,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
        enable_pic: bool,
    ) -> Self {
        Self {
            engine: ExecutionEngine::new(
                module,
                optimization_level,
                shared_library_paths,
                enable_object_dump,
                enable_pic,
            ),
        }
    }

    /// Looks up a symbol by name, returning a pointer to it (null if absent).
    pub fn lookup(&self, name: &str) -> *mut () {
        self.engine.lookup(name)
    }

    /// Invokes a packed-ABI function. `arguments` holds pointers to the
    /// arguments followed by pointers to the results, per the MLIR C interface
    /// (`llvm.emit_c_interface`) calling convention.
    ///
    /// # Safety
    ///
    /// The pointers in `arguments` must be valid and correctly typed for the
    /// invoked function; otherwise behavior is undefined.
    pub unsafe fn invoke_packed(
        &self,
        name: &str,
        arguments: &mut [*mut ()],
    ) -> Result<(), melior::Error> {
        unsafe { self.engine.invoke_packed(name, arguments) }
    }

    /// Registers an external symbol, making `ptr` resolvable from JIT'd code.
    ///
    /// # Safety
    ///
    /// `ptr` must remain valid for as long as the engine may call through it.
    pub unsafe fn register_symbol(&self, name: &str, ptr: *mut ()) {
        unsafe { self.engine.register_symbol(name, ptr) }
    }

    /// Writes the compiled module to an object file. The engine must have been
    /// created with object dumping enabled (see [`Jit::with_options`]).
    pub fn dump_to_object_file(&self, path: &str) {
        self.engine.dump_to_object_file(path)
    }

    /// Returns the underlying melior execution engine.
    pub fn engine(&self) -> &ExecutionEngine {
        &self.engine
    }
}

/// Locates the Reussir runtime shared library.
///
/// `REUSSIR_RT_LIBRARY` takes precedence when it points at an existing file;
/// otherwise a set of conventional locations relative to the working directory
/// is searched, mirroring the C++ bridge's lookup.
pub fn runtime_library_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("REUSSIR_RT_LIBRARY") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let file_name = runtime_library_file_name();
    const SEARCH_DIRS: &[&str] = &["build/lib", "lib", "../lib", "build/bin", "bin", "/usr/lib"];
    SEARCH_DIRS
        .iter()
        .map(|dir| Path::new(dir).join(file_name))
        .find(|candidate| candidate.is_file())
}

// Platform-specific file name of the Reussir runtime shared library.
fn runtime_library_file_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libreussir_rt.dylib"
    }
    #[cfg(target_os = "windows")]
    {
        "reussir_rt.dll"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "libreussir_rt.so"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{LoweringOptions, run_lowering_pipeline};

    #[test]
    fn jit_executes_lowered_function() {
        let context = crate::context();

        // A C-interface function so the engine can invoke it by the packed ABI.
        let source = r#"
            module {
              func.func @add(%arg0: i32, %arg1: i32) -> i32
                  attributes { llvm.emit_c_interface } {
                %0 = arith.addi %arg0, %arg1 : i32
                func.return %0 : i32
              }
            }
        "#;
        let mut module = Module::parse(&context, source).expect("module should parse");

        run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
            .expect("pipeline should succeed");

        let jit = Jit::new(&module, 2, &[]);

        let mut lhs: i32 = 2;
        let mut rhs: i32 = 3;
        let mut result: i32 = 0;
        unsafe {
            jit.invoke_packed(
                "add",
                &mut [
                    &mut lhs as *mut i32 as *mut (),
                    &mut rhs as *mut i32 as *mut (),
                    &mut result as *mut i32 as *mut (),
                ],
            )
            .expect("invocation should succeed");
        }

        assert_eq!(result, 5);
    }
}
