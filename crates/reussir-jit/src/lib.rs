//! Just-in-time execution for Reussir.
//!
//! [`OrcJit`] is an ORC `LLJIT`-based engine (built on `llvm-sys`): a persistent
//! execution session with a main `JITDylib` that modules are added to
//! incrementally and resolved against one another. It ingests modules already
//! lowered to the LLVM dialect — typically the output of
//! `reussir_backend::pipeline::run_lowering_pipeline` — translating them to LLVM
//! IR, and resolves runtime symbols from the Reussir runtime shared library.
//!
//! This replaces MLIR's one-shot `ExecutionEngine`: a single `OrcJit` hosts many
//! modules with cross-module symbol resolution, which is what REPL-style use and
//! the eventual TPDE / lazy-materialization paths require.

use std::path::{Path, PathBuf};

pub mod orc;

pub use orc::{OptLevel, OrcJit};

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
