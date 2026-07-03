//! Deduplicate against the JIT session: demote already-materialized symbols
//! to declarations before lowering.
//!
//! Each REPL input re-monomorphizes the whole accumulated program (the v0
//! mangler and mono's sorted output are deterministic, so symbols reproduce
//! exactly). ORC rejects a duplicate definition of an already-materialized
//! symbol, and MLIR requires every callee symbol to be present in the
//! module — so functions already in the JIT are kept as *body-less private
//! declarations* (which `lower_program` emits as external `func.func`
//! declarations), and already-exported trampolines are dropped entirely.
//! Cross-module calls then resolve through the persistent `JITDylib`.
//!
//! Records are untouched: they are layout metadata and emit no symbols. The
//! LinkonceODR glue codegen emits per module (outlined drop/acquire, box
//! vtables, panic globals) deduplicates inside ORC by design.

use rustc_hash::FxHashSet;

use reussir_core::full::mir;
use reussir_core::surface::Visibility;

/// Strip bodies of `emitted` functions (and demote them to private, since a
/// public symbol *declaration* is invalid) and drop `emitted` trampolines.
/// Everything still carrying a body is promoted to public: in the one-module
/// AOT world a non-`pub` function gets internal linkage, but a REPL session
/// spreads the program across many JIT modules, and *later* inputs must be
/// able to resolve every accepted definition.
pub fn externalize(program: &mut mir::Program<'_>, emitted: &FxHashSet<String>) {
    // Resolve names first: `Program::symbol` borrows the program the mutable
    // walk would lock.
    let already: Vec<bool> = program
        .functions
        .iter()
        .map(|f| emitted.contains(program.symbol(f.symbol)))
        .collect();
    for (function, already) in program.functions.iter_mut().zip(already) {
        if already {
            function.body = None;
            function.visibility = Visibility::Private;
        } else {
            function.visibility = Visibility::Public;
        }
    }

    let keep: Vec<bool> = program
        .trampolines
        .iter()
        .map(|t| !emitted.contains(program.symbol(t.export)))
        .collect();
    let mut keep = keep.into_iter();
    program
        .trampolines
        .retain(|_| keep.next().expect("one flag per trampoline"));
}
