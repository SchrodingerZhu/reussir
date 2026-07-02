//! The Reussir REPL: incremental elaboration over a persistent HIR with
//! JIT execution.
//!
//! The session ([`session::ReplSession`]) keeps, for its whole lifetime:
//! a shared token interner (so `TokenKey`s are stable across the many small
//! parses), one persistent [`Elaborator`](reussir_core::semi::Elaborator)
//! (the accumulated HIR — extended atomically per input, duplicates
//! rejected), and one [`OrcJit`](reussir_jit::OrcJit) (modules added
//! incrementally, resolved against one another).
//!
//! Per input the pipeline is: `parse_repl` → `try_extend` /
//! `try_repl_expr` → `monomorphize` the accumulated program →
//! [`externalize`](externalize::externalize) already-JIT'd symbols to
//! declarations → lower → verify → lowering pipeline → tracked JIT add →
//! (for expressions) look up the `__repl_expr_N` trampoline, call it by its
//! C ABI class, and render the result. Any failure rolls the elaborator
//! back before anything reaches the JIT, so the session never holds state
//! the JIT doesn't (and vice versa).
//!
//! Frontends (the plain script/pipe mode, the ratatui TUI) drive
//! [`session::run`] and render [`session::Outcome`]s; the command
//! dispatcher is shared, so `:help` and friends behave identically
//! everywhere.

pub mod commands;
pub mod externalize;
pub mod frontend;
pub mod session;
pub mod value;
