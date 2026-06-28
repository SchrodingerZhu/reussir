//! Shared helpers for this crate's tests.
//!
//! The same file backs both the in-crate unit tests (via `mod testing` under
//! `#[cfg(test)]`) and the external integration tests (via `#[path]`), so every
//! test obtains its backend context the same way: through [`context`], which also
//! installs the `tracing` subscriber. Lowering emits `tracing` events as it builds
//! the IR, so running a test with `--nocapture` shows exactly what it produced.

/// A fresh backend [`Context`](reussir_backend::melior::Context) with the
/// `tracing` subscriber installed at `DEBUG` level.
///
/// The subscriber is process-global, so it is installed at most once: a second
/// call (the next test in the same binary) finds it already set and leaves it in
/// place.
pub fn context() -> reussir_backend::melior::Context {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
    reussir_backend::context()
}
