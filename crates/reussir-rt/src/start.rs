//! Starting a Reussir program.
//!
//! A Reussir executable's `main` does nothing but hand the function marked
//! `#[main]` to [`__reussir_start`], which stands in for the wrapper Rust
//! puts around its own `fn main`: it runs the entry point, turns an escaping
//! panic into the exit status Rust uses for one, and flushes what the program
//! wrote before returning to the C runtime.
//!
//! The wrapper lives here rather than in generated code because it is the
//! same for every program, and because it is Rust's runtime — this crate's —
//! that has to be entered: the entry point allocates through the language
//! heap, which is this crate's global allocator.
//!
//! What it deliberately is *not* is `std::rt::lang_start`. That is unstable,
//! and the runtime is built by the user's own toolchain (see
//! docs/design/system-rust-runtime.md), which may be a stable one. In
//! practice std initializes what it needs lazily; the pieces `lang_start`
//! would additionally set up — the main thread's stack guard above all — are
//! not available to a library entered from C.

use std::ffi::c_int;
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};

/// The status Rust's runtime exits with when `main` panics.
const PANIC_EXIT_STATUS: c_int = 101;

/// Runs `entry` as the program's main function and returns the status `main`
/// should hand back to the C runtime: 0 normally, 101 if the entry point
/// panicked.
///
/// `entry` is the C-ABI export of the function marked `#[main]` (the
/// compiler names it `__reussir_main`). It is nullable so that a mistaken
/// call reports rather than jumping through a null pointer.
///
/// # Safety
/// `entry` must be a valid function pointer for the lifetime of the call, or
/// null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_start(entry: Option<extern "C" fn()>) -> c_int {
    let Some(entry) = entry else {
        eprintln!("reussir: __reussir_start was given no entry point");
        return PANIC_EXIT_STATUS;
    };

    // `AssertUnwindSafe`: there is nothing for the entry point to poison —
    // it is the whole program, and nothing observes it after it unwinds
    // except this function, which reports the failure and stops.
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| entry()));

    // std buffers stdout when it is not a terminal, and nothing after this
    // point will flush it: returning from `main` reaches the C runtime, which
    // knows nothing of Rust's buffers. Rust's own shim flushes here for the
    // same reason.
    let _ = std::io::stdout().flush();

    match outcome {
        Ok(()) => 0,
        // The panic itself has already been reported — either by
        // `__reussir_panic`, which aborts before ever reaching here, or by
        // the std panic hook on the way out of `entry`.
        Err(_) => PANIC_EXIT_STATUS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static RUNS: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn entry() {
        RUNS.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn runs_the_entry_point_and_reports_success() {
        let before = RUNS.load(Ordering::SeqCst);
        assert_eq!(unsafe { __reussir_start(Some(entry)) }, 0);
        assert_eq!(RUNS.load(Ordering::SeqCst), before + 1);
    }

    #[test]
    fn a_null_entry_point_is_reported_rather_than_called() {
        assert_eq!(unsafe { __reussir_start(None) }, PANIC_EXIT_STATUS);
    }

    /// An entry point that unwinds yields the status Rust uses for a panicking
    /// `main` instead of unwinding into the C runtime.
    #[test]
    fn a_panicking_entry_point_yields_the_rust_panic_status() {
        extern "C-unwind" fn boom() {
            panic!("entry point failed");
        }
        // The hook is silenced only so the test output stays readable.
        let previous = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let status = unsafe {
            __reussir_start(Some(std::mem::transmute::<
                extern "C-unwind" fn(),
                extern "C" fn(),
            >(boom)))
        };
        panic::set_hook(previous);
        assert_eq!(status, PANIC_EXIT_STATUS);
    }
}
