//! Diagnostic entry points for compiler-inserted instrumentation.
//!
//! `rrc --instrument-nonlinear-ffi` (and `rene`'s `instrument_nonlinear_ffi`
//! profile knob) guard every FFI-import call that consumes an rc'd ffi/array
//! value with a reference-count check; a count other than one calls
//! [`__reussir_report_nonlinear_usage`] with the call's source location.
//! Such a call is the signature of non-linear usage: the boundary consumes
//! its arguments, so a still-shared box forces the foreign side to
//! copy-on-write instead of updating in place.

use core::ffi::{CStr, c_char};

/// Reports a non-linear ffi/array consumption on stderr.
///
/// `file` is a NUL-terminated UTF-8 path (may be null or empty when the call
/// site carries no location); `line` and `col` are 1-based, 0 when unknown.
/// Purely diagnostic: it never aborts and has no effect on program state.
///
/// # Safety
///
/// `file` must be null or point to a NUL-terminated string that stays live
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_report_nonlinear_usage(
    file: *const c_char,
    line: u32,
    col: u32,
) {
    let file = if file.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(file) }.to_string_lossy().into_owned()
    };
    let location = if file.is_empty() {
        "<unknown location>".to_owned()
    } else {
        format!("{file}:{line}:{col}")
    };
    eprintln!(
        "[reussir] non-linear ffi/array usage at {location}: value consumed \
         by an FFI call while its reference count is not 1 (the foreign side \
         will copy instead of updating in place)"
    );
}
