use crate::{parse_expr_json, parse_program_json, parse_stmt_json, parse_type_json};
use std::ffi::{CStr, CString, c_char};

unsafe fn parse_c_str<'a>(input: *const c_char) -> Option<&'a str> {
    if input.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(input) }.to_str().ok()
}

unsafe fn parse_file_name<'a>(file_name: *const c_char) -> &'a str {
    if file_name.is_null() {
        return "<input>";
    }
    unsafe { CStr::from_ptr(file_name) }
        .to_str()
        .unwrap_or("<input>")
}

fn into_c_string(value: String) -> *mut c_char {
    CString::new(value)
        .expect("syntax JSON/diagnostics must not contain interior NUL")
        .into_raw()
}

/// Parse a full Reussir program and return a heap-allocated JSON response.
///
/// # Safety
///
/// `input` must point to a valid NUL-terminated C string for the duration of
/// the call. `file_name` may be null; otherwise it must also point to a valid
/// NUL-terminated C string. The returned pointer must be released with
/// [`reussir_syntax_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reussir_syntax_parse_program_json(
    input: *const c_char,
    file_name: *const c_char,
) -> *mut c_char {
    let Some(input) = (unsafe { parse_c_str(input) }) else {
        return into_c_string(r#"{"ok":false,"diagnostic":"invalid UTF-8 input"}"#.to_string());
    };
    into_c_string(parse_program_json(input, unsafe {
        parse_file_name(file_name)
    }))
}

/// Parse a single Reussir statement and return a heap-allocated JSON response.
///
/// # Safety
///
/// `input` must point to a valid NUL-terminated C string for the duration of
/// the call. `file_name` may be null; otherwise it must also point to a valid
/// NUL-terminated C string. The returned pointer must be released with
/// [`reussir_syntax_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reussir_syntax_parse_stmt_json(
    input: *const c_char,
    file_name: *const c_char,
) -> *mut c_char {
    let Some(input) = (unsafe { parse_c_str(input) }) else {
        return into_c_string(r#"{"ok":false,"diagnostic":"invalid UTF-8 input"}"#.to_string());
    };
    into_c_string(parse_stmt_json(input, unsafe {
        parse_file_name(file_name)
    }))
}

/// Parse a single Reussir expression and return a heap-allocated JSON response.
///
/// # Safety
///
/// `input` must point to a valid NUL-terminated C string for the duration of
/// the call. `file_name` may be null; otherwise it must also point to a valid
/// NUL-terminated C string. The returned pointer must be released with
/// [`reussir_syntax_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reussir_syntax_parse_expr_json(
    input: *const c_char,
    file_name: *const c_char,
) -> *mut c_char {
    let Some(input) = (unsafe { parse_c_str(input) }) else {
        return into_c_string(r#"{"ok":false,"diagnostic":"invalid UTF-8 input"}"#.to_string());
    };
    into_c_string(parse_expr_json(input, unsafe {
        parse_file_name(file_name)
    }))
}

/// Parse a single Reussir type and return a heap-allocated JSON response.
///
/// # Safety
///
/// `input` must point to a valid NUL-terminated C string for the duration of
/// the call. `file_name` may be null; otherwise it must also point to a valid
/// NUL-terminated C string. The returned pointer must be released with
/// [`reussir_syntax_string_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reussir_syntax_parse_type_json(
    input: *const c_char,
    file_name: *const c_char,
) -> *mut c_char {
    let Some(input) = (unsafe { parse_c_str(input) }) else {
        return into_c_string(r#"{"ok":false,"diagnostic":"invalid UTF-8 input"}"#.to_string());
    };
    into_c_string(parse_type_json(input, unsafe {
        parse_file_name(file_name)
    }))
}

/// Free a string returned by one of the `reussir_syntax_parse_*_json` exports.
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by this library through
/// `CString::into_raw`. Passing any other pointer, or freeing the same pointer
/// more than once, is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reussir_syntax_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
