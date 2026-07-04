//! String and character literal decoding, shared by the lexer and the AST
//! emitter.
//!
//! The surface language follows Rust's literal escape grammar (`\n`, `\xNN`,
//! `\u{...}`, ...), implemented by [`litrs`]. The lexer validates candidate
//! tokens with the fallible [`parse_string`] / [`parse_char`] and the AST
//! emitter decodes accepted tokens with [`unescape_string`] /
//! [`unescape_char`] — the same parse either way, so validation and decoding
//! cannot drift apart.

use std::borrow::Cow;

/// Parse a string literal token (including the surrounding quotes) into its
/// decoded value.
pub fn parse_string(raw: &str) -> Result<Cow<'_, str>, litrs::ParseError> {
    litrs::StringLit::parse(raw).map(litrs::StringLit::into_value)
}

/// Parse a character literal token (including the surrounding quotes) into
/// the single Unicode scalar value it denotes.
pub fn parse_char(raw: &str) -> Result<char, litrs::ParseError> {
    litrs::CharLit::parse(raw).map(|lit| lit.value())
}

/// Decode a string literal token the lexer has already validated.
///
/// Panics on invalid input: the lexer only emits `StringLit` tokens that
/// [`parse_string`] accepts.
pub fn unescape_string(raw: &str) -> String {
    parse_string(raw)
        .expect("lexer-validated string literal")
        .into_owned()
}

/// Decode a character literal token the lexer has already validated.
///
/// Panics on invalid input: the lexer only emits `CharLit` tokens that
/// [`parse_char`] accepts.
pub fn unescape_char(raw: &str) -> char {
    parse_char(raw).expect("lexer-validated character literal")
}
