//! Input-completeness detection for smart-Enter submission.
//!
//! The TUI submits on Enter only when the buffer looks like a complete
//! input; otherwise Enter inserts a newline and the user keeps typing. An
//! input is *incomplete* when its delimiters are unbalanced-open or its last
//! lexical element is unterminated — the cheap, purely lexical part of the
//! judgment. (A balanced-but-ill-formed input counts as complete: submitting
//! surfaces the parse error, which beats trapping the user in multiline
//! mode.)

use reussir_syntax::kind::SyntaxKind;
use reussir_syntax::lexer;

/// Would `source` plausibly continue on the next line?
pub fn is_incomplete(source: &str) -> bool {
    let (tokens, errors) = lexer::tokenize(source);

    let mut depth = 0i64;
    for token in &tokens {
        match token.kind {
            SyntaxKind::LBrace | SyntaxKind::LParen | SyntaxKind::LBracket => depth += 1,
            SyntaxKind::RBrace | SyntaxKind::RParen | SyntaxKind::RBracket => depth -= 1,
            _ => {}
        }
    }
    if depth > 0 {
        return true;
    }

    // An unterminated string or block comment lexes as an error whose span
    // reaches the end of the input.
    let end = source.len() as u32;
    errors.iter().any(|e| e.span.1 >= end)
}

#[cfg(test)]
mod tests {
    use super::is_incomplete;

    #[test]
    fn open_delimiters_are_incomplete() {
        assert!(is_incomplete("fn f(x: i64) -> i64 {"));
        assert!(is_incomplete("f(1,"));
        assert!(is_incomplete("match x {\n  1 => 2,"));
        assert!(is_incomplete("/* comment"));
    }

    #[test]
    fn balanced_inputs_are_complete() {
        assert!(!is_incomplete("1 + 1"));
        assert!(!is_incomplete("fn f(x: i64) -> i64 { x }"));
        assert!(!is_incomplete("let a = 1; a"));
        // Balanced but ill-formed still submits (surfacing the error).
        assert!(!is_incomplete("fn f() -> }{"));
        // Over-closed submits too.
        assert!(!is_incomplete("f(1))"));
    }
}
