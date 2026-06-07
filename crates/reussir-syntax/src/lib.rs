pub mod ast;
mod ast_parse;
mod diagnostic;
pub mod ffi;
pub mod lexer;
pub mod syntax;

use ast_parse::{parse_expr_ast, parse_program_ast, parse_stmt_ast, parse_type_ast};
use cstree::Syntax;
use diagnostic::render_syntax_errors;
use serde::Serialize;
use syntax::{SyntaxKind, parse_cst};

pub use syntax::{CstParse, SyntaxError};

#[derive(Debug, Serialize)]
pub struct ParseSummary {
    pub root_kind: String,
    pub text_len: usize,
    pub error_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ParseResponse<T: Serialize> {
    Ok { ok: bool, value: T },
    Err { ok: bool, diagnostic: String },
}

fn encode_response<T: Serialize>(response: ParseResponse<T>) -> String {
    serde_json::to_string(&response).expect("syntax response serialization failed")
}

pub fn parse_source(input: &str) -> CstParse {
    parse_cst(input)
}

pub fn parse_source_json(input: &str, file_name: &str) -> String {
    let parsed = parse_cst(input);
    if parsed.errors.is_empty() {
        encode_response(ParseResponse::Ok {
            ok: true,
            value: ParseSummary {
                root_kind: format!("{:?}", SyntaxKind::from_raw(parsed.green.kind())),
                text_len: parsed.green.text_len().into(),
                error_count: 0,
            },
        })
    } else {
        encode_response(ParseResponse::<ParseSummary>::Err {
            ok: false,
            diagnostic: render_syntax_errors(file_name, input, &parsed.errors),
        })
    }
}

pub fn parse_program_json(input: &str, file_name: &str) -> String {
    match parse_program_ast(input) {
        Ok(value) => encode_response(ParseResponse::Ok { ok: true, value }),
        Err(errors) => encode_response(ParseResponse::<ast::Program>::Err {
            ok: false,
            diagnostic: render_syntax_errors(file_name, input, &errors),
        }),
    }
}

pub fn parse_stmt_json(input: &str, file_name: &str) -> String {
    match parse_stmt_ast(input) {
        Ok(value) => encode_response(ParseResponse::Ok { ok: true, value }),
        Err(errors) => encode_response(ParseResponse::<ast::Stmt>::Err {
            ok: false,
            diagnostic: render_syntax_errors(file_name, input, &errors),
        }),
    }
}

pub fn parse_expr_json(input: &str, file_name: &str) -> String {
    match parse_expr_ast(input) {
        Ok(value) => encode_response(ParseResponse::Ok { ok: true, value }),
        Err(errors) => encode_response(ParseResponse::<ast::Expr>::Err {
            ok: false,
            diagnostic: render_syntax_errors(file_name, input, &errors),
        }),
    }
}

pub fn parse_type_json(input: &str, file_name: &str) -> String {
    match parse_type_ast(input) {
        Ok(value) => encode_response(ParseResponse::Ok { ok: true, value }),
        Err(errors) => encode_response(ParseResponse::<ast::Type>::Err {
            ok: false,
            diagnostic: render_syntax_errors(file_name, input, &errors),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cstree::Syntax;

    #[test]
    fn cst_preserves_trivia_and_source_length() {
        let src = "// lead\nfn id(x: i32) -> i32 { x }\n";
        let parsed = parse_source(src);
        assert!(
            parsed.errors.is_empty(),
            "unexpected errors: {:?}",
            parsed.errors
        );
        assert_eq!(SyntaxKind::from_raw(parsed.green.kind()), SyntaxKind::Root);
        let text_len: usize = parsed.green.text_len().into();
        assert_eq!(text_len, src.len());
    }

    #[test]
    fn parser_reports_recoverable_syntax_errors() {
        let parsed = parse_source("fn incomplete(");
        assert!(!parsed.errors.is_empty());
        assert_eq!(SyntaxKind::from_raw(parsed.green.kind()), SyntaxKind::Root);
    }

    #[test]
    fn json_uses_ariadne_diagnostics_on_error() {
        let json = parse_program_json("fn incomplete(", "bad.rr");
        assert!(json.contains(r#""ok":false"#));
        assert!(json.contains("bad.rr"));
        assert!(json.contains("syntax error"));
    }

    #[test]
    fn program_json_serializes_haskell_shaped_ast() {
        let json = parse_program_json("pub fn id(x: i32) -> i32 { x }", "id.rr");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains("function"), "{json}");
        assert!(json.contains("id"), "{json}");
        assert!(json.contains("integral"), "{json}");
    }

    #[test]
    fn program_json_parses_match_scrutinee_without_ctor_ambiguity() {
        let src = "fn test(x: i32) -> i32 { match x { _ => 1 } }";
        let json = parse_program_json(src, "match.rr");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains("match"), "{json}");
    }

    #[test]
    fn program_json_parses_if_condition_without_ctor_ambiguity() {
        let src = "fn test(foo: bool) -> i32 { if foo { 1 } else { 0 } }";
        let json = parse_program_json(src, "if.rr");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains(r#""if""#), "{json}");
    }

    #[test]
    fn program_json_keeps_less_than_as_operator_after_path() {
        let src = "fn test(k: i32, kx: i32) -> i32 { if k < kx { 1 } else { 0 } }";
        let json = parse_program_json(src, "less-than.rr");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains("lt"), "{json}");
    }

    #[test]
    fn cst_parses_match_scrutinee_without_ctor_ambiguity() {
        let src = "fn test(x: i32) -> i32 { match x { _ => 1 } }";
        let parsed = parse_source(src);
        assert!(
            parsed.errors.is_empty(),
            "unexpected errors: {:?}",
            parsed.errors
        );
    }

    #[test]
    fn type_json_serializes_arrow_type() {
        let json = parse_type_json("i32 -> bool", "type.rr");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains("arrow"), "{json}");
    }

    #[test]
    fn lexical_rejects_integer_overflow_without_dropping_tree() {
        let parsed = parse_source("fn f() { 999999999999999999999999999999999999999999 } ");
        assert!(!parsed.errors.is_empty());
        assert_eq!(SyntaxKind::from_raw(parsed.green.kind()), SyntaxKind::Root);
    }
}
