//! Tokenizer for the Reussir surface language, built on [`logos`].
//!
//! Lexical conventions:
//!
//! * identifiers follow Unicode XID rules (`XID_Start` then `XID_Continue*`),
//! * `//` line comments and (non-nesting) `/* ... */` block comments,
//! * integers are plain digit runs; floats are digit runs with a fraction
//!   and/or exponent (`1.5`, `1e3`, `2.5e-7`),
//! * strings are double-quoted with Haskell-style escapes (validated here,
//!   decoded in [`crate::ast`]).
//!
//! Keyword policy: only the words that introduce syntactic forms become
//! keyword tokens. Contextual words (`trampoline`, capability names like
//! `shared`/`flex`, primitive type names like `i32`/`bool`) are lexed as
//! identifiers and matched by text in the parser, so they stay usable as
//! ordinary names.

use logos::Logos;
use std::ops::Range;

use crate::kind::SyntaxKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexError {
    pub span: (u32, u32),
    pub message: &'static str,
}

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawToken {
    #[regex(r"[ \t\r\n\u{0B}\u{0C}]+")]
    Whitespace,
    #[regex(r"//[^\n]*")]
    LineComment,
    #[token("/*", lex_block_comment)]
    BlockComment,

    #[regex(r"\p{XID_Start}\p{XID_Continue}*", classify_ident)]
    Ident(IdentClass),

    #[regex(r"[0-9]+")]
    Int,
    // A float needs a fractional part, an exponent, or both; a bare digit
    // run is an integer.
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?")]
    #[regex(r"[0-9]+[eE][+-]?[0-9]+")]
    Float,

    #[token("\"", lex_string)]
    String,

    #[token("::")]
    PathSep,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token(":=")]
    ColonEq,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("..")]
    DotDot,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("<")]
    LAngle,
    #[token(">")]
    RAngle,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("=")]
    Eq,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("!")]
    Bang,
    #[token("|")]
    Pipe,
    #[token("_")]
    Underscore,
}

/// Pre-classified identifier-or-keyword. Hard keywords get their own class;
/// everything else (including contextual keywords) stays `Plain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentClass {
    Plain,
    Fn,
    Struct,
    Enum,
    Pub,
    Mod,
    Let,
    If,
    Else,
    Match,
    Regional,
    Extern,
    As,
    True,
    False,
}

fn classify_ident(lex: &mut logos::Lexer<RawToken>) -> IdentClass {
    match lex.slice() {
        "fn" => IdentClass::Fn,
        "struct" => IdentClass::Struct,
        "enum" => IdentClass::Enum,
        "pub" => IdentClass::Pub,
        "mod" => IdentClass::Mod,
        "let" => IdentClass::Let,
        "if" => IdentClass::If,
        "else" => IdentClass::Else,
        "match" => IdentClass::Match,
        "regional" => IdentClass::Regional,
        "extern" => IdentClass::Extern,
        "as" => IdentClass::As,
        "true" => IdentClass::True,
        "false" => IdentClass::False,
        _ => IdentClass::Plain,
    }
}

/// Scan a (non-nesting) block comment after the opening `/*`. Returns
/// `false` (filtered error) when the comment is unterminated.
fn lex_block_comment(lex: &mut logos::Lexer<RawToken>) -> bool {
    let rest = lex.remainder();
    match rest.find("*/") {
        Some(idx) => {
            lex.bump(idx + 2);
            true
        }
        None => {
            lex.bump(rest.len());
            false
        }
    }
}

/// Scan a string literal after the opening quote. A backslash always
/// escapes the next character; the literal ends at the first unescaped
/// quote. Returns `false` when the literal is unterminated.
fn lex_string(lex: &mut logos::Lexer<RawToken>) -> bool {
    let rest = lex.remainder();
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                lex.bump(i + 1);
                return true;
            }
            b'\\' => {
                if i + 1 >= bytes.len() {
                    break;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    lex.bump(rest.len());
    false
}

impl RawToken {
    fn syntax_kind(self) -> SyntaxKind {
        match self {
            RawToken::Whitespace => SyntaxKind::Whitespace,
            RawToken::LineComment => SyntaxKind::LineComment,
            RawToken::BlockComment => SyntaxKind::BlockComment,
            RawToken::Ident(class) => match class {
                IdentClass::Plain => SyntaxKind::Ident,
                IdentClass::Fn => SyntaxKind::FnKw,
                IdentClass::Struct => SyntaxKind::StructKw,
                IdentClass::Enum => SyntaxKind::EnumKw,
                IdentClass::Pub => SyntaxKind::PubKw,
                IdentClass::Mod => SyntaxKind::ModKw,
                IdentClass::Let => SyntaxKind::LetKw,
                IdentClass::If => SyntaxKind::IfKw,
                IdentClass::Else => SyntaxKind::ElseKw,
                IdentClass::Match => SyntaxKind::MatchKw,
                IdentClass::Regional => SyntaxKind::RegionalKw,
                IdentClass::Extern => SyntaxKind::ExternKw,
                IdentClass::As => SyntaxKind::AsKw,
                IdentClass::True => SyntaxKind::TrueKw,
                IdentClass::False => SyntaxKind::FalseKw,
            },
            RawToken::Int => SyntaxKind::IntLit,
            RawToken::Float => SyntaxKind::FloatLit,
            RawToken::String => SyntaxKind::StringLit,
            RawToken::PathSep => SyntaxKind::PathSep,
            RawToken::Arrow => SyntaxKind::Arrow,
            RawToken::FatArrow => SyntaxKind::FatArrow,
            RawToken::ColonEq => SyntaxKind::ColonEq,
            RawToken::EqEq => SyntaxKind::EqEq,
            RawToken::BangEq => SyntaxKind::BangEq,
            RawToken::LtEq => SyntaxKind::LtEq,
            RawToken::GtEq => SyntaxKind::GtEq,
            RawToken::AmpAmp => SyntaxKind::AmpAmp,
            RawToken::PipePipe => SyntaxKind::PipePipe,
            RawToken::DotDot => SyntaxKind::DotDot,
            RawToken::LParen => SyntaxKind::LParen,
            RawToken::RParen => SyntaxKind::RParen,
            RawToken::LBrace => SyntaxKind::LBrace,
            RawToken::RBrace => SyntaxKind::RBrace,
            RawToken::LBracket => SyntaxKind::LBracket,
            RawToken::RBracket => SyntaxKind::RBracket,
            RawToken::LAngle => SyntaxKind::LAngle,
            RawToken::RAngle => SyntaxKind::RAngle,
            RawToken::Colon => SyntaxKind::Colon,
            RawToken::Semicolon => SyntaxKind::Semicolon,
            RawToken::Comma => SyntaxKind::Comma,
            RawToken::Dot => SyntaxKind::Dot,
            RawToken::Eq => SyntaxKind::Eq,
            RawToken::Plus => SyntaxKind::Plus,
            RawToken::Minus => SyntaxKind::Minus,
            RawToken::Star => SyntaxKind::Star,
            RawToken::Slash => SyntaxKind::Slash,
            RawToken::Percent => SyntaxKind::Percent,
            RawToken::Bang => SyntaxKind::Bang,
            RawToken::Pipe => SyntaxKind::Pipe,
            RawToken::Underscore => SyntaxKind::Underscore,
        }
    }
}

/// A lexed token: kind plus byte range into the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub kind: SyntaxKind,
    pub range: (u32, u32),
}

impl Token {
    pub fn text<'s>(&self, source: &'s str) -> &'s str {
        &source[self.range.0 as usize..self.range.1 as usize]
    }
}

/// Tokenize the whole source. Lexical errors (unterminated strings or block
/// comments, unrecognized characters) become `ErrorToken`s so the syntax
/// tree stays lossless, and are reported in the error list.
pub fn tokenize(source: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let mut lexer = RawToken::lexer(source);
    while let Some(result) = lexer.next() {
        let Range { start, end } = lexer.span();
        let range = (start as u32, end as u32);
        match result {
            Ok(token) => tokens.push(Token {
                kind: token.syntax_kind(),
                range,
            }),
            Err(()) => {
                let slice = lexer.slice();
                let message = if slice.starts_with('"') {
                    "unterminated string literal"
                } else if slice.starts_with("/*") {
                    "unterminated block comment"
                } else {
                    "unrecognized character"
                };
                errors.push(LexError {
                    span: range,
                    message,
                });
                tokens.push(Token {
                    kind: SyntaxKind::ErrorToken,
                    range,
                });
            }
        }
    }
    (tokens, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::SyntaxKind as K;

    fn kinds(src: &str) -> Vec<SyntaxKind> {
        let (tokens, errors) = tokenize(src);
        assert!(errors.is_empty(), "unexpected lex errors: {errors:?}");
        tokens
            .iter()
            .map(|t| t.kind)
            .filter(|k| !k.is_trivia())
            .collect()
    }

    #[test]
    fn keywords_vs_identifiers() {
        assert_eq!(kinds("fn iffy"), vec![K::FnKw, K::Ident]);
        assert_eq!(kinds("if lettuce"), vec![K::IfKw, K::Ident]);
        // Contextual keywords stay identifiers.
        assert_eq!(kinds("shared trampoline i32 bool"), vec![K::Ident; 4]);
    }

    #[test]
    fn unicode_identifiers() {
        assert_eq!(kinds("变量 αβγ"), vec![K::Ident, K::Ident]);
        // `_` does not start an identifier (XID_Start excludes it).
        assert_eq!(kinds("_x"), vec![K::Underscore, K::Ident]);
    }

    #[test]
    fn numbers() {
        assert_eq!(kinds("42"), vec![K::IntLit]);
        assert_eq!(kinds("1.5"), vec![K::FloatLit]);
        assert_eq!(kinds("1e3 2.5e-7"), vec![K::FloatLit, K::FloatLit]);
        // A dot after an integer is an access, not a float.
        assert_eq!(kinds("1.foo"), vec![K::IntLit, K::Dot, K::Ident]);
    }

    #[test]
    fn compound_punctuation() {
        assert_eq!(
            kinds("a->b := c"),
            vec![K::Ident, K::Arrow, K::Ident, K::ColonEq, K::Ident]
        );
        assert_eq!(kinds("a - b"), vec![K::Ident, K::Minus, K::Ident]);
        assert_eq!(kinds("std::io"), vec![K::Ident, K::PathSep, K::Ident]);
        assert_eq!(kinds(".."), vec![K::DotDot]);
    }

    #[test]
    fn strings_and_comments() {
        assert_eq!(kinds(r#""hello \"world\"""#), vec![K::StringLit]);
        assert_eq!(kinds("a // comment\nb"), vec![K::Ident, K::Ident]);
        assert_eq!(kinds("a /* x ** y */ b"), vec![K::Ident, K::Ident]);
    }

    #[test]
    fn lex_errors_are_recovered() {
        let (tokens, errors) = tokenize("a \"unterminated");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "unterminated string literal");
        assert_eq!(tokens.last().unwrap().kind, K::ErrorToken);
    }
}
