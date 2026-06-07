use logos::{Lexer as LogosLexer, Logos};
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LexicalError {
    pub start: usize,
    pub end: usize,
}

fn parse_i64<'input>(lex: &mut LogosLexer<'input, Tok<'input>>) -> Option<&'input str> {
    lexical_core::parse::<i64>(lex.slice().as_bytes()).ok()?;
    Some(lex.slice())
}

fn parse_f64<'input>(lex: &mut LogosLexer<'input, Tok<'input>>) -> Option<&'input str> {
    lexical_core::parse::<f64>(lex.slice().as_bytes()).ok()?;
    Some(lex.slice())
}

#[derive(Clone, Debug, PartialEq, Logos)]
pub enum Tok<'input> {
    #[regex(r"[ \t\n\f\r]+", |lex| lex.slice())]
    Whitespace(&'input str),
    #[regex(r"//[^\n\r]*", |lex| lex.slice())]
    LineComment(&'input str),
    #[regex(r"/\*([^*]|\*[^/])*\*/", |lex| lex.slice())]
    BlockComment(&'input str),

    #[token("pub")]
    Pub,
    #[token("fn")]
    Fn,
    #[token("regional")]
    Regional,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("mod")]
    Mod,
    #[token("extern")]
    Extern,
    #[token("trampoline")]
    Trampoline,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("let")]
    Let,
    #[token("match")]
    Match,
    #[token("as")]
    As,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("shared")]
    Shared,
    #[token("value")]
    Value,
    #[token("flex")]
    Flex,
    #[token("rigid")]
    Rigid,
    #[token("field")]
    Field,

    #[token("::")]
    DoubleColon,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token(":=")]
    Assign,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    Lte,
    #[token(">=")]
    Gte,
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
    Lt,
    #[token(">")]
    Gt,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
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

    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice())]
    String(&'input str),
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", parse_f64)]
    Double(&'input str),
    #[regex(r"[0-9]+[eE][+-]?[0-9]+", parse_f64)]
    Scientific(&'input str),
    #[regex(r"[0-9]+", parse_i64)]
    Int(&'input str),
    #[regex(r"[A-Za-z][A-Za-z0-9_]*", |lex| lex.slice())]
    Ident(&'input str),
}

impl<'input> Tok<'input> {
    pub fn text(&self) -> &'input str {
        match self {
            Tok::Whitespace(text)
            | Tok::LineComment(text)
            | Tok::BlockComment(text)
            | Tok::String(text)
            | Tok::Double(text)
            | Tok::Scientific(text)
            | Tok::Int(text)
            | Tok::Ident(text) => text,
            Tok::Pub => "pub",
            Tok::Fn => "fn",
            Tok::Regional => "regional",
            Tok::Struct => "struct",
            Tok::Enum => "enum",
            Tok::Mod => "mod",
            Tok::Extern => "extern",
            Tok::Trampoline => "trampoline",
            Tok::If => "if",
            Tok::Else => "else",
            Tok::Let => "let",
            Tok::Match => "match",
            Tok::As => "as",
            Tok::True => "true",
            Tok::False => "false",
            Tok::Shared => "shared",
            Tok::Value => "value",
            Tok::Flex => "flex",
            Tok::Rigid => "rigid",
            Tok::Field => "field",
            Tok::DoubleColon => "::",
            Tok::Arrow => "->",
            Tok::FatArrow => "=>",
            Tok::Assign => ":=",
            Tok::EqEq => "==",
            Tok::BangEq => "!=",
            Tok::Lte => "<=",
            Tok::Gte => ">=",
            Tok::AmpAmp => "&&",
            Tok::PipePipe => "||",
            Tok::DotDot => "..",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Lt => "<",
            Tok::Gt => ">",
            Tok::Colon => ":",
            Tok::Semi => ";",
            Tok::Comma => ",",
            Tok::Dot => ".",
            Tok::Eq => "=",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::Percent => "%",
            Tok::Bang => "!",
            Tok::Pipe => "|",
            Tok::Underscore => "_",
        }
    }

    pub fn is_trivia(&self) -> bool {
        matches!(
            self,
            Tok::Whitespace(_) | Tok::LineComment(_) | Tok::BlockComment(_)
        )
    }
}

pub fn lex_lossless(
    input: &str,
) -> impl Iterator<Item = Result<(usize, Tok<'_>, usize), LexicalError>> {
    Tok::lexer(input)
        .spanned()
        .map(|(tok, Range { start, end })| match tok {
            Ok(tok) => Ok((start, tok, end)),
            Err(()) => Err(LexicalError { start, end }),
        })
}

pub fn unquote_string(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw[1..raw.len() - 1].chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}
