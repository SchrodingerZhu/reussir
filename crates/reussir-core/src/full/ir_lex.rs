//! Logos lexer for the textual Full/Semi IR. The token stream feeds the lalrpop
//! grammar (`full/ir.lalrpop`) via its `extern` token declaration.
//!
//! The IR is machine-emitted, so the lexer is deliberately small and the error
//! path is coarse (a lex failure means a printer bug or corrupt input, not user
//! source). Identifiers borrow the input; widths/values are parsed eagerly.

use logos::Logos;

/// A lexed token. `'a` is the input lifetime ([`Token::Ident`] borrows it).
#[derive(Logos, Clone, Debug, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
pub enum Token<'a> {
    // ----- keywords -----
    #[token("pub")]
    Pub,
    #[token("regional")]
    Regional,
    #[token("fn")]
    Fn,
    #[token("record")]
    Record,
    #[token("extern")]
    Extern,
    #[token("trampoline")]
    Trampoline,
    #[token("let")]
    Let,
    #[token("in")]
    In,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("switch")]
    Switch,
    #[token("region")]
    Region,
    #[token("as")]
    As,
    #[token("closure")]
    Closure,
    #[token("Nullable")]
    Nullable,
    #[token("NonNull")]
    NonNull,
    #[token("Null")]
    Null,
    #[token("true")]
    True,
    #[token("false")]
    False,

    // ----- type keywords -----
    #[token("bool")]
    Bool,
    #[token("str")]
    Str,
    #[token("bf16")]
    BF16,
    #[token("f8")]
    F8,
    /// `i<width>` — a signed integer type.
    #[regex(r"i[0-9]+", |l| l.slice()[1..].parse().ok(), priority = 3)]
    IntS(u16),
    /// `u<width>` — an unsigned integer type.
    #[regex(r"u[0-9]+", |l| l.slice()[1..].parse().ok(), priority = 3)]
    IntU(u16),
    /// `f<width>` — an IEEE float type (16/32/64); `f8` is the `F8` keyword.
    #[regex(r"f(16|32|64)", |l| l.slice()[1..].parse().ok(), priority = 3)]
    Fp(u16),

    // ----- punctuation -----
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
    #[token("::")]
    ColonColon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("=")]
    Eq,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("_", priority = 3)]
    Underscore,

    // ----- operators -----
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
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token("!")]
    Bang,

    // ----- literals & identifiers -----
    /// `v<id>` — a local variable.
    #[regex(r"v[0-9]+", |l| l.slice()[1..].parse().ok(), priority = 3)]
    Var(u32),
    /// A non-negative integer literal.
    #[regex(r"[0-9]+", |l| l.slice().parse().ok())]
    Int(i128),
    /// A floating-point literal.
    #[regex(r"[0-9]+\.[0-9]+", |l| l.slice().parse().ok())]
    Float(f64),
    /// A bare identifier (symbol bodies after `@`, source names, paths).
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |l| l.slice())]
    Ident(&'a str),
    /// A double-quoted string (trampoline abi / export names).
    #[regex(r#""[^"]*""#, |l| { let s = l.slice(); &s[1..s.len() - 1] })]
    Quoted(&'a str),
}

/// A lexing failure: the byte span that did not match any token.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct LexError {
    pub span: std::ops::Range<usize>,
}

/// Lex `input` into the `(start, token, end)` triples lalrpop consumes.
pub fn lex(input: &str) -> impl Iterator<Item = Result<(usize, Token<'_>, usize), LexError>> {
    Token::lexer(input).spanned().map(|(tok, span)| match tok {
        Ok(t) => Ok((span.start, t, span.end)),
        Err(()) => Err(LexError { span }),
    })
}
