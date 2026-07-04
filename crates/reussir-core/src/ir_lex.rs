//! Logos lexer for the textual IR, shared by both grammars (`full/mir/grammar`
//! and `semi/hir/grammar`) via their `extern` token declarations.
//!
//! The IR is machine-emitted, so the lexer is deliberately small and the error
//! path is coarse (a lex failure means a printer bug or corrupt input, not user
//! source). Identifiers borrow the input; widths/values are parsed eagerly.

use logos::Logos;

use crate::literal::Integer;

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
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("value")]
    Value,
    #[token("shared")]
    Shared,
    #[token("field")]
    Field,
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
    #[token("apply")]
    Apply,
    #[token("flex")]
    Flex,
    #[token("rigid")]
    Rigid,
    #[token("proj")]
    Proj,
    #[token("assign")]
    Assign,
    #[token("scrut")]
    Scrut,
    #[token("uncovered")]
    Uncovered,
    #[token("unreachable")]
    Unreachable,
    #[token("Nullable")]
    Nullable,
    #[token("NonNull")]
    NonNull,
    #[token("Null")]
    Null,
    #[token("poison")]
    Poison,
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
    /// The span-range separator (`@start..end`). Longest-match beats `Dot`,
    /// and the float regex requires a digit after its dot, so `12..34` lexes
    /// `Int(12) DotDot Int(34)`.
    #[token("..")]
    DotDot,
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
    /// `$<id>` — a generic parameter (HIR only).
    #[regex(r"\$[0-9]+", |l| l.slice()[1..].parse().ok())]
    Generic(u32),
    /// `?<id>` — an unsolved inference hole (HIR only).
    #[regex(r"\?[0-9]+", |l| l.slice()[1..].parse().ok())]
    Hole(u32),
    /// An integer literal, arbitrary precision (constants print at full
    /// width). A leading `-` is part of the token when directly adjacent —
    /// negative constants exist in MIR via literal-negation folding, and the
    /// printer always spaces binary operators (`a - 1`), so adjacency is
    /// unambiguous in machine-emitted text.
    #[regex(r"-?[0-9]+", |l| l.slice().parse().ok())]
    Int(Integer),
    /// A floating-point literal, kept as its raw text: the grammar parses it
    /// into an exact [`crate::literal::FloatLit`] where a float value is
    /// expected, and *splits* it where it is really two adjacent scrutinee
    /// path indices (`scrut.0.1` lexes `0.1` as one float-shaped token).
    #[regex(r"-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |l| l.slice())]
    #[regex(r"-?[0-9]+[eE][+-]?[0-9]+", |l| l.slice())]
    Float(&'a str),
    /// A bare identifier (symbol bodies after `@`, source names, paths). Matches
    /// Unicode XID like the surface lexer, so source identifiers round-trip.
    #[regex(r"[\p{XID_Start}_]\p{XID_Continue}*", |l| l.slice())]
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
