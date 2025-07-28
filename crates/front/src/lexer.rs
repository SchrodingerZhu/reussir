use std::convert::identity;
use std::num;
use std::num::NonZeroU8;

use chumsky::input::{Input, Stream, ValueInput};
use lexical::parse_integer_options::Options as ParseIntegerOptions;
use lexical::{NumberFormatBuilder, parse_with_options};
use logos::{Lexer, Logos};
use reussir_core::Location;
use reussir_core::literal::{FloatLiteral, IntegerLiteral};
use reussir_core::types::Primitive;

use ustr::Ustr;

fn parse_float<'a>(s: &mut Lexer<'a, Token<'a>>) -> Result<FloatLiteral, Error> {
    let s = s.slice();
    (if let Some(s) = s.strip_suffix("bf16") {
        s.parse().map(FloatLiteral::BF16)
    } else if let Some(s) = s.strip_suffix("f16") {
        s.parse().map(FloatLiteral::F16)
    } else if let Some(s) = s.strip_suffix("f64") {
        s.parse().map(FloatLiteral::F64)
    } else if let Some(s) = s.strip_suffix("f128") {
        s.parse().map(FloatLiteral::F128)
    } else {
        // on default, assume f32
        s.trim_end_matches("f32").parse().map(FloatLiteral::F32)
    })
    .map_err(|e| Error::InvalidFloatLiteral(e.0))
}

const DEFAULT_FORMAT_BUILDER: NumberFormatBuilder = NumberFormatBuilder::new()
    .digit_separator(num::NonZeroU8::new(b'_'))
    .required_digits(true)
    .no_positive_mantissa_sign(true)
    .no_special(true)
    .internal_digit_separator(true)
    .trailing_digit_separator(true)
    .consecutive_digit_separator(true);

const DECIMAL_FORMAT: u128 = DEFAULT_FORMAT_BUILDER.radix(10).build();

const HEXADECIMAL_FORMAT: u128 = DEFAULT_FORMAT_BUILDER
    .radix(16)
    .base_prefix(NonZeroU8::new(b'x'))
    .build();

const BINARY_FORMAT: u128 = DEFAULT_FORMAT_BUILDER
    .radix(2)
    .base_prefix(NonZeroU8::new(b'b'))
    .build();

const OCTAL_FORMAT: u128 = DEFAULT_FORMAT_BUILDER
    .radix(8)
    .base_prefix(NonZeroU8::new(b'o'))
    .build();

fn parse_integer<'s, const FORMAT: u128>(
    input: &mut Lexer<'s, Token<'s>>,
) -> Result<IntegerLiteral, Error> {
    let input = input.slice();
    let options = ParseIntegerOptions::new();
    let res = if let Some(s) = input.strip_suffix("i8") {
        IntegerLiteral::I8(
            parse_with_options::<i8, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("i16") {
        IntegerLiteral::I16(
            parse_with_options::<i16, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("i64") {
        IntegerLiteral::I64(
            parse_with_options::<i64, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("i128") {
        IntegerLiteral::I128(
            parse_with_options::<i128, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("u8") {
        IntegerLiteral::U8(
            parse_with_options::<u8, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("u16") {
        IntegerLiteral::U16(
            parse_with_options::<u16, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("u32") {
        IntegerLiteral::U32(
            parse_with_options::<u32, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("u64") {
        IntegerLiteral::U64(
            parse_with_options::<u64, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("u128") {
        IntegerLiteral::U128(
            parse_with_options::<u128, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("usize") {
        IntegerLiteral::Usize(
            parse_with_options::<usize, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else if let Some(s) = input.strip_suffix("isize") {
        IntegerLiteral::Isize(
            parse_with_options::<isize, _, FORMAT>(&s, &options).map_err(Error::InvalidInteger)?,
        )
    } else {
        IntegerLiteral::I32(
            parse_with_options::<i32, _, FORMAT>(&input.trim_end_matches("i32"), &options)
                .map_err(Error::InvalidInteger)?,
        )
    };
    Ok(res)
}

pub fn parse_primitive_keyword<'s>(input: &mut Lexer<'s, Token<'s>>) -> Result<Primitive, Error> {
    input.slice().parse().map_err(Error::InvalidKeyword)
}

#[derive(thiserror::Error, Debug, Clone, Default, PartialEq)]
pub enum Error {
    #[default]
    #[error("Unknown token encountered")]
    UnknownToken,
    #[error("failed to parse integer: {0}")]
    InvalidInteger(lexical::Error),
    #[error("failed to parse floating point: {0}")]
    InvalidFloatLiteral(&'static str),
    #[error("failed to parse keyword: {0}")]
    InvalidKeyword(strum::ParseError),
}

#[derive(Debug, Clone, PartialEq, Logos)]
#[logos(error = Error)]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"/\*(?:[^*]|\*[^/])*\*/")]
pub enum Token<'src> {
    #[regex(r"\p{XID_Start}\p{XID_Continue}*")]
    Ident(&'src str),

    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("opaque")]
    Opaque,

    #[token("<")]
    LAngle,
    #[token(">")]
    RAngle,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token(".")]
    Dot,
    #[token("::")]
    PathSep,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("=")]
    Eq,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LessEq,
    #[token(">=")]
    GreaterEq,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("|")]
    Or,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("^")]
    Caret,
    #[token("<<")]
    ShiftLeft,
    #[token(">>")]
    ShiftRight,
    #[token("~")]
    Tilde,
    #[token("&")]
    Ampersand,
    #[token("?")]
    Question,
    #[token("!")]
    Bang,
    #[token("*")]
    Asterisk,
    #[token("@")]
    At,
    #[token("let")]
    Let,
    #[token("if")]
    If,
    #[token("cond")]
    Cond,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("()")]
    Unit,
    #[token("_")]
    Underscore,
    #[token("..")]
    Ellipsis,
    #[token("fn")]
    Fn,
    #[token("pub")]
    Pub,
    #[token("reg")]
    Region,
    #[token("as")]
    As,
    #[token("return")]
    Return,
    #[token("yield")]
    Yield,

    #[token("i8", parse_primitive_keyword)]
    #[token("i16", parse_primitive_keyword)]
    #[token("i32", parse_primitive_keyword)]
    #[token("i64", parse_primitive_keyword)]
    #[token("i128", parse_primitive_keyword)]
    #[token("isize", parse_primitive_keyword)]
    #[token("u8", parse_primitive_keyword)]
    #[token("u16", parse_primitive_keyword)]
    #[token("u32", parse_primitive_keyword)]
    #[token("u64", parse_primitive_keyword)]
    #[token("u128", parse_primitive_keyword)]
    #[token("usize", parse_primitive_keyword)]
    #[token("bf16", parse_primitive_keyword)]
    #[token("f16", parse_primitive_keyword)]
    #[token("f32", parse_primitive_keyword)]
    #[token("f64", parse_primitive_keyword)]
    #[token("f128", parse_primitive_keyword)]
    #[token("char", parse_primitive_keyword)]
    #[token("str", parse_primitive_keyword)]
    #[token("bool", parse_primitive_keyword)]
    Primitive(Primitive),

    #[token("true", |_|true)]
    #[token("false", |_|false)]
    Boolean(bool),

    #[regex(r"-?\d[\d_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize)?", parse_integer::<DECIMAL_FORMAT>)]
    #[regex(r"-?0b[01][01_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize)?", parse_integer::<BINARY_FORMAT>)]
    #[regex(r"-?0x[0-9a-fA-F][0-9a-fA-F_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize)?", parse_integer::<HEXADECIMAL_FORMAT>)]
    #[regex(r"-?0o[0-7][0-7_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize)?", parse_integer::<OCTAL_FORMAT>)]
    Integer(IntegerLiteral),

    #[regex(
        r"[+\-]?([\d]+(\.\d*)|[\d]+(\.\d*)?([eE][+\-]?\d+))(bf16|f16|f32|f64|f128)?",
        parse_float
    )]
    Float(FloatLiteral),
    Error(Box<Error>),
}

impl Token<'_> {
    pub fn stream<'a>(
        file: Ustr,
        src: &'a str,
    ) -> impl ValueInput<'a, Token = Token<'a>, Span = Location> {
        let iter = Token::lexer(src)
            .spanned()
            .map(move |(res, range)| match res {
                Ok(tk) => (
                    tk,
                    Location::new(file, (range.start as u32, range.end as u32)),
                ),
                Err(err) => (
                    Token::Error(Box::new(err)),
                    Location::new(file, (range.start as u32, range.end as u32)),
                ),
            });
        Stream::from_iter(iter).map(Location::new(file, (0, src.len() as u32)), identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_streaming() {
        let src = "struct Foo { bar: i32 }";
        let lexer = Token::lexer(src);
        let tokens: Vec<_> = lexer.collect();
        println!("{:?}", tokens);
        assert_eq!(tokens.len(), 7);
    }

    #[test]
    fn test_float_literal_parsing() {
        use rustc_apfloat::ieee::IeeeFloat;
        use std::str::FromStr;
        macro_rules! assert_float {
            ($([$input:expr, $ty:ident, $raw:expr],)+) => {
                $(
                    let mut lexer = Token::lexer($input);
                    let token = lexer.next().unwrap().unwrap();
                    assert_eq!(token, Token::Float(reussir_core::literal::FloatLiteral::$ty(IeeeFloat::from_str(stringify!($raw)).unwrap())));
                )+
            };
        }
        assert_float! {
            ["1.0", F32, 1.0],
            ["-3.14", F32, -3.14],
            ["0.618f32", F32, 0.618],
            ["2.718f64", F64, 2.718],
            ["1.0f16", F16, 1.0],
            ["1.0bf16", BF16, 1.0],
            ["1.0E-13f128", F128, 1.0E-13],
            ["1.0f64", F64, 1.0],
            ["-1.0f128", F128, -1.0],
            ["3.4028235E38f32", F32, 3.4028235E38],
            ["1.7976931348623157E308f64", F64, 1.7976931348623157E308],
            ["3402823466385288598117041123.f128", F128, 3402823466385288598117041123],
        }
    }

    #[test]
    fn test_integer_literal_parsing() {
        use IntegerLiteral::*;
        macro_rules! assert_integer {
            ($([$input:expr, $expected:expr]),+) => {
                $(
                    let mut lexer = Token::lexer($input);
                    let token = lexer.next().unwrap().unwrap();
                    assert_eq!(token, Token::Integer($expected));
                )+
            };

        }
        assert_integer! {
            ["42", I32(42)],
            ["-42i8", I8(-42)],
            ["0b1010u16", U16(10)],
            ["0x1F4i64", I64(500)],
            ["0o377u8", U8(255)],
            ["12345678901234567890u128", U128(12345678901234567890)],
            ["-12345678901234567890i128", I128(-12345678901234567890)]
        }
    }
}
