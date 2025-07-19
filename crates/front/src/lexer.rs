use std::{num, str::FromStr};

use lexical::parse_integer_options::Options as ParseIntegerOptions;
use lexical::{FromLexicalWithOptions, NumberFormatBuilder};
use logos::{Lexer, Logos};
use rustc_apfloat::{
    ParseError as FPError,
    ieee::{BFloat, Double, Half, Quad, Single},
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum FloatLiteral {
    BF16(BFloat),
    F16(Half),
    F32(Single),
    F64(Double),
    F128(Quad),
}

impl FromStr for FloatLiteral {
    type Err = FPError;

    fn from_str(s: &str) -> Result<Self, FPError> {
        if s.ends_with("bf16") {
            s.trim_end_matches("bf16").parse().map(FloatLiteral::BF16)
        } else if s.ends_with("f16") {
            s.trim_end_matches("f16").parse().map(FloatLiteral::F16)
        } else if s.ends_with("f64") {
            s.trim_end_matches("f64").parse().map(FloatLiteral::F64)
        } else if s.ends_with("f128") {
            s.trim_end_matches("f128").parse().map(FloatLiteral::F128)
        } else {
            // on default, assume f32
            s.trim_end_matches("f32").parse().map(FloatLiteral::F32)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntegerLiteral {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
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

const HEXADECIMAL_FORMAT: u128 = DEFAULT_FORMAT_BUILDER.radix(16).build();

const BINARY_FORMAT: u128 = DEFAULT_FORMAT_BUILDER.radix(2).build();

const OCTAL_FORMAT: u128 = DEFAULT_FORMAT_BUILDER.radix(8).build();

impl From<u8> for IntegerLiteral {
    fn from(value: u8) -> Self {
        IntegerLiteral::U8(value)
    }
}

fn parse_integer<
    's,
    const FORMAT: u128,
    T: FromLexicalWithOptions<Options = ParseIntegerOptions>,
>(
    input: &mut Lexer<'s, Token<'s>>,
    prefix: &'static str,
    suffix: &'static str,
    f: fn(T) -> IntegerLiteral,
) -> Result<IntegerLiteral, Error> {
    let options = lexical::parse_integer_options::Options::new();
    let input = input
        .slice()
        .trim_end_matches(suffix)
        .trim_start_matches(prefix);
    let value = lexical::parse_with_options::<T, _, FORMAT>(input, &options)
        .map_err(Error::InvalidInteger)?;
    Ok(f(value))
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
}

#[derive(Debug, Clone, PartialEq, Logos)]
#[logos(error = Error)]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"/\*(?:[^*]|\*[^/])*\*/")]
pub enum Token<'src> {
    #[regex(r"\p{XID_Start}\p{XID_Continue}*")]
    Ident(&'src str),
    #[regex(r"-?\d[\d_]*i8", |lex| {parse_integer::<DECIMAL_FORMAT, i8>(lex, "", "i8", IntegerLiteral::I8)} )]
    #[regex(r"-?\d[\d_]*i16", |lex| {parse_integer::<DECIMAL_FORMAT, i16>(lex, "", "i16", IntegerLiteral::I16)} )]
    #[regex(r"-?\d[\d_]*(i32)?", |lex| {parse_integer::<DECIMAL_FORMAT, i32>(lex, "", "i32", IntegerLiteral::I32)} )]
    #[regex(r"-?\d[\d_]*i64", |lex| {parse_integer::<DECIMAL_FORMAT, i64>(lex, "", "i64", IntegerLiteral::I64)} )]
    #[regex(r"-?\d[\d_]*i128", |lex| {parse_integer::<DECIMAL_FORMAT, i128>(lex, "", "i128", IntegerLiteral::I128)} )]
    #[regex(r"[\d_]+u8", |lex| {parse_integer::<DECIMAL_FORMAT, u8>(lex, "", "u8", IntegerLiteral::U8)} )]
    #[regex(r"[\d_]+u16", |lex| {parse_integer::<DECIMAL_FORMAT, u16>(lex, "", "u16", IntegerLiteral::U16)} )]
    #[regex(r"[\d_]+u32", |lex| {parse_integer::<DECIMAL_FORMAT, u32>(lex, "", "u32", IntegerLiteral::U32)} )]
    #[regex(r"[\d_]+u64", |lex| {parse_integer::<DECIMAL_FORMAT, u64>(lex, "", "u64", IntegerLiteral::U64)} )]
    #[regex(r"[\d_]+u128", |lex| {parse_integer::<DECIMAL_FORMAT, u128>(lex, "", "u128", IntegerLiteral::U128)} )]
    #[regex(r"-?0b[01_]+i8", |lex| {parse_integer::<BINARY_FORMAT, i8>(lex, "0b", "i8", IntegerLiteral::I8)} )]
    #[regex(r"-?0b[01_]+i16", |lex| {parse_integer::<BINARY_FORMAT, i16>(lex, "0b", "i16", IntegerLiteral::I16)} )]
    #[regex(r"-?0b[01_]+(i32)?", |lex| {parse_integer::<BINARY_FORMAT, i32>(lex, "0b", "i32", IntegerLiteral::I32)} )]
    #[regex(r"-?0b[01_]+i64", |lex| {parse_integer::<BINARY_FORMAT, i64>(lex, "0b", "i64", IntegerLiteral::I64)} )]
    #[regex(r"-?0b[01_]+i128", |lex| {parse_integer::<BINARY_FORMAT, i128>(lex, "0b", "i128", IntegerLiteral::I128)} )]
    #[regex(r"0b[01_]+u8", |lex| {parse_integer::<BINARY_FORMAT, u8>(lex, "0b", "u8", IntegerLiteral::U8)} )]
    #[regex(r"0b[01_]+u16", |lex| {parse_integer::<BINARY_FORMAT, u16>(lex, "0b", "u16", IntegerLiteral::U16)} )]
    #[regex(r"0b[01_]+u32", |lex| {parse_integer::<BINARY_FORMAT, u32>(lex, "0b", "u32", IntegerLiteral::U32)} )]
    #[regex(r"0b[01_]+u64", |lex| {parse_integer::<BINARY_FORMAT, u64>(lex, "0b", "u64", IntegerLiteral::U64)} )]
    #[regex(r"0b[01_]+u128", |lex| {parse_integer::<BINARY_FORMAT, u128>(lex, "0b", "u128", IntegerLiteral::U128)} )]
    #[regex(r"-?0x[0-9a-fA-F_]+i8", |lex| {parse_integer::<HEXADECIMAL_FORMAT, i8>(lex, "0x", "i8", IntegerLiteral::I8)} )]
    #[regex(r"-?0x[0-9a-fA-F_]+i16", |lex| {parse_integer::<HEXADECIMAL_FORMAT, i16>(lex, "0x", "i16", IntegerLiteral::I16)} )]
    #[regex(r"-?0x[0-9a-fA-F_]+(i32)?", |lex| {parse_integer::<HEXADECIMAL_FORMAT, i32>(lex, "0x", "i32", IntegerLiteral::I32)} )]
    #[regex(r"-?0x[0-9a-fA-F_]+i64", |lex| {parse_integer::<HEXADECIMAL_FORMAT, i64>(lex, "0x", "i64", IntegerLiteral::I64)} )]
    #[regex(r"-?0x[0-9a-fA-F_]+i128", |lex| {parse_integer::<HEXADECIMAL_FORMAT, i128>(lex, "0x", "i128", IntegerLiteral::I128)} )]
    #[regex(r"0x[0-9a-fA-F_]+u8", |lex| {parse_integer::<HEXADECIMAL_FORMAT, u8>(lex, "0x", "u8", IntegerLiteral::U8)} )]
    #[regex(r"0x[0-9a-fA-F_]+u16", |lex| {parse_integer::<HEXADECIMAL_FORMAT, u16>(lex, "0x", "u16", IntegerLiteral::U16)} )]
    #[regex(r"0x[0-9a-fA-F_]+u32", |lex| {parse_integer::<HEXADECIMAL_FORMAT, u32>(lex, "0x", "u32", IntegerLiteral::U32)} )]
    #[regex(r"0x[0-9a-fA-F_]+u64", |lex| {parse_integer::<HEXADECIMAL_FORMAT, u64>(lex, "0x", "u64", IntegerLiteral::U64)} )]
    #[regex(r"0x[0-9a-fA-F_]+u128", |lex| {parse_integer::<HEXADECIMAL_FORMAT, u128>(lex, "0x", "u128", IntegerLiteral::U128)} )]
    #[regex(r"-?0o[0-7_]+i8", |lex| {parse_integer::<OCTAL_FORMAT, i8>(lex, "0o", "i8", IntegerLiteral::I8)} )]
    #[regex(r"-?0o[0-7_]+i16", |lex| {parse_integer::<OCTAL_FORMAT, i16>(lex, "0o", "i16", IntegerLiteral::I16)} )]
    #[regex(r"-?0o[0-7_]+(i32)?", |lex| {parse_integer::<OCTAL_FORMAT, i32>(lex, "0o", "i32", IntegerLiteral::I32)} )]
    #[regex(r"-?0o[0-7_]+i64", |lex| {parse_integer::<OCTAL_FORMAT, i64>(lex, "0o", "i64", IntegerLiteral::I64)} )]
    #[regex(r"-?0o[0-7_]+i128", |lex| {parse_integer::<OCTAL_FORMAT, i128>(lex, "0o", "i128", IntegerLiteral::I128)} )]
    #[regex(r"0o[0-7_]+u8", |lex| {parse_integer::<OCTAL_FORMAT, u8>(lex, "0o", "u8", IntegerLiteral::U8)} )]
    #[regex(r"0o[0-7_]+u16", |lex| {parse_integer::<OCTAL_FORMAT, u16>(lex, "0o", "u16", IntegerLiteral::U16)} )]
    #[regex(r"0o[0-7_]+u32", |lex| {parse_integer::<OCTAL_FORMAT, u32>(lex, "0o", "u32", IntegerLiteral::U32)} )]
    #[regex(r"0o[0-7_]+u64", |lex| {parse_integer::<OCTAL_FORMAT, u64>(lex, "0o", "u64", IntegerLiteral::U64)} )]
    #[regex(r"0o[0-7_]+u128", |lex| {parse_integer::<OCTAL_FORMAT, u128>(lex, "0o", "u128", IntegerLiteral::U128)} )]
    Integer(IntegerLiteral),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_literal_parsing() {
        assert!(FloatLiteral::from_str("1.0").is_ok());
        assert!(FloatLiteral::from_str("3.14f32").is_ok());
        assert!(FloatLiteral::from_str("2.718f64").is_ok());
        assert!(FloatLiteral::from_str("1.0f16").is_ok());
        assert!(FloatLiteral::from_str("1.0bf16").is_ok());
        assert!(FloatLiteral::from_str("1.0E-13f128").is_ok());
        assert!(FloatLiteral::from_str("invalid").is_err());
    }

    #[test]
    fn test_integer_literal_parsing() {
        let mut lexer = Token::lexer("-42i32");
        let token = lexer.next().unwrap().unwrap();
        assert_eq!(token, Token::Integer(IntegerLiteral::I32(-42)));
    }
}
