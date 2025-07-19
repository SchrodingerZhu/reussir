use std::alloc::Layout;

use chumsky::{
    Parser,
    error::Rich,
    input::ValueInput,
    prelude::{just, via_parser},
    select,
};
use reussir_core::{
    Location,
    types::{OpaqueType, Record},
};

use crate::{IntegerLiteral, ParserExtra, ParserState, Token};

pub enum TypeDecl {
    OpaqueDecl(OpaqueType),
    RecordDecl(Record),
}

pub fn opaque_type<'a, I>() -> impl Parser<'a, I, OpaqueType, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => x
    };
    let alignment = ident
        .filter(|x| *x == "alignment")
        .labelled("alignment keyword");
    let size = ident.filter(|x| *x == "size").labelled("size keyword");
    let value = select! {
        Token::Integer(x) => x
    };

    let valid_value = value
        .try_map(|x, location| {
            if let IntegerLiteral::U64(val) = x {
                Ok(val)
            } else {
                Err(Rich::custom(
                    location,
                    "size and alignment must be in u64 format",
                ))
            }
        })
        .recover_with(via_parser(value.map(|_| 1u64)))
        .labelled("size or alignment value");

    just(Token::Opaque)
        .ignore_then(ident)
        .then_ignore(just(Token::LBrace))
        .then_ignore(alignment)
        .then_ignore(just(Token::Colon))
        .then(valid_value)
        .then_ignore(just(Token::Comma))
        .then_ignore(size)
        .then_ignore(just(Token::Colon))
        .then(valid_value)
        .then_ignore(just(Token::RBrace))
        .try_map_with(|parsed, extra| {
            let ((ident, alignment), size) = parsed;
            #[allow(clippy::explicit_auto_deref)]
            let state: &ParserState = *extra.state();
            let path = state.module_path.clone().append(ident.into());
            let location = extra.span();
            let layout = match Layout::from_size_align(size as usize, alignment as usize) {
                Ok(layout) => layout,
                Err(e) => return Err(Rich::custom(location, e)),
            };

            Ok(OpaqueType {
                path,
                location: Some(location),
                layout,
            })
        })
        .labelled("opaque type declaration")
}

#[cfg(test)]
mod tests {
    use super::*;
    use reussir_core::type_path;
    use ustr::Ustr;

    #[test]
    fn test_opaque_parser() {
        let input = r#"
        opaque MyType { alignment: 8u64, size: 64u64 }
        "#
        .as_bytes();
        let state = ParserState::load(input, type_path!("test"), "<stdin>").unwrap();
        let parser = opaque_type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), &state.source);
        let result = parser.parse_with_state(token_stream, &mut &state).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn test_invalid_opaque_parser() {
        let input = r#"
        opaque MyType { alignment: 8, size: -8.0f64 }
        "#
        .as_bytes();
        let state = ParserState::load(input, type_path!("test"), "<stdin>").unwrap();
        let parser = opaque_type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), &state.source);
        let result = parser.parse_with_state(token_stream, &mut &state);
        state.print_result(&result);
        assert!(result.has_errors());
    }
}
