use std::alloc::Layout;

use chumsky::{
    IterParser, Parser,
    error::Rich,
    input::ValueInput,
    number::format::XML,
    prelude::{choice, just, recursive, skip_then_retry_until, skip_until, via_parser},
    select,
};
use reussir_core::{
    Location, type_path,
    types::{Capacity, OpaqueType, Record, Type, TypeExpr},
};

use crate::{IntegerLiteral, ParserExtra, ParserState, SmallCollector, Token, path};

pub enum TypeDecl {
    OpaqueDecl(OpaqueType),
    RecordDecl(Record),
}

fn type_expr<'a, I, P>(types: P) -> impl Parser<'a, I, TypeExpr, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
    P: Parser<'a, I, Type, ParserExtra<'a>> + Clone,
{
    path()
        .or(select! { Token::Primitive(x) => type_path!(x.into()) })
        .then(
            types
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<SmallCollector<_, 4>>()
                .delimited_by(just(Token::LAngle), just(Token::RAngle))
                .map(|x| x.0.into_boxed_slice())
                .or_not()
                .map(|x| x.unwrap_or_else(|| Box::new([]))),
        )
        .map_with(|(path, args), m| {
            if path.prefix().is_empty() && args.is_empty() {
                if let Some(usize) = m.state().lookup_type_var(path.basename()) {
                    return TypeExpr::Var(usize);
                }
            }
            TypeExpr::App(path, args)
        })
        .labelled("type application")
}

fn r#type<'a, I>() -> impl Parser<'a, I, Type, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    recursive(|ty| {
        let ty_expr = type_expr(ty);
        let capacity = choice((
            just(Token::Bang).to(Capacity::Value),
            just(Token::Asterisk).to(Capacity::Rigid),
            just(Token::Question).to(Capacity::Flex),
            just(Token::At).to(Capacity::Field),
        ))
        .or_not()
        .map(|x| x.unwrap_or(Capacity::Default));
        capacity
            .then(ty_expr)
            .map(|(capacity, expr)| Type { capacity, expr })
            .labelled("type")
    })
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
            let state: &ParserState = extra.state();
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
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = opaque_type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn test_invalid_opaque_parser() {
        let input = r#"
        opaque MyType { alignment: 8, size: -8.0f64 }
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = opaque_type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state);
        state.print_result(&result, input);
        assert!(result.has_errors());
    }

    #[test]
    fn test_type_parser() {
        let input = r#"
        MyType<T, i32>
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = r#type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:?}", result);
    }
    #[test]
    fn test_type_parser_error() {
        let input = r#"
        MyType<T, 12, f128, 21>
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = r#type();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state);
        state.print_result(&result, input);
        assert!(result.has_errors());
    }
}
