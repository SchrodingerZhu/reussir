use std::alloc::Layout;

use chumsky::{
    IterParser, Parser,
    error::Rich,
    input::ValueInput,
    prelude::{choice, just, recursive, via_parser},
    select,
};
use reussir_core::{
    Location, type_path,
    types::{Capability, Compound, OpaqueType, Record, RecordKind, Type, TypeExpr, Variant},
};
use ustr::Ustr;

use crate::{IntegerLiteral, ParserExtra, ParserState, SmallCollector, Token, path};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
                .or_not(),
        )
        .map(|(path, args)| TypeExpr { path, args })
        .labelled("type application")
}

fn r#type<'a, I>() -> impl Parser<'a, I, Type, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    recursive(|ty| {
        let ty_expr = type_expr(ty);
        let capability = choice((
            just(Token::Bang).to(Capability::Value),
            just(Token::Asterisk).to(Capability::Rigid),
            just(Token::Question).to(Capability::Flex),
            just(Token::At).to(Capability::Field),
        ))
        .or_not()
        .map(|x| x.unwrap_or(Capability::Default));
        capability
            .then(ty_expr)
            .map(|(capability, expr)| Type { capability, expr })
            .labelled("type")
    })
}

fn type_arglist<'a, I>() -> impl Parser<'a, I, Option<Box<[Ustr]>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => x
    };
    let comma = just(Token::Comma);
    ident
        .map(Ustr::from)
        .separated_by(comma)
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LAngle), just(Token::RAngle))
        .map(|args| args.0.into_boxed_slice())
        .or_not()
        .labelled("type argument list")
}

fn compound_struct<'a, I>() -> impl Parser<'a, I, Record, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => Ustr::from(x)
    };
    let types = ident
        .then_ignore(just(Token::Colon))
        .then(r#type())
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(|x| x.0.into_boxed_slice());
    just(Token::Struct)
        .ignore_then(ident)
        .then(type_arglist())
        .then(types)
        .map_with(|((name, type_args), fields), m| {
            let path = m.state().module_path.clone().append(name);
            let location = Some(m.span());
            let compound = Compound::Struct(fields);
            let kind = RecordKind::Compound(Box::new(compound));
            Record {
                path,
                location,
                kind,
                type_args,
            }
        })
}

fn compound_variant<'a, I>() -> impl Parser<'a, I, Variant, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => Ustr::from(x)
    };
    let types = ident
        .then_ignore(just(Token::Colon))
        .then(r#type())
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(|x| x.0.into_boxed_slice());
    ident
        .then(types)
        .map_with(|(name, fields), m| {
            let body = Box::new(Compound::Struct(fields));
            let location = Some(m.span());
            Variant {
                name,
                body,
                location,
            }
        })
}

fn tuple_variant<'a, I>() -> impl Parser<'a, I, Variant, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => Ustr::from(x)
    };
    let types = r#type()
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(|x| x.0.into_boxed_slice())
        .or_not();
    ident
        .then(types)
        .map_with(|(name, fields), m| {
            let body = Box::new(Compound::Tuple(fields));
            let location = Some(m.span());
            Variant {
                name,
                body,
                location,
            }
        })
}

fn enum_record<'a, I>() -> impl Parser<'a, I, Record, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => Ustr::from(x)
    };
    let variants = choice((compound_variant(), tuple_variant()))
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(|x| x.0.into_boxed_slice());

    just(Token::Enum)
        .ignore_then(ident)
        .then(type_arglist())
        .then(variants)
        .map_with(|((name, type_args), variants), m| {
            let path = m.state().module_path.clone().append(name);
            let location = Some(m.span());
            let kind = RecordKind::Enum(variants);
            Record {
                path,
                location,
                kind,
                type_args,
            }
        })
}

fn tuple_like_struct<'a, I>() -> impl Parser<'a, I, Record, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let ident = select! {
        Token::Ident(x) => x
    };
    let types = r#type()
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(|x| x.0.into_boxed_slice())
        .or_not();
    just(Token::Struct)
        .ignore_then(ident)
        .then(type_arglist())
        .then(types)
        .map_with(|((name, type_args), fields), m| {
            let path = m.state().module_path.clone().append(name.into());
            let location = Some(m.span());
            let compound = Compound::Tuple(fields);
            let kind = RecordKind::Compound(Box::new(compound));
            Record {
                path,
                location,
                kind,
                type_args,
            }
        })
}

fn opaque_type<'a, I>() -> impl Parser<'a, I, OpaqueType, ParserExtra<'a>> + Clone
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

pub fn type_decl<'a, I>() -> impl Parser<'a, I, TypeDecl, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    choice((
        opaque_type().map(TypeDecl::OpaqueDecl),
        enum_record().map(TypeDecl::RecordDecl),
        compound_struct().map(TypeDecl::RecordDecl),
        tuple_like_struct().map(TypeDecl::RecordDecl),
    ))
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

    #[test]
    fn test_tuple_like_struct_parser() {
        let input = r#"
        struct MyStruct<T>(T, i32, f64);
        struct MyStruct2<T2>(T2, std::vec::Vec<T2>, f64);
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = tuple_like_struct()
            .separated_by(just(Token::Semicolon))
            .allow_trailing()
            .collect::<Vec<_>>();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn test_type_decls() {
        let input = r#"
        enum MyEnum<T> {
            Variant1(!T),
            Variant2(i32, f64),
            Variant3,
        }
        struct Singleton
        struct MyStruct<T>(T, i32, f64)
        "#;
        let mut state = ParserState::new(type_path!("test"), "<stdin>").unwrap();
        let parser = type_decl()
            .repeated()
            .collect::<Vec<_>>();
        let token_stream = Token::stream(Ustr::from("<stdin>"), input);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:?}", result);
    }
}
