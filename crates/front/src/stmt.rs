use crate::expr::ExprBox;
use crate::{ParserExtra, SpanBox, Token, types::TypeDecl};
use crate::{ParserState, make_spanbox_with};
use chumsky::Parser;
use chumsky::input::ValueInput;
use chumsky::prelude::*;
use reussir_core::func::{FunctionProto, Param};
use reussir_core::types::{Capability, Type, TypeExpr};
use reussir_core::{Location, path};
use ustr::Ustr;

pub type StmtBox = SpanBox<Stmt>;

#[derive(Debug, Clone)]
pub struct Function {
    pub proto: FunctionProto,
    pub body: Option<ExprBox>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    TypeDecl(TypeDecl),
    FunctionDecl(Function),
}

fn param<'a, I>() -> impl Parser<'a, I, Param, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let name = select! {
        Token::Ident(ident) => Some(Ustr::from(ident)),
        Token::Underscore => None,
    };
    name.then_ignore(just(Token::Colon))
        .then(crate::types::r#type())
        .map_with(|(name, ty), extra| Param {
            name,
            location: extra.span(),
            ty,
        })
}

fn function_proto<'a, I>() -> impl Parser<'a, I, FunctionProto, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let is_public = just(Token::Pub).or_not().map(|x| x.is_some());
    let is_opaque = just(Token::Opaque).or_not().map(|x| x.is_some());
    let has_region = just(Token::Region).or_not().map(|x| x.is_some());
    let ident = select! {
        Token::Ident(ident) => Ustr::from(ident),
    };
    let type_args = ident
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LAngle), just(Token::RAngle))
        .map(|x| x.into_boxed_slice())
        .or_not();
    let params = param()
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(|x| x.into_boxed_slice())
        .or(just(Token::Unit).to(vec![].into_boxed_slice()));
    let return_type = just(Token::Arrow)
        .ignore_then(crate::types::r#type())
        .or_not()
        .map(|x| {
            x.unwrap_or_else(|| Type::Atom {
                capability: Capability::Default,
                expr: TypeExpr {
                    path: path!("unit"),
                    args: None,
                },
            })
        });
    is_public
        .then(is_opaque)
        .then(has_region)
        .then_ignore(just(Token::Fn))
        .then(ident)
        .then(type_args)
        .then(params)
        .then(return_type)
        .map_with(
            |((((((is_public, is_opaque), has_region), path), type_args), params), return_type),
             extra| {
                let state: &ParserState = extra.state();
                FunctionProto {
                    path: state.module_path.clone().append(path),
                    location: extra.span(),
                    type_args,
                    params,
                    return_type,
                    has_region,
                    is_public,
                    is_opaque,
                }
            },
        )
}

fn function<'a, I>() -> impl Parser<'a, I, Function, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let body = crate::expr::braced_expr_sequence(crate::expr::expr()).or_not();
    // further checks will be done in semantic analysis
    function_proto()
        .then(body)
        .try_map_with(|(mut proto, body), extra| {
            if body.is_some() && !proto.is_opaque {
                proto.location = extra.span();
            }
            Ok(Function { proto, body })
        })
}

pub fn stmt<'a, I>() -> impl Parser<'a, I, StmtBox, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    let type_decl = crate::types::type_decl().map(Stmt::TypeDecl);
    let function_decl = function().map(Stmt::FunctionDecl);
    type_decl.or(function_decl).map_with(make_spanbox_with)
}
