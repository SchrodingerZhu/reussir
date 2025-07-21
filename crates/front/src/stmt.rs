use chumsky::input::ValueInput;
use chumsky::Parser;
use reussir_core::func::FunctionProto;
use reussir_core::Location;
use crate::{SpanBox, types::TypeDecl, ParserExtra, Token};
use crate::expr::ExprBox;
use chumsky::prelude::*;

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

fn function_proto<'a, I>() -> impl Parser<'a, I, FunctionProto, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    // let is_public = just(Token::Pub).or_not().map(Option::is_some);
    // let is_opaque = just(Token::Opaque).or_not().map(Option::is_some);
    // let has_region = just(Token::Region).or_not().map(Option::is_some);
    // let path = super::path();
    function_proto()
}