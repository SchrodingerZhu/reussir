use crate::types::{TypeBox, type_box};
use crate::{
    FloatLiteral, IntegerLiteral, SmallCollector, SpanBox, Token, WithSpan, make_spanbox_with, path,
};
use chumsky::prelude::*;
use reussir_core::Path;

pub type ExprBox = SpanBox<Expr>;
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mod,
    Mul,
    Div,
    LAnd,
    BAnd,
    LOr,
    BOr,
    Xor,
    Shr,
    Shl,
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UnaryOp {
    // Currently there is only negation
    Neg,
    Not,
}
pub enum Expr {
    Unit,
    Binary(ExprBox, WithSpan<BinaryOp>, ExprBox),
    Unary(WithSpan<UnaryOp>, ExprBox),
    Boolean(bool),
    Integer(IntegerLiteral),
    Float(FloatLiteral),
    Variable(String),
    IfThenElse(ExprBox, ExprBox, ExprBox),
    Sequence(Box<[ExprBox]>),
    Let(WithSpan<String>, Option<TypeBox>, ExprBox),
    Call(CallTarget, Option<Box<[ExprBox]>>),
}

macro_rules! expr_parser {

    (standalone $vis:vis $name:ident $body:block) => {
        $vis fn $name<'a, I>() -> impl Parser<'a, I, ExprBox, $crate::ParserExtra<'a>> + Clone
        where
            I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = reussir_core::Location>,
        {
            $body
        }
    };

    (recursive $vis:vis $name:ident $body:expr) => {
        $vis fn $name<'a, I, P>(toplevel: P) -> impl Parser<'a, I, ExprBox, $crate::ParserExtra<'a>> + Clone
        where
            I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = reussir_core::Location>,
            P: Parser<'a, I, ExprBox, $crate::ParserExtra<'a>> + Clone,
        {
            $body(toplevel)
        }
    };

    () => {};

    ( $vis:vis $name:ident => $body:expr ; $($trailing:tt)* ) => {
        expr_parser!{recursive $vis $name $body}
        expr_parser!{$($trailing)*}
    };

    ( $vis:vis $name:ident -> $body:block ; $($trailing:tt)* ) => {
        expr_parser!{standalone $vis $name $body}
        expr_parser!{$($trailing)*}
    };
}

pub enum CallTargetSegment {
    TurboFish(Box<[TypeBox]>),
    Path(Path),
}

pub type CallTarget = WithSpan<Box<[CallTargetSegment]>>;

fn spanned_ident<'a, I>() -> impl Parser<'a, I, WithSpan<String>, crate::ParserExtra<'a>> + Clone
where
    I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = crate::Location>,
{
    select! {
        Token::Ident(x) => x.to_string()
    }
    .map_with(|s, m| WithSpan(s, m.span()))
}

fn turbo_fish_body<'a, I>() -> impl Parser<'a, I, Box<[TypeBox]>, crate::ParserExtra<'a>> + Clone
where
    I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = crate::Location>,
{
    type_box()
        .delimited_by(just(Token::LAngle), just(Token::RAngle))
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map(|types| types.0.into_boxed_slice())
}

fn call_target<'a, I>() -> impl Parser<'a, I, CallTarget, crate::ParserExtra<'a>> + Clone
where
    I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = crate::Location>,
{
    // path ~ (:: (turbo_fish | path))*
    path()
        .map(CallTargetSegment::Path)
        .then(
            just(Token::PathSep)
                .ignore_then(
                    turbo_fish_body()
                        .map(CallTargetSegment::TurboFish)
                        .or(path().map(|p| CallTargetSegment::Path(p)))
                        .repeated()
                        .collect::<SmallCollector<_, 4>>(),
                )
                .or_not(),
        )
        .map_with(|(path, segments), m| {
            let res = [path]
                .into_iter()
                .chain(segments.into_iter().map(|x| x.0).flatten())
                .collect::<Box<[_]>>();
            WithSpan(res, m.span())
        })
}

expr_parser! {
    primitive -> {
        select! {
            Token::Integer(x) => Expr::Integer(x),
            Token::Float(x) => Expr::Float(x),
            Token::Boolean(x) => Expr::Boolean(x),
            Token::Unit => Expr::Unit,
        }
        .map_with(make_spanbox_with)
    };

    let_expr => | expr | {
        let type_annotation = just(Token::Colon)
            .ignore_then(type_box())
            .or_not();
        just(Token::Let)
            .ignore_then(spanned_ident())
            .then(type_annotation)
            .then_ignore(just(Token::Eq))
            .then(expr)
            .map(|((i, a), e)| Expr::Let(i, a, e) )
            .map_with(make_spanbox_with)
    };

    braced_expr_sequence => | expr : P | {
        expr
            .separated_by(just(Token::Semicolon))
            .collect::<SmallCollector<_, 8>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(|exprs| Expr::Sequence(exprs.0.into_boxed_slice()))
            .map_with(make_spanbox_with)
    };

    variable -> {
        select! { Token::Ident(x) => x.to_string() }.map(Expr::Variable).map_with(make_spanbox_with)
    };
    paren_expr => |expr : P| {
        expr.delimited_by(just(Token::LParen), just(Token::RParen))
    };
    pratt_expr => |atom : P| {
        use chumsky::pratt::*;
        let uop = | a, b | just(a).to_span().map(move |s| WithSpan(b, s));
        let bop = | a, b | just(a).to_span().map(move |s| WithSpan(b, s));
        let unary = | a, b, p | prefix(p, uop(a, b), move |op, rhs, m| { make_spanbox_with(Expr::Unary(op, rhs), m) });
        let binary = | a, b, p | infix(right(p), bop(a, b), move |lhs, op, rhs, m| { make_spanbox_with(Expr::Binary(lhs, op, rhs), m) });
        atom.pratt((
            unary(Token::Bang, UnaryOp::Not, 50),
            unary(Token::Minus, UnaryOp::Neg, 50),
            binary(Token::Asterisk, BinaryOp::Mul, 40),
            binary(Token::Slash, BinaryOp::Div, 40),
            binary(Token::Percent, BinaryOp::Mod, 30),
            binary(Token::Plus, BinaryOp::Add, 30),
            binary(Token::Minus, BinaryOp::Sub, 30),
            binary(Token::ShiftRight, BinaryOp::Shr, 25),
            binary(Token::ShiftLeft, BinaryOp::Shl, 25),
            binary(Token::Ampersand, BinaryOp::BAnd, 24),
            binary(Token::Caret, BinaryOp::Xor, 23),
            binary(Token::Or, BinaryOp::BOr, 22),
            binary(Token::EqEq, BinaryOp::Eq, 20),
            binary(Token::NotEq, BinaryOp::Ne, 20),
            binary(Token::LAngle, BinaryOp::Lt, 20),
            binary(Token::LessEq, BinaryOp::Le, 20),
            binary(Token::RAngle, BinaryOp::Gt, 20),
            binary(Token::GreaterEq, BinaryOp::Ge, 20),
            binary(Token::AndAnd, BinaryOp::LAnd, 10),
            binary(Token::OrOr, BinaryOp::LOr, 5),
        ))
    };
}
