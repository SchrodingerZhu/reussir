use crate::types::{TypeBox, type_box};
use crate::{FloatLiteral, IntegerLiteral, SpanBox, Token, WithSpan, make_spanbox_with};
use chumsky::prelude::*;
use either::Either;
use ustr::Ustr;

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

#[derive(Debug, Clone)]
pub enum CallTargetSegment {
    TurboFish(Box<[TypeBox]>),
    Ident(Ustr),
}

pub type CallTarget = WithSpan<Box<[CallTargetSegment]>>;

#[derive(Debug, Clone)]
pub enum Expr {
    Unit,
    Binary(ExprBox, WithSpan<BinaryOp>, ExprBox),
    Unary(WithSpan<UnaryOp>, ExprBox),
    Boolean(bool),
    Integer(IntegerLiteral),
    Float(FloatLiteral),
    Variable(String),
    IfThenElse(ExprBox, ExprBox, Option<ExprBox>),
    Sequence(Box<[ExprBox]>),
    Let(WithSpan<String>, Option<TypeBox>, ExprBox),
    Call(CallTarget, Option<Box<[ExprBox]>>),
    CtorCall(CallTarget, Box<[(WithSpan<Ustr>, ExprBox)]>),
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
        .separated_by(just(Token::Comma))
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LAngle), just(Token::RAngle))
        .map(|x| x.into_boxed_slice())
}

fn call_target<'a, I>() -> impl Parser<'a, I, CallTarget, crate::ParserExtra<'a>> + Clone
where
    I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = crate::Location>,
{
    // path ~ (:: (turbo_fish | path))*
    let ident = select! {
    Token::Ident(x) => CallTargetSegment::Ident(Ustr::from(x))};
    ident
        .then(
            just(Token::PathSep)
                .ignore_then(
                    turbo_fish_body()
                        .map(CallTargetSegment::TurboFish)
                        .or(ident),
                )
                .repeated()
                .collect::<Vec<_>>()
                .or_not(),
        )
        .map_with(|(path, segments), m| {
            let res = [path]
                .into_iter()
                .chain(segments.into_iter().flatten())
                .collect::<Box<[_]>>();
            WithSpan(res, m.span())
        })
}

type CallBody = Either<Option<Box<[ExprBox]>>, Box<[(WithSpan<Ustr>, ExprBox)]>>;

fn call_body<'a, I, P>(expr: P) -> impl Parser<'a, I, CallBody, crate::ParserExtra<'a>> + Clone
where
    I: chumsky::input::ValueInput<'a, Token = Token<'a>, Span = crate::Location>,
    P: Parser<'a, I, ExprBox, crate::ParserExtra<'a>> + Clone,
{
    let ident = select! {
            Token::Ident(x) = m => WithSpan(Ustr::from(x), m.span())
    };
    let ctor_body = ident
        .then_ignore(just(Token::Colon))
        .then(expr.clone())
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(|x| Either::Right(x.into_boxed_slice()));
    let function_or_tuple = expr
        .separated_by(just(Token::Comma))
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(|x| Some(x.into_boxed_slice()))
        .or(just(Token::Unit).map(|_| None))
        .map(Either::Left);
    ctor_body.or(function_or_tuple)
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

    if_then_else => | expr : P | {
        just(Token::If)
        .ignore_then(paren_expr(expr.clone()))
            .then(expr
                .clone()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .then(just(Token::Else)
                .ignore_then(expr.delimited_by(just(Token::LBrace), just(Token::RBrace)))
                .or_not())
            .map(|((cond, then), else_expr)| Expr::IfThenElse(cond, then, else_expr))
            .map_with(make_spanbox_with)
    };

    call_expr => | expr : P | {
        call_target()
            .then(call_body(expr))
            .map(|(target, args)| match args {
                Either::Left(args) => Expr::Call(target, args),
                Either::Right(ctor_args) => Expr::CtorCall(target, ctor_args),
            })
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
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(|exprs| Expr::Sequence(exprs.into_boxed_slice()))
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

    pub expr -> {
        recursive(|expr| {
            let atom = choice((
                call_expr(expr.clone()),
                primitive(),
                variable(),
                if_then_else(expr.clone()),
                let_expr(expr.clone()),
                braced_expr_sequence(expr.clone()),
                paren_expr(expr),
            ));
            pratt_expr(atom)
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParserState;
    use reussir_core::path;
    use ustr::Ustr;

    #[test]
    fn test_expr_parser() {
        let source = "foo + bar * 123.0 + foo::baz::qux::<f32>(1, 2.0, true)";
        let mut state = ParserState::new(path!("test"), "<stdin>").unwrap();
        let parser = expr();
        let token_stream = Token::stream(Ustr::from("<stdin>"), source);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:#?}", result);
    }

    #[test]
    fn test_expr_compound_parser() {
        let source = r#"
{
    let x = 42;
    let m = Point {
        x: 123.0f32,
        y: 456.0,
    };
    {
        ()
    }
}
        "#;
        let mut state = ParserState::new(path!("test"), "<stdin>").unwrap();
        let parser = expr();
        let token_stream = Token::stream(Ustr::from("<stdin>"), source);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:#?}", result);
    }
}
