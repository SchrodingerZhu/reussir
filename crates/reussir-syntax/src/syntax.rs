use crate::lexer::{LexicalError, Tok, lex_lossless};
use cstree::build::GreenNodeBuilder;
use cstree::green::GreenNode;
use cstree::{RawSyntaxKind, Syntax};
use serde::{Deserialize, Serialize};
use strum_macros::FromRepr;

#[derive(Clone, Copy, Debug, Eq, FromRepr, PartialEq, Serialize, Deserialize)]
#[repr(u32)]
pub enum SyntaxKind {
    Root,
    Error,
    Stmt,
    Function,
    ParamList,
    Param,
    ReturnType,
    Struct,
    Enum,
    FieldList,
    Field,
    Variant,
    Mod,
    ExternTrampoline,
    GenericParamList,
    GenericParam,
    TypeArgList,
    Type,
    Expr,
    BlockExpr,
    IfExpr,
    LetExpr,
    MatchExpr,
    MatchCase,
    LambdaExpr,
    Pattern,
    Path,
    ArgList,
    Access,
    Token,
    Whitespace,
    LineComment,
    BlockComment,
    Pub,
    Fn,
    Regional,
    StructKw,
    EnumKw,
    ModKw,
    Extern,
    Trampoline,
    If,
    Else,
    Let,
    Match,
    As,
    True,
    False,
    Shared,
    Value,
    Flex,
    Rigid,
    FieldKw,
    DoubleColon,
    Arrow,
    FatArrow,
    Assign,
    EqEq,
    BangEq,
    Lte,
    Gte,
    AmpAmp,
    PipePipe,
    DotDot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Lt,
    Gt,
    Colon,
    Semi,
    Comma,
    Dot,
    Eq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Pipe,
    Underscore,
    String,
    Double,
    Scientific,
    Int,
    Ident,
}

fn is_ident_like(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::Ident
            | SyntaxKind::Pub
            | SyntaxKind::Fn
            | SyntaxKind::Regional
            | SyntaxKind::StructKw
            | SyntaxKind::EnumKw
            | SyntaxKind::ModKw
            | SyntaxKind::Extern
            | SyntaxKind::Trampoline
            | SyntaxKind::If
            | SyntaxKind::Else
            | SyntaxKind::Let
            | SyntaxKind::Match
            | SyntaxKind::As
            | SyntaxKind::True
            | SyntaxKind::False
            | SyntaxKind::Shared
            | SyntaxKind::Value
            | SyntaxKind::Flex
            | SyntaxKind::Rigid
            | SyntaxKind::FieldKw
    )
}

impl Syntax for SyntaxKind {
    fn from_raw(raw: RawSyntaxKind) -> Self {
        Self::from_repr(raw.0).unwrap_or(Self::Error)
    }

    fn into_raw(self) -> RawSyntaxKind {
        RawSyntaxKind(self as u32)
    }

    fn static_text(self) -> Option<&'static str> {
        None
    }
}

impl<'input> From<&Tok<'input>> for SyntaxKind {
    fn from(tok: &Tok<'input>) -> Self {
        use SyntaxKind::*;
        match tok {
            Tok::Whitespace(_) => Whitespace,
            Tok::LineComment(_) => LineComment,
            Tok::BlockComment(_) => BlockComment,
            Tok::Pub => Pub,
            Tok::Fn => Fn,
            Tok::Regional => Regional,
            Tok::Struct => StructKw,
            Tok::Enum => EnumKw,
            Tok::Mod => ModKw,
            Tok::Extern => Extern,
            Tok::Trampoline => Trampoline,
            Tok::If => If,
            Tok::Else => Else,
            Tok::Let => Let,
            Tok::Match => Match,
            Tok::As => As,
            Tok::True => True,
            Tok::False => False,
            Tok::Shared => Shared,
            Tok::Value => Value,
            Tok::Flex => Flex,
            Tok::Rigid => Rigid,
            Tok::Field => FieldKw,
            Tok::DoubleColon => DoubleColon,
            Tok::Arrow => Arrow,
            Tok::FatArrow => FatArrow,
            Tok::Assign => Assign,
            Tok::EqEq => EqEq,
            Tok::BangEq => BangEq,
            Tok::Lte => Lte,
            Tok::Gte => Gte,
            Tok::AmpAmp => AmpAmp,
            Tok::PipePipe => PipePipe,
            Tok::DotDot => DotDot,
            Tok::LParen => LParen,
            Tok::RParen => RParen,
            Tok::LBrace => LBrace,
            Tok::RBrace => RBrace,
            Tok::LBracket => LBracket,
            Tok::RBracket => RBracket,
            Tok::Lt => Lt,
            Tok::Gt => Gt,
            Tok::Colon => Colon,
            Tok::Semi => Semi,
            Tok::Comma => Comma,
            Tok::Dot => Dot,
            Tok::Eq => Eq,
            Tok::Plus => Plus,
            Tok::Minus => Minus,
            Tok::Star => Star,
            Tok::Slash => Slash,
            Tok::Percent => Percent,
            Tok::Bang => Bang,
            Tok::Pipe => Pipe,
            Tok::Underscore => Underscore,
            Tok::String(_) => String,
            Tok::Double(_) => Double,
            Tok::Scientific(_) => Scientific,
            Tok::Int(_) => Int,
            Tok::Ident(_) => Ident,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SyntaxError {
    pub start: usize,
    pub end: usize,
    pub message: String,
}

#[derive(Debug)]
pub struct CstParse {
    pub green: GreenNode,
    pub errors: Vec<SyntaxError>,
}

#[derive(Clone, Debug)]
struct Token<'input> {
    kind: SyntaxKind,
    text: &'input str,
    start: usize,
    end: usize,
    trivia: bool,
}

impl From<LexicalError> for SyntaxError {
    fn from(err: LexicalError) -> Self {
        Self {
            start: err.start,
            end: err.end,
            message: "invalid token".to_string(),
        }
    }
}

pub fn parse_cst(input: &str) -> CstParse {
    let tokens = lex_lossless(input)
        .map(|item| match item {
            Ok((start, tok, end)) => Token {
                kind: SyntaxKind::from(&tok),
                text: tok.text(),
                start,
                end,
                trivia: tok.is_trivia(),
            },
            Err(err) => Token {
                kind: SyntaxKind::Error,
                text: input.get(err.start..err.end).unwrap_or(""),
                start: err.start,
                end: err.end,
                trivia: false,
            },
        })
        .collect();
    Parser::new(tokens).parse()
}

struct Parser<'input> {
    tokens: Vec<Token<'input>>,
    pos: usize,
    builder: GreenNodeBuilder<'static, 'static, SyntaxKind>,
    errors: Vec<SyntaxError>,
}

impl<'input> Parser<'input> {
    fn new(tokens: Vec<Token<'input>>) -> Self {
        Self {
            tokens,
            pos: 0,
            builder: GreenNodeBuilder::new(),
            errors: Vec::new(),
        }
    }

    fn parse(mut self) -> CstParse {
        self.builder.start_node(SyntaxKind::Root);
        while self.pos < self.tokens.len() {
            self.bump_trivia();
            if self.pos >= self.tokens.len() {
                break;
            }
            self.parse_stmt_or_recover();
        }
        self.builder.finish_node();
        let (green, _) = self.builder.finish();
        CstParse {
            green,
            errors: self.errors,
        }
    }

    fn parse_stmt_or_recover(&mut self) {
        let checkpoint = self.pos;
        self.builder.start_node(SyntaxKind::Stmt);
        if self.eat(SyntaxKind::Pub) {
            self.bump_trivia();
        }
        match self.peek() {
            Some(SyntaxKind::Regional) | Some(SyntaxKind::Fn) => self.parse_function_body(),
            Some(SyntaxKind::StructKw) => self.parse_struct_body(),
            Some(SyntaxKind::EnumKw) => self.parse_enum_body(),
            Some(SyntaxKind::ModKw) => self.parse_mod_body(),
            Some(SyntaxKind::Extern) => self.parse_extern_body(),
            _ => {
                self.error_here("expected a top-level statement");
                if self.pos == checkpoint {
                    self.bump_any();
                }
            }
        }
        self.builder.finish_node();
    }

    fn parse_function_body(&mut self) {
        self.builder.start_node(SyntaxKind::Function);
        self.eat(SyntaxKind::Regional);
        self.expect(SyntaxKind::Fn, "expected `fn`");
        self.expect_ident_like("expected function name");
        self.parse_generic_params();
        self.builder.start_node(SyntaxKind::ParamList);
        if self.expect(
            SyntaxKind::LParen,
            "expected `(` before function parameters",
        ) {
            self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_param());
            self.expect(SyntaxKind::RParen, "expected `)` after function parameters");
        }
        self.builder.finish_node();
        if self.eat(SyntaxKind::Arrow) {
            self.builder.start_node(SyntaxKind::ReturnType);
            self.parse_capability_or_flag();
            self.parse_type();
            self.builder.finish_node();
        }
        if self.at(SyntaxKind::LBrace) {
            self.parse_block_expr();
        } else {
            self.expect(SyntaxKind::Semi, "expected function body or `;`");
        }
        self.builder.finish_node();
    }

    fn parse_param(&mut self) {
        self.builder.start_node(SyntaxKind::Param);
        self.expect_ident_like("expected parameter name");
        self.expect(SyntaxKind::Colon, "expected `:` after parameter name");
        self.parse_capability_or_flag();
        self.parse_type();
        self.builder.finish_node();
    }

    fn parse_struct_body(&mut self) {
        self.builder.start_node(SyntaxKind::Struct);
        self.expect(SyntaxKind::StructKw, "expected `struct`");
        self.parse_capability_or_flag();
        self.expect_ident_like("expected struct name");
        self.parse_generic_params();
        self.builder.start_node(SyntaxKind::FieldList);
        if self.eat(SyntaxKind::LBrace) {
            self.parse_comma_list_until(SyntaxKind::RBrace, |p| p.parse_named_field());
            self.expect(SyntaxKind::RBrace, "expected `}` after struct fields");
        } else if self.eat(SyntaxKind::LParen) {
            self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_unnamed_field());
            self.expect(SyntaxKind::RParen, "expected `)` after struct fields");
        } else {
            self.error_here("expected struct field list");
        }
        self.builder.finish_node();
        self.builder.finish_node();
    }

    fn parse_enum_body(&mut self) {
        self.builder.start_node(SyntaxKind::Enum);
        self.expect(SyntaxKind::EnumKw, "expected `enum`");
        self.parse_capability_or_flag();
        self.expect_ident_like("expected enum name");
        self.parse_generic_params();
        if self.expect(SyntaxKind::LBrace, "expected `{` before enum variants") {
            self.parse_comma_list_until(SyntaxKind::RBrace, |p| p.parse_variant());
            self.expect(SyntaxKind::RBrace, "expected `}` after enum variants");
        }
        self.builder.finish_node();
    }

    fn parse_mod_body(&mut self) {
        self.builder.start_node(SyntaxKind::Mod);
        self.expect(SyntaxKind::ModKw, "expected `mod`");
        self.expect_ident_like("expected module name");
        self.expect(SyntaxKind::Semi, "expected `;` after module declaration");
        self.builder.finish_node();
    }

    fn parse_extern_body(&mut self) {
        self.builder.start_node(SyntaxKind::ExternTrampoline);
        self.expect(SyntaxKind::Extern, "expected `extern`");
        self.expect(SyntaxKind::String, "expected ABI string");
        self.expect(SyntaxKind::Trampoline, "expected `trampoline`");
        self.expect(SyntaxKind::String, "expected trampoline symbol string");
        self.expect(
            SyntaxKind::Eq,
            "expected `=` in extern trampoline declaration",
        );
        self.parse_path();
        self.parse_type_arg_list();
        self.expect(
            SyntaxKind::Semi,
            "expected `;` after extern trampoline declaration",
        );
        self.builder.finish_node();
    }

    fn parse_named_field(&mut self) {
        self.builder.start_node(SyntaxKind::Field);
        self.expect_ident_like("expected field name");
        self.expect(SyntaxKind::Colon, "expected `:` after field name");
        self.parse_capability_or_flag();
        self.parse_type();
        self.builder.finish_node();
    }

    fn parse_unnamed_field(&mut self) {
        self.builder.start_node(SyntaxKind::Field);
        self.parse_capability_or_flag();
        self.parse_type();
        self.builder.finish_node();
    }

    fn parse_variant(&mut self) {
        self.builder.start_node(SyntaxKind::Variant);
        self.expect_ident_like("expected variant name");
        if self.eat(SyntaxKind::LParen) {
            self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_type());
            self.expect(
                SyntaxKind::RParen,
                "expected `)` after variant payload types",
            );
        }
        self.builder.finish_node();
    }

    fn parse_generic_params(&mut self) {
        if !self.at(SyntaxKind::Lt) {
            return;
        }
        self.builder.start_node(SyntaxKind::GenericParamList);
        self.bump_any();
        self.parse_comma_list_until(SyntaxKind::Gt, |p| {
            p.builder.start_node(SyntaxKind::GenericParam);
            p.expect_ident_like("expected generic parameter name");
            if p.eat(SyntaxKind::Colon) {
                p.parse_path();
                while p.eat(SyntaxKind::Plus) {
                    p.parse_path();
                }
            }
            p.builder.finish_node();
        });
        self.expect(SyntaxKind::Gt, "expected `>` after generic parameters");
        self.builder.finish_node();
    }

    fn parse_type_arg_list(&mut self) {
        if !self.at(SyntaxKind::Lt) {
            return;
        }
        self.builder.start_node(SyntaxKind::TypeArgList);
        self.bump_any();
        self.parse_comma_list_until(SyntaxKind::Gt, |p| {
            if !p.eat(SyntaxKind::Underscore) {
                p.parse_type();
            }
        });
        self.expect(SyntaxKind::Gt, "expected `>` after type arguments");
        self.builder.finish_node();
    }

    fn parse_type(&mut self) {
        self.builder.start_node(SyntaxKind::Type);
        self.parse_type_atom();
        if self.eat(SyntaxKind::Arrow) {
            self.parse_type();
        }
        self.builder.finish_node();
    }
    fn parse_expr_type_arg_list(&mut self) {
        if self.next_angle_group_is_type_args() {
            self.parse_type_arg_list();
        }
    }

    fn next_angle_group_is_type_args(&self) -> bool {
        let Some(start) = self.peek_index() else {
            return false;
        };
        if self.tokens[start].kind != SyntaxKind::Lt {
            return false;
        }
        let mut depth = 0usize;
        for tok in self.tokens.iter().skip(start).filter(|tok| !tok.trivia) {
            match tok.kind {
                SyntaxKind::Lt => depth += 1,
                SyntaxKind::Gt => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return true;
                    }
                }
                SyntaxKind::LBrace
                | SyntaxKind::RBrace
                | SyntaxKind::Semi
                | SyntaxKind::FatArrow
                    if depth > 0 =>
                {
                    return false;
                }
                _ => {}
            }
        }
        false
    }

    fn parse_type_atom(&mut self) {
        match self.peek() {
            Some(kind) if self.is_builtin_type(kind) => {
                self.bump_any();
            }
            Some(kind) if is_ident_like(kind) => {
                self.parse_path();
                self.parse_type_arg_list();
            }
            Some(SyntaxKind::LParen) => {
                self.bump_any();
                self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_type());
                self.expect(SyntaxKind::RParen, "expected `)` after type list");
            }
            _ => {
                self.error_here("expected type");
                self.bump_any();
            }
        }
    }

    fn parse_expr(&mut self) {
        self.parse_expr_with_opts(true);
    }

    fn parse_expr_with_opts(&mut self, allow_struct: bool) {
        self.builder.start_node(SyntaxKind::Expr);
        self.parse_expr_bp_with_opts(0, allow_struct);
        self.builder.finish_node();
    }

    fn parse_expr_bp_with_opts(&mut self, min_bp: u8, allow_struct: bool) {
        self.parse_prefix_or_term_with_opts(allow_struct);
        loop {
            self.bump_trivia();
            if self.at(SyntaxKind::As) {
                if 8 < min_bp {
                    break;
                }
                self.bump_any();
                self.parse_type();
                continue;
            }
            if self.at(SyntaxKind::LParen) {
                self.builder.start_node(SyntaxKind::ArgList);
                self.bump_any();
                self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_expr());
                self.expect(SyntaxKind::RParen, "expected `)` after call arguments");
                self.builder.finish_node();
                continue;
            }
            if self.at(SyntaxKind::Dot) {
                self.builder.start_node(SyntaxKind::Access);
                self.bump_any();
                if !(self.eat_ident_like() || self.eat(SyntaxKind::Int)) {
                    self.error_here("expected access field after `.`");
                }
                self.builder.finish_node();
                continue;
            }
            let Some((left_bp, right_bp)) = self.infix_binding_power() else {
                break;
            };
            if left_bp < min_bp {
                break;
            }
            let op = self.peek();
            self.bump_any();
            if op == Some(SyntaxKind::Arrow) {
                self.builder.start_node(SyntaxKind::Access);
                if !(self.eat_ident_like() || self.eat(SyntaxKind::Int)) {
                    self.error_here("expected assignment field after `->`");
                }
                self.builder.finish_node();
                self.expect(SyntaxKind::Assign, "expected `:=` in regional assignment");
            }
            self.parse_expr_bp_with_opts(right_bp, allow_struct);
        }
    }

    fn parse_prefix_or_term_with_opts(&mut self, allow_struct: bool) {
        self.bump_trivia();
        match self.peek() {
            Some(SyntaxKind::Minus | SyntaxKind::Bang) => {
                self.bump_any();
                self.parse_prefix_or_term_with_opts(allow_struct);
            }
            Some(SyntaxKind::If) => self.parse_if_expr(),
            Some(SyntaxKind::Let) => self.parse_let_expr(),
            Some(SyntaxKind::Match) => self.parse_match_expr(),
            Some(SyntaxKind::Pipe) => self.parse_lambda_expr(),
            Some(SyntaxKind::Regional) => {
                self.builder.start_node(SyntaxKind::Expr);
                self.bump_any();
                self.parse_block_expr();
                self.builder.finish_node();
            }
            Some(SyntaxKind::LBrace) => self.parse_block_expr(),
            Some(SyntaxKind::LParen) => {
                self.bump_any();
                self.parse_expr();
                self.expect(SyntaxKind::RParen, "expected `)` after expression");
            }
            Some(
                SyntaxKind::String
                | SyntaxKind::Double
                | SyntaxKind::Scientific
                | SyntaxKind::Int
                | SyntaxKind::True
                | SyntaxKind::False,
            ) => {
                self.bump_any();
            }
            Some(kind) if is_ident_like(kind) => {
                self.parse_path();
                self.parse_expr_type_arg_list();
                if allow_struct && self.eat(SyntaxKind::LBrace) {
                    self.builder.start_node(SyntaxKind::ArgList);
                    self.parse_comma_list_until(SyntaxKind::RBrace, |p| p.parse_expr());
                    self.expect(
                        SyntaxKind::RBrace,
                        "expected `}` after constructor arguments",
                    );
                    self.builder.finish_node();
                }
            }
            _ => {
                self.error_here("expected expression");
                self.bump_any();
            }
        }
    }

    fn parse_if_expr(&mut self) {
        self.builder.start_node(SyntaxKind::IfExpr);
        self.expect(SyntaxKind::If, "expected `if`");
        self.parse_expr_with_opts(false);
        self.parse_block_expr();
        self.expect(SyntaxKind::Else, "expected `else` in if expression");
        self.parse_block_expr();
        self.builder.finish_node();
    }

    fn parse_let_expr(&mut self) {
        self.builder.start_node(SyntaxKind::LetExpr);
        self.expect(SyntaxKind::Let, "expected `let`");
        self.expect_ident_like("expected binding name");
        if self.eat(SyntaxKind::Colon) {
            self.parse_capability_or_flag();
            self.parse_type();
        }
        self.expect(SyntaxKind::Eq, "expected `=` after let binding");
        self.parse_expr();
        self.builder.finish_node();
    }

    fn parse_match_expr(&mut self) {
        self.builder.start_node(SyntaxKind::MatchExpr);
        self.expect(SyntaxKind::Match, "expected `match`");
        self.parse_expr_with_opts(false);
        if self.expect(SyntaxKind::LBrace, "expected `{` before match cases") {
            self.parse_comma_list_until(SyntaxKind::RBrace, |p| {
                p.builder.start_node(SyntaxKind::MatchCase);
                p.parse_pattern();
                p.expect(SyntaxKind::FatArrow, "expected `=>` after match pattern");
                p.parse_expr();
                p.builder.finish_node();
            });
            self.expect(SyntaxKind::RBrace, "expected `}` after match cases");
        }
        self.builder.finish_node();
    }

    fn parse_lambda_expr(&mut self) {
        self.builder.start_node(SyntaxKind::LambdaExpr);
        self.expect(SyntaxKind::Pipe, "expected `|`");
        self.parse_comma_list_until(SyntaxKind::Pipe, |p| {
            p.expect_ident_like("expected lambda parameter name");
            if p.eat(SyntaxKind::Colon) {
                p.parse_type();
            }
        });
        self.expect(SyntaxKind::Pipe, "expected `|` after lambda parameters");
        if self.eat(SyntaxKind::Arrow) {
            self.parse_type();
        }
        self.parse_expr();
        self.builder.finish_node();
    }

    fn parse_pattern(&mut self) {
        self.builder.start_node(SyntaxKind::Pattern);
        match self.peek() {
            Some(
                SyntaxKind::Underscore
                | SyntaxKind::String
                | SyntaxKind::Double
                | SyntaxKind::Scientific
                | SyntaxKind::Int
                | SyntaxKind::True
                | SyntaxKind::False,
            ) => {
                self.bump_any();
            }
            Some(kind) if is_ident_like(kind) => {
                self.parse_path();
                if self.eat(SyntaxKind::LBrace) {
                    self.parse_comma_list_until(SyntaxKind::RBrace, |p| p.parse_pattern_field());
                    self.expect(SyntaxKind::RBrace, "expected `}` after pattern fields");
                } else if self.eat(SyntaxKind::LParen) {
                    self.parse_comma_list_until(SyntaxKind::RParen, |p| p.parse_pattern());
                    self.expect(SyntaxKind::RParen, "expected `)` after pattern arguments");
                }
            }
            _ => {
                self.error_here("expected pattern");
                self.bump_any();
            }
        }
        if self.eat(SyntaxKind::If) {
            self.parse_expr();
        }
        self.builder.finish_node();
    }

    fn parse_pattern_field(&mut self) {
        self.expect_ident_like("expected pattern field name");
        if self.eat(SyntaxKind::Colon) {
            self.parse_pattern();
        }
    }

    fn parse_block_expr(&mut self) {
        self.builder.start_node(SyntaxKind::BlockExpr);
        if self.expect(SyntaxKind::LBrace, "expected `{` before block") {
            while !self.at_eof() && !self.at(SyntaxKind::RBrace) {
                self.parse_expr();
                self.eat(SyntaxKind::Semi);
            }
            self.expect(SyntaxKind::RBrace, "expected `}` after block");
        }
        self.builder.finish_node();
    }

    fn parse_path(&mut self) {
        self.builder.start_node(SyntaxKind::Path);
        self.expect_ident_like("expected identifier");
        while self.eat(SyntaxKind::DoubleColon) {
            self.expect_ident_like("expected identifier after `::`");
        }
        self.builder.finish_node();
    }

    fn parse_capability_or_flag(&mut self) {
        if !self.eat(SyntaxKind::LBracket) {
            return;
        }
        match self.peek() {
            Some(
                SyntaxKind::Shared
                | SyntaxKind::Value
                | SyntaxKind::Flex
                | SyntaxKind::Rigid
                | SyntaxKind::FieldKw
                | SyntaxKind::Regional,
            ) => {
                self.bump_any();
            }
            _ => self.error_here("expected capability or field flag"),
        }
        self.expect(
            SyntaxKind::RBracket,
            "expected `]` after capability or flag",
        );
    }

    fn parse_comma_list_until<F>(&mut self, end: SyntaxKind, mut parse_item: F)
    where
        F: FnMut(&mut Self),
    {
        self.bump_trivia();
        while !self.at_eof() && !self.at(end) {
            parse_item(self);
            self.bump_trivia();
            if !self.eat(SyntaxKind::Comma) {
                break;
            }
            self.bump_trivia();
        }
    }

    fn infix_binding_power(&self) -> Option<(u8, u8)> {
        match self.peek()? {
            SyntaxKind::Arrow => Some((1, 1)),
            SyntaxKind::PipePipe => Some((2, 3)),
            SyntaxKind::AmpAmp => Some((4, 5)),
            SyntaxKind::EqEq
            | SyntaxKind::BangEq
            | SyntaxKind::Lt
            | SyntaxKind::Gt
            | SyntaxKind::Lte
            | SyntaxKind::Gte => Some((6, 7)),
            SyntaxKind::Plus | SyntaxKind::Minus => Some((8, 9)),
            SyntaxKind::Star | SyntaxKind::Slash | SyntaxKind::Percent => Some((10, 11)),
            _ => None,
        }
    }

    fn is_builtin_type(&self, kind: SyntaxKind) -> bool {
        matches!(kind, SyntaxKind::Ident)
            && matches!(
                self.peek_text(),
                Some(
                    "i8" | "i16"
                        | "i32"
                        | "i64"
                        | "u8"
                        | "u16"
                        | "u32"
                        | "u64"
                        | "f16"
                        | "f32"
                        | "f64"
                        | "bfloat16"
                        | "float8"
                        | "bool"
                        | "str"
                        | "unit"
                )
            )
    }

    fn expect(&mut self, kind: SyntaxKind, message: &'static str) -> bool {
        if self.eat(kind) {
            true
        } else {
            self.error_here(message);
            false
        }
    }

    fn expect_ident_like(&mut self, message: &'static str) -> bool {
        if self.eat_ident_like() {
            true
        } else {
            self.error_here(message);
            false
        }
    }

    fn eat_ident_like(&mut self) -> bool {
        self.bump_trivia();
        if self.peek().is_some_and(is_ident_like) {
            self.bump_any();
            true
        } else {
            false
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        self.bump_trivia();
        if self.at(kind) {
            self.bump_any();
            true
        } else {
            false
        }
    }

    fn at(&self, kind: SyntaxKind) -> bool {
        self.peek() == Some(kind)
    }

    fn at_eof(&self) -> bool {
        self.peek_index().is_none()
    }

    fn peek(&self) -> Option<SyntaxKind> {
        self.peek_index().map(|idx| self.tokens[idx].kind)
    }

    fn peek_text(&self) -> Option<&'input str> {
        self.peek_index().map(|idx| self.tokens[idx].text)
    }

    fn peek_index(&self) -> Option<usize> {
        self.tokens
            .iter()
            .enumerate()
            .skip(self.pos)
            .find(|(_, tok)| !tok.trivia)
            .map(|(idx, _)| idx)
    }

    fn bump_trivia(&mut self) {
        while self.pos < self.tokens.len() && self.tokens[self.pos].trivia {
            self.emit_current();
        }
    }

    fn bump_any(&mut self) {
        if self.pos < self.tokens.len() {
            self.emit_current();
        }
    }

    fn emit_current(&mut self) {
        let tok = &self.tokens[self.pos];
        self.builder.token(tok.kind, tok.text);
        if tok.kind == SyntaxKind::Error {
            self.errors.push(SyntaxError {
                start: tok.start,
                end: tok.end,
                message: "invalid token".to_string(),
            });
        }
        self.pos += 1;
    }

    fn error_here(&mut self, message: &'static str) {
        let (start, end) = self
            .peek_index()
            .map(|idx| (self.tokens[idx].start, self.tokens[idx].end))
            .unwrap_or_else(|| {
                self.tokens
                    .last()
                    .map(|tok| (tok.end, tok.end.saturating_add(1)))
                    .unwrap_or((0, 1))
            });
        self.errors.push(SyntaxError {
            start,
            end,
            message: message.to_string(),
        });
    }
}
