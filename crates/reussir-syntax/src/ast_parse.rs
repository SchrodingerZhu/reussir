use crate::ast::{
    Access, BinaryOp, Capability, Constant, CtorCall, Expr, FloatingPointType, FuncCall, Function,
    Identifier, IntegralType, LambdaExpr, Path, Pattern, PatternCtorArg, PatternKind, Program,
    Record, RecordFields, RecordKind, Stmt, Type, Visibility, WithSpan,
};
use crate::lexer::{lex_lossless, unquote_string};
use crate::syntax::{SyntaxError, SyntaxKind};

#[derive(Clone, Debug)]
struct Token<'input> {
    kind: SyntaxKind,
    text: &'input str,
    start: usize,
    end: usize,
}

pub fn parse_program_ast(input: &str) -> Result<Program, Vec<SyntaxError>> {
    let mut parser = AstParser::new(input);
    let program = parser.parse_program();
    parser.finish(program)
}

pub fn parse_stmt_ast(input: &str) -> Result<Stmt, Vec<SyntaxError>> {
    let mut parser = AstParser::new(input);
    let stmt = parser.parse_stmt().unwrap_or_else(|| {
        parser.error_here("expected statement");
        Stmt::Mod {
            visibility: Visibility::Private,
            name: Identifier("<error>".to_string()),
        }
    });
    parser.finish(stmt)
}

pub fn parse_expr_ast(input: &str) -> Result<Expr, Vec<SyntaxError>> {
    let mut parser = AstParser::new(input);
    let expr = parser.parse_expr();
    parser.finish(expr)
}

pub fn parse_type_ast(input: &str) -> Result<Type, Vec<SyntaxError>> {
    let mut parser = AstParser::new(input);
    let ty = parser.parse_type();
    parser.finish(ty)
}

struct AstParser<'input> {
    tokens: Vec<Token<'input>>,
    pos: usize,
    errors: Vec<SyntaxError>,
}

impl<'input> AstParser<'input> {
    fn new(input: &'input str) -> Self {
        let mut errors = Vec::new();
        let tokens = lex_lossless(input)
            .filter_map(|item| match item {
                Ok((start, tok, end)) if tok.is_trivia() => None,
                Ok((start, tok, end)) => Some(Token {
                    kind: SyntaxKind::from(&tok),
                    text: tok.text(),
                    start,
                    end,
                }),
                Err(err) => {
                    errors.push(SyntaxError {
                        start: err.start,
                        end: err.end,
                        message: "invalid token".to_string(),
                    });
                    None
                }
            })
            .collect();
        Self {
            tokens,
            pos: 0,
            errors,
        }
    }

    fn finish<T>(mut self, value: T) -> Result<T, Vec<SyntaxError>> {
        if !self.at_eof() {
            self.error_here("unexpected trailing input");
        }
        if self.errors.is_empty() {
            Ok(value)
        } else {
            Err(self.errors)
        }
    }

    fn parse_program(&mut self) -> Program {
        let mut stmts = Vec::new();
        while !self.at_eof() {
            if let Some(stmt) = self.parse_stmt() {
                stmts.push(stmt);
            } else {
                self.error_here("expected top-level statement");
                self.bump();
            }
        }
        stmts
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        let start = self.start_offset();
        let visibility = if self.eat(SyntaxKind::Pub) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let stmt = match self.peek()? {
            SyntaxKind::Regional | SyntaxKind::Fn => self.parse_function(visibility),
            SyntaxKind::StructKw => self.parse_struct(visibility),
            SyntaxKind::EnumKw => self.parse_enum(visibility),
            SyntaxKind::ModKw => self.parse_mod(visibility),
            SyntaxKind::Extern if visibility == Visibility::Private => {
                self.parse_extern_trampoline()
            }
            _ => return None,
        };
        let end = self.end_offset();
        Some(Stmt::Spanned(WithSpan::new(Box::new(stmt), start, end)))
    }

    fn parse_function(&mut self, visibility: Visibility) -> Stmt {
        let is_regional = self.eat(SyntaxKind::Regional);
        self.expect(SyntaxKind::Fn, "expected `fn`");
        let name = self.expect_ident("expected function name");
        let generics = self.parse_generic_params();
        self.expect(
            SyntaxKind::LParen,
            "expected `(` before function parameters",
        );
        let params = self.parse_comma_list(SyntaxKind::RParen, |p| p.parse_typed_param());
        self.expect(SyntaxKind::RParen, "expected `)` after function parameters");
        let return_type = if self.eat(SyntaxKind::Arrow) {
            let flex = self.parse_flag(SyntaxKind::Flex);
            Some((self.parse_type(), flex))
        } else {
            None
        };
        let body = if self.at(SyntaxKind::LBrace) {
            Some(self.parse_block_expr())
        } else {
            self.expect(SyntaxKind::Semi, "expected function body or `;`");
            None
        };
        Stmt::Function(Function {
            visibility,
            name,
            generics,
            params,
            return_type,
            is_regional,
            body,
        })
    }

    fn parse_struct(&mut self, visibility: Visibility) -> Stmt {
        self.expect(SyntaxKind::StructKw, "expected `struct`");
        let default_cap = self.parse_capability().unwrap_or(Capability::Shared);
        let name = self.expect_ident("expected struct name");
        let ty_params = self.parse_generic_params();
        let fields = if self.eat(SyntaxKind::LBrace) {
            let fields = self.parse_comma_list(SyntaxKind::RBrace, |p| p.parse_named_field());
            self.expect(SyntaxKind::RBrace, "expected `}` after struct fields");
            RecordFields::Named(fields)
        } else if self.eat(SyntaxKind::LParen) {
            let fields = self.parse_comma_list(SyntaxKind::RParen, |p| p.parse_unnamed_field());
            self.expect(SyntaxKind::RParen, "expected `)` after struct fields");
            RecordFields::Unnamed(fields)
        } else {
            self.error_here("expected struct fields");
            RecordFields::Named(Vec::new())
        };
        Stmt::Record(Record {
            name,
            ty_params,
            fields,
            kind: RecordKind::Struct,
            visibility,
            default_cap,
        })
    }

    fn parse_enum(&mut self, visibility: Visibility) -> Stmt {
        self.expect(SyntaxKind::EnumKw, "expected `enum`");
        let default_cap = self.parse_capability().unwrap_or(Capability::Shared);
        let name = self.expect_ident("expected enum name");
        let ty_params = self.parse_generic_params();
        self.expect(SyntaxKind::LBrace, "expected `{` before enum variants");
        let variants = self.parse_comma_list(SyntaxKind::RBrace, |p| p.parse_variant());
        self.expect(SyntaxKind::RBrace, "expected `}` after enum variants");
        Stmt::Record(Record {
            name,
            ty_params,
            fields: RecordFields::Variants(variants),
            kind: RecordKind::Enum,
            visibility,
            default_cap,
        })
    }

    fn parse_mod(&mut self, visibility: Visibility) -> Stmt {
        self.expect(SyntaxKind::ModKw, "expected `mod`");
        let name = self.expect_ident("expected module name");
        self.expect(SyntaxKind::Semi, "expected `;` after module declaration");
        Stmt::Mod { visibility, name }
    }

    fn parse_extern_trampoline(&mut self) -> Stmt {
        self.expect(SyntaxKind::Extern, "expected `extern`");
        let abi = self.expect_string("expected ABI string");
        self.expect(SyntaxKind::Trampoline, "expected `trampoline`");
        let name = Identifier(self.expect_string("expected trampoline symbol string"));
        self.expect(
            SyntaxKind::Eq,
            "expected `=` in extern trampoline declaration",
        );
        let func = self.parse_path();
        let func_ty_args = self.parse_type_args_plain();
        self.expect(
            SyntaxKind::Semi,
            "expected `;` after extern trampoline declaration",
        );
        Stmt::ExternTrampoline {
            name,
            abi,
            func,
            func_ty_args,
        }
    }

    fn parse_typed_param(&mut self) -> (Identifier, Type, bool) {
        let name = self.expect_ident("expected parameter name");
        self.expect(SyntaxKind::Colon, "expected `:` after parameter name");
        let flex = self.parse_flag(SyntaxKind::Flex);
        let ty = self.parse_type();
        (name, ty, flex)
    }

    fn parse_named_field(&mut self) -> WithSpan<(Identifier, Type, bool)> {
        let start = self.start_offset();
        let name = self.expect_ident("expected field name");
        self.expect(SyntaxKind::Colon, "expected `:` after field name");
        let field = self.parse_flag(SyntaxKind::FieldKw);
        let ty = self.parse_type();
        WithSpan::new((name, ty, field), start, self.end_offset())
    }

    fn parse_unnamed_field(&mut self) -> WithSpan<(Type, bool)> {
        let start = self.start_offset();
        let field = self.parse_flag(SyntaxKind::FieldKw);
        let ty = self.parse_type();
        WithSpan::new((ty, field), start, self.end_offset())
    }

    fn parse_variant(&mut self) -> WithSpan<(Identifier, Vec<Type>)> {
        let start = self.start_offset();
        let name = self.expect_ident("expected variant name");
        let tys = if self.eat(SyntaxKind::LParen) {
            let tys = self.parse_comma_list(SyntaxKind::RParen, |p| p.parse_type());
            self.expect(
                SyntaxKind::RParen,
                "expected `)` after variant payload types",
            );
            tys
        } else {
            Vec::new()
        };
        WithSpan::new((name, tys), start, self.end_offset())
    }

    fn parse_generic_params(&mut self) -> Vec<(Identifier, Vec<Path>)> {
        if !self.eat(SyntaxKind::Lt) {
            return Vec::new();
        }
        let params = self.parse_comma_list(SyntaxKind::Gt, |p| {
            let name = p.expect_ident("expected generic parameter name");
            let bounds = if p.eat(SyntaxKind::Colon) {
                let mut bounds = vec![p.parse_path()];
                while p.eat(SyntaxKind::Plus) {
                    bounds.push(p.parse_path());
                }
                bounds
            } else {
                Vec::new()
            };
            (name, bounds)
        });
        self.expect(SyntaxKind::Gt, "expected `>` after generic parameters");
        params
    }

    fn parse_type(&mut self) -> Type {
        let start = self.start_offset();
        let ty = self.parse_arrow_type();
        Type::Spanned(WithSpan::new(Box::new(ty), start, self.end_offset()))
    }

    fn parse_arrow_type(&mut self) -> Type {
        let lhs = self.parse_type_atom_or_arg_list();
        if self.eat(SyntaxKind::Arrow) {
            let ret = self.parse_arrow_type();
            match lhs {
                TypeArgParse::Single(ty) => Type::Arrow {
                    args: vec![ty],
                    ret: Box::new(ret),
                },
                TypeArgParse::List(args) => Type::Arrow {
                    args,
                    ret: Box::new(ret),
                },
            }
        } else {
            match lhs {
                TypeArgParse::Single(ty) => ty,
                TypeArgParse::List(mut tys) => {
                    if tys.len() == 1 {
                        tys.remove(0)
                    } else {
                        self.error_here("tuple types are only valid as arrow arguments");
                        Type::Unit
                    }
                }
            }
        }
    }

    fn parse_type_atom_or_arg_list(&mut self) -> TypeArgParse {
        if self.eat(SyntaxKind::LParen) {
            let tys = self.parse_comma_list(SyntaxKind::RParen, |p| p.parse_arrow_type());
            self.expect(SyntaxKind::RParen, "expected `)` after type list");
            TypeArgParse::List(tys)
        } else {
            TypeArgParse::Single(self.parse_type_atom())
        }
    }

    fn parse_type_atom(&mut self) -> Type {
        match self.peek_text() {
            Some("i8") => {
                self.bump();
                Type::Integral(IntegralType::Signed(8))
            }
            Some("i16") => {
                self.bump();
                Type::Integral(IntegralType::Signed(16))
            }
            Some("i32") => {
                self.bump();
                Type::Integral(IntegralType::Signed(32))
            }
            Some("i64") => {
                self.bump();
                Type::Integral(IntegralType::Signed(64))
            }
            Some("u8") => {
                self.bump();
                Type::Integral(IntegralType::Unsigned(8))
            }
            Some("u16") => {
                self.bump();
                Type::Integral(IntegralType::Unsigned(16))
            }
            Some("u32") => {
                self.bump();
                Type::Integral(IntegralType::Unsigned(32))
            }
            Some("u64") => {
                self.bump();
                Type::Integral(IntegralType::Unsigned(64))
            }
            Some("f16") => {
                self.bump();
                Type::Float(FloatingPointType::Ieee(16))
            }
            Some("f32") => {
                self.bump();
                Type::Float(FloatingPointType::Ieee(32))
            }
            Some("f64") => {
                self.bump();
                Type::Float(FloatingPointType::Ieee(64))
            }
            Some("bfloat16") => {
                self.bump();
                Type::Float(FloatingPointType::BFloat16)
            }
            Some("float8") => {
                self.bump();
                Type::Float(FloatingPointType::Float8)
            }
            Some("bool") => {
                self.bump();
                Type::Bool
            }
            Some("str") => {
                self.bump();
                Type::Str
            }
            Some("unit") => {
                self.bump();
                Type::Unit
            }
            _ if self.at_ident_like() => {
                let path = self.parse_path();
                let args = self.parse_type_args_plain();
                Type::Expr { path, args }
            }
            _ => {
                self.error_here("expected type");
                self.bump();
                Type::Unit
            }
        }
    }

    fn parse_type_args_plain(&mut self) -> Vec<Type> {
        if !self.eat(SyntaxKind::Lt) {
            return Vec::new();
        }
        let args = self.parse_comma_list(SyntaxKind::Gt, |p| p.parse_type());
        self.expect(SyntaxKind::Gt, "expected `>` after type arguments");
        args
    }

    fn parse_type_args_optional(&mut self) -> Vec<Option<Type>> {
        self.try_parse_type_args_optional().unwrap_or_default()
    }

    fn try_parse_type_args_optional(&mut self) -> Option<Vec<Option<Type>>> {
        if !self.at(SyntaxKind::Lt) {
            return Some(Vec::new());
        }
        let saved_pos = self.pos;
        let saved_errors = self.errors.len();
        self.bump();
        let args = self.parse_comma_list(SyntaxKind::Gt, |p| {
            if p.eat(SyntaxKind::Underscore) {
                None
            } else {
                Some(p.parse_type())
            }
        });
        if self.eat(SyntaxKind::Gt) {
            Some(args)
        } else {
            self.pos = saved_pos;
            self.errors.truncate(saved_errors);
            None
        }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_expr_with_opts(true)
    }

    fn parse_expr_with_opts(&mut self, allow_struct: bool) -> Expr {
        self.parse_expr_bp_with_opts(0, allow_struct)
    }

    fn parse_expr_bp_with_opts(&mut self, min_bp: u8, allow_struct: bool) -> Expr {
        let mut lhs = self.parse_prefix_or_term_with_opts(allow_struct);
        loop {
            if self.eat(SyntaxKind::As) {
                if 8 < min_bp {
                    break;
                }
                let ty = self.parse_type();
                lhs = Expr::Cast {
                    ty,
                    expr: Box::new(lhs),
                };
                continue;
            }
            if self.at(SyntaxKind::LParen) {
                let args = self.parse_call_args();
                lhs = Expr::Call {
                    callee: Box::new(lhs),
                    args,
                };
                continue;
            }
            if self.eat(SyntaxKind::Dot) {
                let accesses = vec![self.parse_access_tail("expected access field after `.`")];
                lhs = match lhs {
                    Expr::AccessChain {
                        base,
                        accesses: mut existing,
                    } => {
                        existing.extend(accesses);
                        Expr::AccessChain {
                            base,
                            accesses: existing,
                        }
                    }
                    other => Expr::AccessChain {
                        base: Box::new(other),
                        accesses,
                    },
                };
                continue;
            }
            let Some((op, left_bp, right_bp)) = self.infix_binding_power() else {
                break;
            };
            if left_bp < min_bp {
                break;
            }
            self.bump();
            if op == BinaryOrAssign::AssignArrow {
                let access = self.parse_access_tail("expected assignment field after `->`");
                self.expect(SyntaxKind::Assign, "expected `:=` in regional assignment");
                let value = self.parse_expr_bp_with_opts(right_bp, allow_struct);
                lhs = Expr::Assign {
                    base: Box::new(lhs),
                    access,
                    value: Box::new(value),
                };
            } else if let BinaryOrAssign::Binary(op) = op {
                let rhs = self.parse_expr_bp_with_opts(right_bp, allow_struct);
                lhs = Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            }
        }
        lhs
    }

    fn parse_prefix_or_term_with_opts(&mut self, allow_struct: bool) -> Expr {
        if self.eat(SyntaxKind::Minus) {
            return Expr::UnaryOp {
                op: crate::ast::UnaryOp::Negate,
                expr: Box::new(self.parse_prefix_or_term_with_opts(allow_struct)),
            };
        }
        if self.eat(SyntaxKind::Bang) {
            return Expr::UnaryOp {
                op: crate::ast::UnaryOp::Not,
                expr: Box::new(self.parse_prefix_or_term_with_opts(allow_struct)),
            };
        }
        let start = self.start_offset();
        match self.peek() {
            Some(SyntaxKind::If) => self.parse_if_expr(),
            Some(SyntaxKind::Let) => self.parse_let_expr(),
            Some(SyntaxKind::Match) => self.parse_match_expr(),
            Some(SyntaxKind::Pipe) => self.parse_lambda_expr(),
            Some(SyntaxKind::Regional) => {
                self.bump();
                Expr::Regional(Box::new(self.parse_block_expr()))
            }
            Some(SyntaxKind::LBrace) => self.parse_block_expr(),
            Some(SyntaxKind::LParen) => {
                self.bump();
                let expr = self.parse_expr();
                self.expect(SyntaxKind::RParen, "expected `)` after expression");
                expr
            }
            Some(
                SyntaxKind::String
                | SyntaxKind::Double
                | SyntaxKind::Scientific
                | SyntaxKind::Int
                | SyntaxKind::True
                | SyntaxKind::False,
            ) => Expr::Const(self.parse_constant()),
            Some(_) if self.at_ident_like() => {
                let path = self.parse_path();
                let ty_args = self.parse_type_args_optional();
                let expr = if allow_struct && self.at(SyntaxKind::LBrace) {
                    let args = self.parse_ctor_brace_args();
                    Expr::CtorCall(CtorCall {
                        name: path,
                        ty_args,
                        args,
                    })
                } else if self.at(SyntaxKind::LParen) {
                    let args = self.parse_call_args();
                    Expr::FuncCall(FuncCall {
                        name: path,
                        ty_args,
                        args,
                    })
                } else if ty_args.is_empty() {
                    Expr::Var(path)
                } else {
                    Expr::CtorCall(CtorCall {
                        name: path,
                        ty_args,
                        args: Vec::new(),
                    })
                };
                Expr::Spanned(WithSpan::new(Box::new(expr), start, self.end_offset()))
            }
            _ => {
                self.error_here("expected expression");
                self.bump();
                Expr::Seq(Vec::new())
            }
        }
    }

    fn parse_if_expr(&mut self) -> Expr {
        self.expect(SyntaxKind::If, "expected `if`");
        let cond = self.parse_expr_with_opts(false);
        let then_expr = self.parse_block_expr();
        self.expect(SyntaxKind::Else, "expected `else` in if expression");
        let else_expr = self.parse_block_expr();
        Expr::If {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    fn parse_let_expr(&mut self) -> Expr {
        self.expect(SyntaxKind::Let, "expected `let`");
        let start = self.start_offset();
        let name = self.expect_ident("expected binding name");
        let end = self.end_offset();
        let ty = if self.eat(SyntaxKind::Colon) {
            let flex = self.parse_flag(SyntaxKind::Flex);
            Some((self.parse_type(), flex))
        } else {
            None
        };
        self.expect(SyntaxKind::Eq, "expected `=` after let binding");
        let value = self.parse_expr();
        Expr::Let {
            name: WithSpan::new(name, start, end),
            ty,
            value: Box::new(value),
        }
    }

    fn parse_match_expr(&mut self) -> Expr {
        self.expect(SyntaxKind::Match, "expected `match`");
        let scrutinee = self.parse_expr_with_opts(false);
        self.expect(SyntaxKind::LBrace, "expected `{` before match cases");
        let cases = self.parse_comma_list(SyntaxKind::RBrace, |p| {
            let pat = p.parse_pattern();
            p.expect(SyntaxKind::FatArrow, "expected `=>` after match pattern");
            let expr = p.parse_expr();
            (pat, expr)
        });
        self.expect(SyntaxKind::RBrace, "expected `}` after match cases");
        Expr::Match {
            scrutinee: Box::new(scrutinee),
            cases,
        }
    }

    fn parse_lambda_expr(&mut self) -> Expr {
        let start = self.start_offset();
        self.expect(SyntaxKind::Pipe, "expected `|`");
        let args = self.parse_comma_list(SyntaxKind::Pipe, |p| {
            let name = p.expect_ident("expected lambda parameter name");
            let ty = if p.eat(SyntaxKind::Colon) {
                Some(p.parse_type())
            } else {
                None
            };
            (name, ty)
        });
        self.expect(SyntaxKind::Pipe, "expected `|` after lambda parameters");
        let ret_ty = if self.eat(SyntaxKind::Arrow) {
            Some(self.parse_type())
        } else {
            None
        };
        let body = self.parse_expr();
        Expr::Spanned(WithSpan::new(
            Box::new(Expr::Lambda(LambdaExpr {
                args,
                body: Box::new(body),
                ret_ty,
            })),
            start,
            self.end_offset(),
        ))
    }

    fn parse_pattern(&mut self) -> Pattern {
        let kind = match self.peek() {
            Some(SyntaxKind::Underscore) => {
                self.bump();
                PatternKind::Wildcard
            }
            Some(
                SyntaxKind::String
                | SyntaxKind::Double
                | SyntaxKind::Scientific
                | SyntaxKind::Int
                | SyntaxKind::True
                | SyntaxKind::False,
            ) => PatternKind::Const(self.parse_constant()),
            Some(_) if self.at_ident_like() => {
                let path = self.parse_path();
                if self.eat(SyntaxKind::LBrace) {
                    let (args, has_ellipsis) = self.parse_pattern_args(SyntaxKind::RBrace, true);
                    self.expect(SyntaxKind::RBrace, "expected `}` after pattern fields");
                    PatternKind::Ctor {
                        path,
                        args,
                        has_ellipsis,
                        is_named: true,
                    }
                } else if self.eat(SyntaxKind::LParen) {
                    let (args, has_ellipsis) = self.parse_pattern_args(SyntaxKind::RParen, false);
                    self.expect(SyntaxKind::RParen, "expected `)` after pattern arguments");
                    PatternKind::Ctor {
                        path,
                        args,
                        has_ellipsis,
                        is_named: false,
                    }
                } else if path.segments.is_empty()
                    && path
                        .basename
                        .0
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_lowercase())
                {
                    PatternKind::Bind(path.basename)
                } else {
                    PatternKind::Ctor {
                        path,
                        args: Vec::new(),
                        has_ellipsis: false,
                        is_named: false,
                    }
                }
            }
            _ => {
                self.error_here("expected pattern");
                self.bump();
                PatternKind::Wildcard
            }
        };
        let guard = if self.eat(SyntaxKind::If) {
            Some(Box::new(self.parse_expr()))
        } else {
            None
        };
        Pattern { kind, guard }
    }

    fn parse_pattern_args(&mut self, end: SyntaxKind, named: bool) -> (Vec<PatternCtorArg>, bool) {
        if self.eat(SyntaxKind::DotDot) {
            return (Vec::new(), true);
        }
        let mut args = Vec::new();
        let mut ellipsis = false;
        while !self.at_eof() && !self.at(end) {
            if named {
                let field = self.expect_ident("expected pattern field name");
                let kind = if self.eat(SyntaxKind::Colon) {
                    self.parse_pattern().kind
                } else {
                    PatternKind::Bind(field.clone())
                };
                args.push(PatternCtorArg {
                    field: Some(field),
                    kind,
                });
            } else {
                args.push(PatternCtorArg {
                    field: None,
                    kind: self.parse_pattern().kind,
                });
            }
            if !self.eat(SyntaxKind::Comma) {
                break;
            }
            if self.eat(SyntaxKind::DotDot) {
                ellipsis = true;
                break;
            }
        }
        (args, ellipsis)
    }

    fn parse_block_expr(&mut self) -> Expr {
        self.expect(SyntaxKind::LBrace, "expected `{` before block");
        let mut exprs = Vec::new();
        while !self.at_eof() && !self.at(SyntaxKind::RBrace) {
            exprs.push(self.parse_expr());
            self.eat(SyntaxKind::Semi);
        }
        self.expect(SyntaxKind::RBrace, "expected `}` after block");
        Expr::Seq(exprs)
    }

    fn parse_call_args(&mut self) -> Vec<Expr> {
        self.expect(SyntaxKind::LParen, "expected `(` before call arguments");
        let args = self.parse_comma_list(SyntaxKind::RParen, |p| p.parse_expr());
        self.expect(SyntaxKind::RParen, "expected `)` after call arguments");
        args
    }

    fn parse_ctor_brace_args(&mut self) -> Vec<(Option<Identifier>, Expr)> {
        self.expect(
            SyntaxKind::LBrace,
            "expected `{` before constructor arguments",
        );
        let args = self.parse_comma_list(SyntaxKind::RBrace, |p| {
            if p.at_ident_like() && p.peek_nth(1) == Some(SyntaxKind::Colon) {
                let name = p.expect_ident("expected constructor field name");
                p.expect(
                    SyntaxKind::Colon,
                    "expected `:` after constructor field name",
                );
                (Some(name), p.parse_expr())
            } else {
                (None, p.parse_expr())
            }
        });
        self.expect(
            SyntaxKind::RBrace,
            "expected `}` after constructor arguments",
        );
        args
    }

    fn parse_constant(&mut self) -> Constant {
        match self.peek() {
            Some(SyntaxKind::Int) => {
                let text = self.peek_text().unwrap_or("0");
                self.bump();
                Constant::Int(text.parse().unwrap_or(0))
            }
            Some(SyntaxKind::Double | SyntaxKind::Scientific) => {
                let text = self.peek_text().unwrap_or("0.0").to_string();
                self.bump();
                Constant::Double(text)
            }
            Some(SyntaxKind::String) => {
                let text = unquote_string(self.peek_text().unwrap_or("\"\""));
                self.bump();
                Constant::String(text)
            }
            Some(SyntaxKind::True) => {
                self.bump();
                Constant::Bool(true)
            }
            Some(SyntaxKind::False) => {
                self.bump();
                Constant::Bool(false)
            }
            _ => {
                self.error_here("expected constant");
                self.bump();
                Constant::Int(0)
            }
        }
    }

    fn parse_access_tail(&mut self, message: &'static str) -> Access {
        if self.at_ident_like() {
            Access::Named(self.expect_ident(message))
        } else if self.at(SyntaxKind::Int) {
            let text = self.peek_text().unwrap_or("0");
            self.bump();
            Access::Unnamed(text.parse().unwrap_or(0))
        } else {
            self.error_here(message);
            Access::Named(Identifier("<error>".to_string()))
        }
    }

    fn parse_path(&mut self) -> Path {
        let mut parts = vec![self.expect_ident("expected identifier")];
        while self.eat(SyntaxKind::DoubleColon) {
            parts.push(self.expect_ident("expected identifier after `::`"));
        }
        Path::from_parts(parts)
    }

    fn parse_capability(&mut self) -> Option<Capability> {
        if !self.eat(SyntaxKind::LBracket) {
            return None;
        }
        let cap = match self.peek() {
            Some(SyntaxKind::Shared) => {
                self.bump();
                Capability::Shared
            }
            Some(SyntaxKind::Value) => {
                self.bump();
                Capability::Value
            }
            Some(SyntaxKind::Flex) => {
                self.bump();
                Capability::Flex
            }
            Some(SyntaxKind::Rigid) => {
                self.bump();
                Capability::Rigid
            }
            Some(SyntaxKind::FieldKw) => {
                self.bump();
                Capability::Field
            }
            Some(SyntaxKind::Regional) => {
                self.bump();
                Capability::Regional
            }
            _ => {
                self.error_here("expected capability");
                Capability::Shared
            }
        };
        self.expect(SyntaxKind::RBracket, "expected `]` after capability");
        Some(cap)
    }

    fn parse_flag(&mut self, flag: SyntaxKind) -> bool {
        if !self.eat(SyntaxKind::LBracket) {
            return false;
        }
        let found = self.eat(flag);
        if !found {
            self.error_here("expected flag");
        }
        self.expect(SyntaxKind::RBracket, "expected `]` after flag");
        found
    }

    fn parse_comma_list<T, F>(&mut self, end: SyntaxKind, mut parse_item: F) -> Vec<T>
    where
        F: FnMut(&mut Self) -> T,
    {
        let mut items = Vec::new();
        while !self.at_eof() && !self.at(end) {
            items.push(parse_item(self));
            if !self.eat(SyntaxKind::Comma) {
                break;
            }
        }
        items
    }

    fn infix_binding_power(&self) -> Option<(BinaryOrAssign, u8, u8)> {
        match self.peek()? {
            SyntaxKind::Arrow => Some((BinaryOrAssign::AssignArrow, 1, 1)),
            SyntaxKind::PipePipe => Some((BinaryOrAssign::Binary(BinaryOp::Or), 2, 3)),
            SyntaxKind::AmpAmp => Some((BinaryOrAssign::Binary(BinaryOp::And), 4, 5)),
            SyntaxKind::EqEq => Some((BinaryOrAssign::Binary(BinaryOp::Equ), 6, 7)),
            SyntaxKind::BangEq => Some((BinaryOrAssign::Binary(BinaryOp::Neq), 6, 7)),
            SyntaxKind::Lt => Some((BinaryOrAssign::Binary(BinaryOp::Lt), 6, 7)),
            SyntaxKind::Gt => Some((BinaryOrAssign::Binary(BinaryOp::Gt), 6, 7)),
            SyntaxKind::Lte => Some((BinaryOrAssign::Binary(BinaryOp::Lte), 6, 7)),
            SyntaxKind::Gte => Some((BinaryOrAssign::Binary(BinaryOp::Gte), 6, 7)),
            SyntaxKind::Plus => Some((BinaryOrAssign::Binary(BinaryOp::Add), 8, 9)),
            SyntaxKind::Minus => Some((BinaryOrAssign::Binary(BinaryOp::Sub), 8, 9)),
            SyntaxKind::Star => Some((BinaryOrAssign::Binary(BinaryOp::Mul), 10, 11)),
            SyntaxKind::Slash => Some((BinaryOrAssign::Binary(BinaryOp::Div), 10, 11)),
            SyntaxKind::Percent => Some((BinaryOrAssign::Binary(BinaryOp::Mod), 10, 11)),
            _ => None,
        }
    }

    fn expect(&mut self, kind: SyntaxKind, message: &'static str) -> bool {
        if self.eat(kind) {
            true
        } else {
            self.error_here(message);
            false
        }
    }

    fn expect_ident(&mut self, message: &'static str) -> Identifier {
        let text = if self.at_ident_like() {
            let text = self.peek_text().unwrap_or("<error>").to_string();
            self.bump();
            text
        } else {
            self.error_here(message);
            "<error>".to_string()
        };
        Identifier(text)
    }

    fn expect_string(&mut self, message: &'static str) -> String {
        let text = if self.at(SyntaxKind::String) {
            unquote_string(self.peek_text().unwrap_or("\"\""))
        } else {
            self.error_here(message);
            String::new()
        };
        self.eat(SyntaxKind::String);
        text
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn at(&self, kind: SyntaxKind) -> bool {
        self.peek() == Some(kind)
    }
    fn at_ident_like(&self) -> bool {
        self.peek().is_some_and(is_ident_like)
    }
    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }
    fn peek(&self) -> Option<SyntaxKind> {
        self.tokens.get(self.pos).map(|t| t.kind)
    }
    fn peek_text(&self) -> Option<&'input str> {
        self.tokens.get(self.pos).map(|t| t.text)
    }
    fn peek_nth(&self, offset: usize) -> Option<SyntaxKind> {
        self.tokens.get(self.pos + offset).map(|t| t.kind)
    }
    fn bump(&mut self) {
        if !self.at_eof() {
            self.pos += 1;
        }
    }

    fn start_offset(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map(|t| t.start)
            .unwrap_or_else(|| self.tokens.last().map(|t| t.end).unwrap_or(0))
    }
    fn end_offset(&self) -> usize {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.end)
            .unwrap_or_else(|| self.start_offset())
    }

    fn error_here(&mut self, message: &'static str) {
        let start = self
            .tokens
            .get(self.pos)
            .map(|t| t.start)
            .unwrap_or_else(|| self.tokens.last().map(|t| t.end).unwrap_or(0));
        let end = self
            .tokens
            .get(self.pos)
            .map(|t| t.end)
            .unwrap_or(start.saturating_add(1));
        self.errors.push(SyntaxError {
            start,
            end,
            message: message.to_string(),
        });
    }
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

enum TypeArgParse {
    Single(Type),
    List(Vec<Type>),
}
#[derive(PartialEq)]
enum BinaryOrAssign {
    Binary(BinaryOp),
    AssignArrow,
}
