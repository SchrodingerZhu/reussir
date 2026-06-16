//! The typed surface AST: thin typed *views* over the lossless `cstree` CST.
//!
//! There is no separate owned syntax tree and no serialization here. Following
//! the `cstree` "AST layer" recommendation (the rust-analyzer model), each typed
//! node — [`Expr`], [`Type`], [`Stmt`], [`Pattern`] — is a small handle that owns
//! a reference-counted [`ResolvedNode`] and projects its children on demand
//! through accessor methods.
//!
//! Identifiers are not copied as strings. Token text is interned in the green
//! tree, so [`Expr::kind`] (etc.) hands back the interned [`TokenKey`] for each
//! name — a `Copy`, `Eq`, `Hash` handle. Equal keys mean equal text within a
//! parse, so names compare and hash without touching the source; turning a key
//! back into text (for built-in-name checks, diagnostics, or a symbol table)
//! needs the parse's [`reussir_syntax::kind::Resolver`]. Because keys carry no
//! lifetime, the whole surface AST is lifetime-free.
//!
//! Spans are byte offsets into the source ([`ResolvedNode::text_range`]);
//! conversion to character or line/column positions is a diagnostics-rendering
//! concern handled elsewhere.

use reussir_syntax::ast::unescape_string;
use reussir_syntax::kind::{ResolvedNode, ResolvedToken, SyntaxKind, TokenKey};
// Bare names resolve to `SyntaxKind` variants (used pervasively in matches).
// A handful of variant names collide with surface AST *types* defined in this
// module; for those the local type wins for the bare name (a glob import has
// lower priority), and the kind is written as `SyntaxKind::Foo` (or `PathKind`).
use SyntaxKind::*;
use SyntaxKind::Path as PathKind;

// ===== spans =====

/// A byte-offset span into the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// A value carrying its source span.
#[derive(Clone, Copy, Debug)]
pub struct Spanned<T> {
    pub value: T,
    pub start: u32,
    pub end: u32,
}

impl<T> Spanned<T> {
    pub fn span(&self) -> Span {
        Span {
            start: self.start,
            end: self.end,
        }
    }
}

/// A dotted name, e.g. `std::collections::List`, as interned token keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub basename: TokenKey,
    pub segments: Vec<TokenKey>,
}

// ===== plain enums =====

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

/// The surface capability written on a record declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    Value,
    Shared,
    Regional,
    Flex,
    Rigid,
    Field,
    Unspecified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordKind {
    StructKind,
    EnumKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Gt,
    Lte,
    Gte,
    Equ,
    Neq,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegralType {
    Signed(u16),
    Unsigned(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpType {
    Ieee(u16),
    BFloat16,
    Float8,
}

/// A literal constant. String literals are *decoded* (escapes resolved), so they
/// are owned rather than an interned token key.
#[derive(Clone, Debug)]
pub enum Const {
    ConstInt(i64),
    ConstDouble(f64),
    ConstString(String),
    ConstBool(bool),
}

/// A field access in a chain or an assignment target.
#[derive(Clone, Copy, Debug)]
pub enum Access {
    Named(TokenKey),
    Unnamed(i64),
}

// ===== tree-walking helpers =====

fn node_span(node: &ResolvedNode) -> Span {
    let r = node.text_range();
    Span {
        start: r.start().into(),
        end: r.end().into(),
    }
}

fn token_span(token: &ResolvedToken) -> Span {
    let r = token.text_range();
    Span {
        start: r.start().into(),
        end: r.end().into(),
    }
}

fn spanned<T>(node: &ResolvedNode, value: T) -> Spanned<T> {
    let s = node_span(node);
    Spanned {
        value,
        start: s.start,
        end: s.end,
    }
}

/// The interned key for a token's text. Every token in the tree is interned, so
/// this is infallible for the identifier/keyword tokens we read it from.
fn key(token: &ResolvedToken) -> TokenKey {
    token.text_key().expect("token text is interned")
}

fn tokens(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedToken> {
    node.children_with_tokens()
        .filter_map(|el| el.as_token().copied())
        .filter(|t| !t.kind().is_trivia())
}

fn nodes(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedNode> {
    node.children()
}

fn child_node(node: &ResolvedNode, kind: SyntaxKind) -> Option<&ResolvedNode> {
    node.children().find(|n| n.kind() == kind)
}

fn is_expr_kind(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        BlockExpr
            | ParenExpr
            | IfExpr
            | LetExpr
            | MatchExpr
            | RegionalExpr
            | LambdaExpr
            | BinExpr
            | PrefixExpr
            | CastExpr
            | CallExpr
            | AccessChain
            | AssignExpr
            | LiteralExpr
            | VarExpr
            | FuncCallExpr
            | CtorCallExpr
    )
}

fn is_type_kind(kind: SyntaxKind) -> bool {
    matches!(kind, PrimType | PathType | ArrowType | ParenTypeList)
}

fn expr_children(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedNode> {
    node.children().filter(|n| is_expr_kind(n.kind()))
}

/// The defining name: the first identifier-like direct token after `kw`.
fn name_after(node: &ResolvedNode, kw: SyntaxKind) -> &ResolvedToken {
    let mut seen = false;
    for t in tokens(node) {
        if t.kind() == kw {
            seen = true;
        } else if seen && t.kind().is_ident_like() {
            return t;
        }
    }
    panic!("missing name token in {:?}", node.kind())
}

fn path_of(node: &ResolvedNode) -> Path {
    let mut segments: Vec<TokenKey> = tokens(node)
        .filter(|t| t.kind().is_ident_like())
        .map(key)
        .collect();
    let basename = segments.pop().expect("non-empty path");
    Path { basename, segments }
}

/// `(name, bounds)` declarations from a `GenericParamList` child.
fn generics_of(node: &ResolvedNode) -> Vec<(TokenKey, Vec<Path>)> {
    let Some(list) = child_node(node, GenericParamList) else {
        return Vec::new();
    };
    nodes(list)
        .filter(|n| n.kind() == GenericParam)
        .map(|param| {
            let name = tokens(param)
                .find(|t| t.kind().is_ident_like())
                .expect("generic parameter name");
            let bounds = nodes(param)
                .filter(|n| n.kind() == PathKind)
                .map(path_of)
                .collect();
            (key(name), bounds)
        })
        .collect()
}

/// `<T, _, U>` arguments on a path-based expression; `_` is `None`.
fn type_args_of(node: &ResolvedNode) -> Vec<Option<Type>> {
    let Some(list) = child_node(node, TypeArgList) else {
        return Vec::new();
    };
    nodes(list)
        .filter(|n| is_type_kind(n.kind()) || n.kind() == InferType)
        .map(|n| {
            if n.kind() == InferType {
                None
            } else {
                Some(Type::new(n))
            }
        })
        .collect()
}

/// Flatten `AccessSeg` children into accesses. A fused float token like `0.1`
/// contributes two numeric accesses.
fn access_segs_of(node: &ResolvedNode) -> Vec<Access> {
    let mut out = Vec::new();
    for seg in nodes(node).filter(|n| n.kind() == AccessSeg) {
        for t in tokens(seg) {
            match t.kind() {
                Dot | Arrow => {}
                IntLit => out.push(Access::Unnamed(int_value(t.text()))),
                FloatLit => {
                    for part in t.text().split('.') {
                        out.push(Access::Unnamed(int_value(part)));
                    }
                }
                k if k.is_ident_like() => out.push(Access::Named(key(t))),
                k => unreachable!("unexpected access token {k:?}"),
            }
        }
    }
    out
}

fn constant(token: &ResolvedToken) -> Const {
    match token.kind() {
        IntLit => Const::ConstInt(int_value(token.text())),
        FloatLit => Const::ConstDouble(
            token
                .text()
                .parse()
                .expect("float literal validated by lexer"),
        ),
        StringLit => Const::ConstString(unescape_string(token.text())),
        TrueKw => Const::ConstBool(true),
        FalseKw => Const::ConstBool(false),
        k => unreachable!("unexpected constant token {k:?}"),
    }
}

fn int_value(text: &str) -> i64 {
    text.parse()
        .expect("integer literal validated by the lexer")
}

fn binary_op(kind: SyntaxKind) -> Option<BinOp> {
    Some(match kind {
        Plus => BinOp::Add,
        Minus => BinOp::Sub,
        Star => BinOp::Mul,
        Slash => BinOp::Div,
        Percent => BinOp::Mod,
        LAngle => BinOp::Lt,
        RAngle => BinOp::Gt,
        LtEq => BinOp::Lte,
        GtEq => BinOp::Gte,
        EqEq => BinOp::Equ,
        BangEq => BinOp::Neq,
        AmpAmp => BinOp::And,
        PipePipe => BinOp::Or,
        _ => return None,
    })
}

fn prim_type(text: &str) -> TypeKind {
    use IntegralType::*;
    let int = TypeKind::TypeIntegral;
    let fp = TypeKind::TypeFp;
    match text {
        "i8" => int(Signed(8)),
        "i16" => int(Signed(16)),
        "i32" => int(Signed(32)),
        "i64" => int(Signed(64)),
        "u8" => int(Unsigned(8)),
        "u16" => int(Unsigned(16)),
        "u32" => int(Unsigned(32)),
        "u64" => int(Unsigned(64)),
        "f16" => fp(FpType::Ieee(16)),
        "f32" => fp(FpType::Ieee(32)),
        "f64" => fp(FpType::Ieee(64)),
        "bfloat16" => fp(FpType::BFloat16),
        "float8" => fp(FpType::Float8),
        "bool" => TypeKind::TypeBool,
        "str" => TypeKind::TypeStr,
        "unit" => TypeKind::TypeUnit,
        other => unreachable!("unexpected primitive type {other}"),
    }
}

// ===== types =====

#[derive(Clone, Debug)]
pub enum TypeKind {
    TypeIntegral(IntegralType),
    TypeFp(FpType),
    TypeBool,
    TypeStr,
    TypeUnit,
    /// A named type applied to arguments.
    TypeExpr(Path, Vec<Type>),
    /// A closure type: argument types and a result.
    TypeArrow(Vec<Type>, Type),
}

/// A type expression (a view over a `PrimType` / `PathType` / `ArrowType` node).
#[derive(Clone, Debug)]
pub struct Type {
    node: ResolvedNode,
}

impl Type {
    /// Wrap a type node, peeling transparent single-element parentheses.
    fn new(node: &ResolvedNode) -> Type {
        let mut n = node.clone();
        while n.kind() == ParenTypeList {
            let inner = nodes(&n)
                .find(|c| is_type_kind(c.kind()))
                .expect("parenthesized type")
                .clone();
            n = inner;
        }
        Type { node: n }
    }

    pub fn span(&self) -> Span {
        node_span(&self.node)
    }

    pub fn kind(&self) -> TypeKind {
        let node = &self.node;
        match node.kind() {
            PrimType => prim_type(tokens(node).next().expect("prim type token").text()),
            PathType => {
                let path = child_node(node, PathKind).expect("type path");
                let args = child_node(node, TypeArgList)
                    .map(|l| nodes(l).filter(|n| is_type_kind(n.kind())).map(Type::new).collect())
                    .unwrap_or_default();
                TypeKind::TypeExpr(path_of(path), args)
            }
            ArrowType => {
                let mut children = nodes(node).filter(|n| is_type_kind(n.kind()));
                let lhs = children.next().expect("arrow lhs");
                let rhs = children.next().expect("arrow rhs");
                // The left-hand side may be a parenthesized argument *list*
                // (multiple types), which we must not collapse via `Type::new`.
                let args = if lhs.kind() == ParenTypeList {
                    nodes(lhs).filter(|n| is_type_kind(n.kind())).map(Type::new).collect()
                } else {
                    vec![Type::new(lhs)]
                };
                TypeKind::TypeArrow(args, Type::new(rhs))
            }
            k => unreachable!("unexpected type node {k:?}"),
        }
    }
}

// ===== patterns =====

#[derive(Clone, Debug)]
pub enum PatternKind {
    Wildcard,
    Const(Const),
    Bind(TokenKey),
    Ctor(CtorPat),
}

/// A constructor pattern.
#[derive(Clone, Debug)]
pub struct CtorPat {
    pub path: Path,
    pub args: Vec<PatArg>,
    pub has_ellipsis: bool,
    pub is_named: bool,
}

#[derive(Clone, Debug)]
pub struct PatArg {
    pub field: Option<TokenKey>,
    pub kind: PatternKind,
}

/// A pattern (a view over a `Pattern` node), carrying an optional guard.
#[derive(Clone, Debug)]
pub struct Pattern {
    node: ResolvedNode,
}

impl Pattern {
    fn new(node: &ResolvedNode) -> Pattern {
        Pattern { node: node.clone() }
    }

    pub fn kind(&self) -> PatternKind {
        let k = nodes(&self.node)
            .find(|n| matches!(n.kind(), WildcardPat | BindPat | SyntaxKind::CtorPat | ConstPat))
            .expect("pattern kind");
        pattern_kind_of(k)
    }

    pub fn guard(&self) -> Option<Expr> {
        child_node(&self.node, PatGuard)
            .map(|g| Expr::new(expr_children(g).next().expect("guard expression")))
    }
}

fn pattern_kind_of(node: &ResolvedNode) -> PatternKind {
    match node.kind() {
        WildcardPat => PatternKind::Wildcard,
        ConstPat => PatternKind::Const(constant(tokens(node).next().expect("constant token"))),
        BindPat => {
            let path = child_node(node, PathKind).expect("binding path");
            PatternKind::Bind(key(tokens(path).next().expect("binding name")))
        }
        SyntaxKind::CtorPat => {
            let path = child_node(node, PathKind).expect("constructor path");
            let (args, has_ellipsis, is_named) = match child_node(node, PatArgList) {
                None => (Vec::new(), false, false),
                Some(list) => {
                    let named = tokens(list).next().is_some_and(|t| t.kind() == LBrace);
                    let ellipsis = child_node(list, PatRest).is_some();
                    let args = nodes(list)
                        .filter(|n| n.kind() == SyntaxKind::PatArg)
                        .map(pat_arg_of)
                        .collect();
                    (args, ellipsis, named)
                }
            };
            PatternKind::Ctor(CtorPat {
                path: path_of(path),
                args,
                has_ellipsis,
                is_named,
            })
        }
        k => unreachable!("unexpected pattern node {k:?}"),
    }
}

fn pat_arg_of(node: &ResolvedNode) -> PatArg {
    let field = tokens(node).find(|t| t.kind().is_ident_like()).map(key);
    let kind = nodes(node).find(|n| matches!(n.kind(), WildcardPat | BindPat | SyntaxKind::CtorPat | ConstPat));
    let kind = match kind {
        Some(k) => pattern_kind_of(k),
        // `{ x }` shorthand binds the field to a variable of the same name.
        None => PatternKind::Bind(field.expect("shorthand field name")),
    };
    PatArg { field, kind }
}

// ===== expressions =====

/// A lambda's payload.
#[derive(Clone, Debug)]
pub struct Lambda {
    pub args: Vec<(TokenKey, Option<Type>)>,
    pub body: Expr,
    pub ret_ty: Option<Type>,
}

/// A direct function call.
#[derive(Clone, Debug)]
pub struct FuncCall {
    pub name: Path,
    /// Explicit type arguments; `None` means "infer".
    pub ty_args: Vec<Option<Type>>,
    pub args: Vec<Expr>,
}

/// A constructor call.
#[derive(Clone, Debug)]
pub struct CtorCall {
    pub name: Path,
    pub ty_args: Vec<Option<Type>>,
    /// Each argument carries an optional field name (for named ctor syntax).
    pub args: Vec<(Option<TokenKey>, Expr)>,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    ConstExpr(Const),
    ExprSeq(Vec<Expr>),
    If(Expr, Expr, Expr),
    Let(Spanned<TokenKey>, Option<(Type, bool)>, Expr),
    Match(Expr, Vec<(Pattern, Expr)>),
    RegionalExpr(Expr),
    Lambda(Lambda),
    BinOpExpr(BinOp, Expr, Expr),
    UnaryOpExpr(UnaryOp, Expr),
    Cast(Type, Expr),
    CallExpr(Expr, Vec<Expr>),
    AccessChain(Expr, Vec<Access>),
    Assign(Expr, Access, Expr),
    Var(Path),
    FuncCallExpr(FuncCall),
    CtorCallExpr(CtorCall),
}

/// An expression (a view over the red tree). Parentheses are transparent.
#[derive(Clone, Debug)]
pub struct Expr {
    node: ResolvedNode,
}

impl Expr {
    /// Wrap an expression node, peeling transparent parentheses.
    fn new(node: &ResolvedNode) -> Expr {
        let mut n = node.clone();
        while n.kind() == ParenExpr {
            let inner = expr_children(&n).next().expect("inner expression").clone();
            n = inner;
        }
        Expr { node: n }
    }

    pub fn span(&self) -> Span {
        node_span(&self.node)
    }

    pub fn kind(&self) -> ExprKind {
        let node = &self.node;
        match node.kind() {
            LiteralExpr => ExprKind::ConstExpr(constant(tokens(node).next().expect("literal token"))),
            BlockExpr => ExprKind::ExprSeq(expr_children(node).map(Expr::new).collect()),
            IfExpr => {
                let mut parts = expr_children(node);
                let cond = Expr::new(parts.next().expect("condition"));
                let then = Expr::new(parts.next().expect("then branch"));
                let other = Expr::new(parts.next().expect("else branch"));
                ExprKind::If(cond, then, other)
            }
            LetExpr => {
                let name = let_name(node).expect("let binding name");
                let flex = child_node(node, FlexFlag).is_some();
                let ty = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .map(|t| (Type::new(t), flex));
                let value = expr_children(node).last().expect("bound value");
                let ns = token_span(name);
                ExprKind::Let(
                    Spanned {
                        value: key(name),
                        start: ns.start,
                        end: ns.end,
                    },
                    ty,
                    Expr::new(value),
                )
            }
            MatchExpr => {
                let scrutinee = Expr::new(expr_children(node).next().expect("scrutinee"));
                let arms = nodes(node)
                    .filter(|n| n.kind() == MatchArm)
                    .map(|arm| {
                        let pat = child_node(arm, SyntaxKind::Pattern).expect("arm pattern");
                        let body = expr_children(arm).next().expect("arm body");
                        (Pattern::new(pat), Expr::new(body))
                    })
                    .collect();
                ExprKind::Match(scrutinee, arms)
            }
            RegionalExpr => {
                ExprKind::RegionalExpr(Expr::new(expr_children(node).next().expect("regional body")))
            }
            LambdaExpr => {
                let args = child_node(node, LambdaParamList)
                    .map(|list| {
                        nodes(list)
                            .filter(|n| n.kind() == LambdaParam)
                            .map(|param| {
                                let name = tokens(param)
                                    .find(|t| t.kind().is_ident_like())
                                    .expect("lambda parameter name");
                                let ty = nodes(param)
                                    .find(|n| is_type_kind(n.kind()))
                                    .map(Type::new);
                                (key(name), ty)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let ret_ty = child_node(node, RetType)
                    .map(|r| Type::new(nodes(r).find(|n| is_type_kind(n.kind())).expect("ret")));
                let body = Expr::new(expr_children(node).last().expect("lambda body"));
                ExprKind::Lambda(Lambda { args, body, ret_ty })
            }
            BinExpr => {
                let op = tokens(node)
                    .find_map(|t| binary_op(t.kind()))
                    .expect("binary operator");
                let mut operands = expr_children(node);
                let lhs = Expr::new(operands.next().expect("lhs"));
                let rhs = Expr::new(operands.next().expect("rhs"));
                ExprKind::BinOpExpr(op, lhs, rhs)
            }
            PrefixExpr => {
                let op = tokens(node)
                    .find_map(|t| match t.kind() {
                        Minus => Some(UnaryOp::Negate),
                        Bang => Some(UnaryOp::Not),
                        _ => None,
                    })
                    .expect("unary operator");
                ExprKind::UnaryOpExpr(op, Expr::new(expr_children(node).next().expect("operand")))
            }
            CastExpr => {
                let operand = expr_children(node).next().expect("cast operand");
                let ty = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("cast type");
                ExprKind::Cast(Type::new(ty), Expr::new(operand))
            }
            CallExpr => {
                let callee = Expr::new(expr_children(node).next().expect("callee"));
                let args = child_node(node, ArgList)
                    .map(|l| expr_children(l).map(Expr::new).collect())
                    .unwrap_or_default();
                ExprKind::CallExpr(callee, args)
            }
            AccessChain => {
                let base = Expr::new(expr_children(node).next().expect("access base"));
                ExprKind::AccessChain(base, access_segs_of(node))
            }
            AssignExpr => {
                let mut operands = expr_children(node);
                let lhs = Expr::new(operands.next().expect("assignment target"));
                let rhs = Expr::new(operands.next().expect("assignment value"));
                let access = access_segs_of(node)
                    .into_iter()
                    .next()
                    .expect("assignment field");
                ExprKind::Assign(lhs, access, rhs)
            }
            VarExpr => ExprKind::Var(path_of(child_node(node, PathKind).expect("variable path"))),
            FuncCallExpr => {
                let path = child_node(node, PathKind).expect("function path");
                let args = child_node(node, ArgList)
                    .map(|l| expr_children(l).map(Expr::new).collect())
                    .unwrap_or_default();
                ExprKind::FuncCallExpr(FuncCall {
                    name: path_of(path),
                    ty_args: type_args_of(node),
                    args,
                })
            }
            CtorCallExpr => {
                let path = child_node(node, PathKind).expect("constructor path");
                let args = child_node(node, CtorArgList)
                    .map(|l| {
                        nodes(l)
                            .filter(|n| n.kind() == CtorArg)
                            .map(|arg| {
                                let named = tokens(arg).any(|t| t.kind() == Colon);
                                let field = if named {
                                    tokens(arg).find(|t| t.kind().is_ident_like()).map(key)
                                } else {
                                    None
                                };
                                let value = Expr::new(expr_children(arg).next().expect("arg value"));
                                (field, value)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                ExprKind::CtorCallExpr(CtorCall {
                    name: path_of(path),
                    ty_args: type_args_of(node),
                    args,
                })
            }
            k => unreachable!("unexpected expression node {k:?}"),
        }
    }
}

/// The binding name token of a `let`: the first identifier-like direct token
/// after the `let` keyword.
fn let_name(node: &ResolvedNode) -> Option<&ResolvedToken> {
    let mut seen = false;
    for t in tokens(node) {
        if t.kind() == LetKw {
            seen = true;
        } else if seen && t.kind().is_ident_like() {
            return Some(t);
        }
    }
    None
}

// ===== statements =====

/// A function definition. `params`/`return` carry the `[flex]` flag as a bool.
#[derive(Clone, Debug)]
pub struct Function {
    pub visibility: Visibility,
    pub name: TokenKey,
    /// Each generic is `(name, bounds)`.
    pub generics: Vec<(TokenKey, Vec<Path>)>,
    pub params: Vec<(TokenKey, Type, bool)>,
    pub return_type: Option<(Type, bool)>,
    pub is_regional: bool,
    pub body: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum RecordFields {
    Named(Vec<Spanned<(TokenKey, Type, bool)>>),
    Unnamed(Vec<Spanned<(Type, bool)>>),
    Variants(Vec<Spanned<(TokenKey, Vec<Type>)>>),
}

#[derive(Clone, Debug)]
pub struct Record {
    pub name: TokenKey,
    pub ty_params: Vec<(TokenKey, Vec<Path>)>,
    pub fields: RecordFields,
    pub kind: RecordKind,
    pub visibility: Visibility,
    pub default_cap: Capability,
}

/// An `extern "C" trampoline` declaration. The ABI and symbol strings are
/// decoded (escapes resolved), so they are owned.
#[derive(Clone, Debug)]
pub struct ExternTrampoline {
    pub name: String,
    pub abi: String,
    pub func: Path,
    pub ty_args: Vec<Type>,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Function(Function),
    Record(Record),
    /// `mod` declaration: `(visibility, name)`.
    Mod(Visibility, TokenKey),
    ExternTrampoline(ExternTrampoline),
}

/// A statement (a view over a top-level item node).
#[derive(Clone, Debug)]
pub struct Stmt {
    node: ResolvedNode,
}

impl Stmt {
    fn new(node: &ResolvedNode) -> Stmt {
        Stmt { node: node.clone() }
    }

    pub fn span(&self) -> Span {
        node_span(&self.node)
    }

    pub fn kind(&self) -> StmtKind {
        let node = &self.node;
        match node.kind() {
            FnStmt => StmtKind::Function(function_of(node)),
            StructStmt => StmtKind::Record(record_of(node, RecordKind::StructKind)),
            EnumStmt => StmtKind::Record(record_of(node, RecordKind::EnumKind)),
            ModStmt => StmtKind::Mod(visibility_of(node), key(name_after(node, ModKw))),
            ExternTrampolineStmt => StmtKind::ExternTrampoline(extern_of(node)),
            k => unreachable!("unexpected statement node {k:?}"),
        }
    }
}

fn visibility_of(node: &ResolvedNode) -> Visibility {
    if tokens(node).any(|t| t.kind() == PubKw) {
        Visibility::Public
    } else {
        Visibility::Private
    }
}

fn function_of(node: &ResolvedNode) -> Function {
    let body = expr_children(node).last().map(Expr::new);
    let return_type = child_node(node, RetType).map(|r| {
        let flex = child_node(r, FlexFlag).is_some();
        let ty = nodes(r)
            .find(|n| is_type_kind(n.kind()))
            .expect("return type");
        (Type::new(ty), flex)
    });
    let params = child_node(node, ParamList).map_or_else(Vec::new, |list| {
        nodes(list)
            .filter(|n| n.kind() == Param)
            .map(|param| {
                let name = tokens(param)
                    .find(|t| t.kind().is_ident_like())
                    .expect("parameter name");
                let flex = child_node(param, FlexFlag).is_some();
                let ty = nodes(param)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("parameter type");
                (key(name), Type::new(ty), flex)
            })
            .collect()
    });
    Function {
        visibility: visibility_of(node),
        name: key(name_after(node, FnKw)),
        generics: generics_of(node),
        params,
        return_type,
        is_regional: tokens(node).any(|t| t.kind() == RegionalKw),
        body,
    }
}

fn record_of(node: &ResolvedNode, kind: RecordKind) -> Record {
    let intro = if kind == RecordKind::StructKind {
        StructKw
    } else {
        EnumKw
    };
    let default_cap = child_node(node, SyntaxKind::Capability)
        .and_then(|c| tokens(c).find(|t| t.kind().is_ident_like() || t.kind() == RegionalKw))
        .map_or(Capability::Shared, |t| match t.text() {
            "shared" => Capability::Shared,
            "value" => Capability::Value,
            "flex" => Capability::Flex,
            "rigid" => Capability::Rigid,
            "field" => Capability::Field,
            "regional" => Capability::Regional,
            other => unreachable!("unexpected capability {other}"),
        });
    let fields = if let Some(named) = child_node(node, NamedFields) {
        RecordFields::Named(
            nodes(named)
                .filter(|n| n.kind() == NamedField)
                .map(|f| {
                    let name = tokens(f)
                        .find(|t| t.kind().is_ident_like())
                        .expect("field name");
                    let flag = child_node(f, FieldFlag).is_some();
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    spanned(f, (key(name), Type::new(ty), flag))
                })
                .collect(),
        )
    } else if let Some(unnamed) = child_node(node, UnnamedFields) {
        RecordFields::Unnamed(
            nodes(unnamed)
                .filter(|n| n.kind() == UnnamedField)
                .map(|f| {
                    let flag = child_node(f, FieldFlag).is_some();
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    spanned(f, (Type::new(ty), flag))
                })
                .collect(),
        )
    } else {
        let list = child_node(node, VariantList).expect("variant list");
        RecordFields::Variants(
            nodes(list)
                .filter(|n| n.kind() == Variant)
                .map(|v| {
                    let name = tokens(v)
                        .find(|t| t.kind().is_ident_like())
                        .expect("variant name");
                    let tys = nodes(v)
                        .filter(|n| is_type_kind(n.kind()))
                        .map(Type::new)
                        .collect();
                    spanned(v, (key(name), tys))
                })
                .collect(),
        )
    };
    Record {
        name: key(name_after(node, intro)),
        ty_params: generics_of(node),
        fields,
        kind,
        visibility: visibility_of(node),
        default_cap,
    }
}

fn extern_of(node: &ResolvedNode) -> ExternTrampoline {
    let mut strings = tokens(node).filter(|t| t.kind() == StringLit);
    let abi = unescape_string(strings.next().expect("ABI string").text());
    let sym = unescape_string(strings.next().expect("symbol string").text());
    let func = child_node(node, PathKind).expect("trampoline target path");
    let ty_args = child_node(node, TypeArgList)
        .map(|l| nodes(l).filter(|n| is_type_kind(n.kind())).map(Type::new).collect())
        .unwrap_or_default();
    ExternTrampoline {
        name: sym,
        abi,
        func: path_of(func),
        ty_args,
    }
}

// ===== program =====

/// A whole parsed source file: the top-level statement views.
pub type Program = Vec<Stmt>;

/// Build the surface program from a parsed `SourceFile` root.
pub fn program(root: &ResolvedNode) -> Program {
    assert_eq!(root.kind(), SourceFile, "expected a SourceFile root");
    root.children().map(Stmt::new).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> reussir_syntax::Parse {
        let parse = reussir_syntax::parse(source);
        assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
        parse
    }

    #[test]
    fn views_a_function_with_calls() {
        let parse = parse(
            "fn f(x: i32, y: i32) -> i32 {\n    let z = x * 2 + y % 3;\n    if (z >= 0 && !(z == 4)) { id<i32>(z) } else { -z }\n}",
        );
        let prog = program(&parse.root);
        assert_eq!(prog.len(), 1);
        let StmtKind::Function(f) = prog[0].kind() else {
            panic!("expected a function");
        };
        assert_eq!(parse.resolver().resolve(f.name), "f");
        assert_eq!(f.params.len(), 2);
        assert!(f.return_type.is_some());
        assert!(f.body.is_some());
    }

    #[test]
    fn views_enum_match_and_ctor() {
        let parse = parse(
            "enum List<T> { Nil, Cons(T, List<T>) }\nfn h(a: List<i32>) -> i32 {\n    match a {\n        List::Nil => 0,\n        List::Cons(x, xs) => x\n    }\n}",
        );
        let prog = program(&parse.root);
        assert_eq!(prog.len(), 2);
        let StmtKind::Record(r) = prog[0].kind() else {
            panic!("expected a record");
        };
        assert_eq!(parse.resolver().resolve(r.name), "List");
        assert_eq!(r.kind, RecordKind::EnumKind);
        assert!(matches!(r.fields, RecordFields::Variants(_)));
    }

    #[test]
    fn views_regional_and_struct() {
        let parse = parse(
            "struct [regional] L<T> { v: T, next: [field] L<T> }\nregional fn push<T>(c : [flex] L<T>, e : T) { c->next := Nullable::NonNull{c} }",
        );
        let prog = program(&parse.root);
        assert_eq!(prog.len(), 2);
        let StmtKind::Record(r) = prog[0].kind() else {
            panic!("expected a record");
        };
        assert_eq!(r.default_cap, Capability::Regional);
        let StmtKind::Function(f) = prog[1].kind() else {
            panic!("expected a function");
        };
        assert!(f.is_regional);
        // The `[flex]` flag on the first parameter is preserved.
        assert!(f.params[0].2);
    }
}
