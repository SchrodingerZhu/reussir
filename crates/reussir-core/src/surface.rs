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
//! Spans are byte offsets into the source (`ResolvedNode::text_range`);
//! conversion to character or line/column positions is a diagnostics-rendering
//! concern handled elsewhere.

use reussir_syntax::kind::{ResolvedNode, ResolvedToken, SyntaxKind, TokenKey};
use reussir_syntax::literal::{unescape_char, unescape_string};
use smallvec::{SmallVec, smallvec};

use crate::literal::{self, FloatLit, Integer};
// Bare names resolve to `SyntaxKind` variants (used pervasively in matches).
// A handful of variant names collide with surface AST *types* defined in this
// module; for those the local type wins for the bare name (a glob import has
// lower priority), and the kind is written as `SyntaxKind::Foo` (or `PathKind`).
use SyntaxKind::Path as PathKind;
use SyntaxKind::*;

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

/// A dotted name, e.g. `std::collections::List`, as interned token keys. The
/// qualifier `segments` are usually empty or one deep, so they stay inline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub basename: TokenKey,
    pub segments: SmallVec<[TokenKey; 2]>,
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
/// are owned rather than an interned token key. Numeric literals are *exact*
/// ([`crate::literal`]): carried arbitrary-precision and rounded to their
/// ground type only at the end of the pipeline.
#[derive(Clone, Debug)]
pub enum Const {
    ConstInt(Integer),
    ConstFloat(FloatLit),
    ConstString(String),
    ConstChar(u32),
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
    matches!(
        kind,
        PrimType | PathType | ArrowType | ParenTypeList | SyntaxKind::ArrayType
    )
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
    let mut segments: SmallVec<[TokenKey; 2]> = tokens(node)
        .filter(|t| t.kind().is_ident_like())
        .map(key)
        .collect();
    let basename = segments.pop().expect("non-empty path");
    Path { basename, segments }
}

/// Merge every direct `Path` child of a call node into one path: committed
/// mid-path type arguments (`Arc<List<i32>>::Cons{…}`) split the spelling
/// into two path nodes around the argument list.
fn call_path_of(node: &ResolvedNode) -> Path {
    let mut segments: SmallVec<[TokenKey; 2]> = SmallVec::new();
    for p in nodes(node).filter(|n| n.kind() == PathKind) {
        segments.extend(tokens(p).filter(|t| t.kind().is_ident_like()).map(key));
    }
    let basename = segments.pop().expect("non-empty path");
    Path { basename, segments }
}

/// `(name, bounds)` declarations from a `GenericParamList` child.
fn generics_of(node: &ResolvedNode) -> SmallVec<[(TokenKey, Vec<Path>); 2]> {
    let Some(list) = child_node(node, GenericParamList) else {
        return SmallVec::new();
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
fn type_args_of(node: &ResolvedNode) -> SmallVec<[Option<Type>; 2]> {
    let Some(list) = child_node(node, TypeArgList) else {
        return SmallVec::new();
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
fn access_segs_of(node: &ResolvedNode) -> SmallVec<[Access; 2]> {
    let mut out = SmallVec::new();
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
        IntLit => Const::ConstInt(literal::parse_int(token.text())),
        FloatLit => Const::ConstFloat(literal::parse_float(token.text())),
        StringLit => Const::ConstString(unescape_string(token.text())),
        CharLit => Const::ConstChar(unescape_char(token.text()) as u32),
        TrueKw => Const::ConstBool(true),
        FalseKw => Const::ConstBool(false),
        k => unreachable!("unexpected constant token {k:?}"),
    }
}

/// A tuple-access index. An index beyond `i64` can never name a field, so
/// saturate rather than panic and let field resolution report it.
fn int_value(text: &str) -> i64 {
    i64::try_from(&literal::parse_int(text)).unwrap_or(i64::MAX)
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
        "char" => TypeKind::TypeChar,
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
    TypeChar,
    TypeUnit,
    /// A named type applied to arguments.
    TypeExpr(Path, SmallVec<[Type; 2]>),
    /// A closure type: argument types and a result.
    TypeArrow(SmallVec<[Type; 2]>, Type),
    /// A statically shaped array: an element type and one extent expression
    /// per dimension (`[f64; 512]`, `[f64; 5, 16, 8]`). The extents are kept
    /// as expressions; the elaborator decides which forms it can evaluate.
    TypeArray(Type, SmallVec<[Expr; 2]>),
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
            SyntaxKind::ArrayType => {
                let elem = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("array element type");
                let extents = expr_children(node).map(Expr::new).collect();
                TypeKind::TypeArray(Type::new(elem), extents)
            }
            PathType => {
                let path = child_node(node, PathKind).expect("type path");
                let args = child_node(node, TypeArgList)
                    .map(|l| {
                        nodes(l)
                            .filter(|n| is_type_kind(n.kind()))
                            .map(Type::new)
                            .collect()
                    })
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
                    nodes(lhs)
                        .filter(|n| is_type_kind(n.kind()))
                        .map(Type::new)
                        .collect()
                } else {
                    smallvec![Type::new(lhs)]
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
    // `Vec`, not `SmallVec`: `PatArg` is recursive (`PatArg -> PatternKind ->
    // CtorPat`), so the heap indirection is needed to give it a finite size.
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

    pub fn span(&self) -> Span {
        node_span(&self.node)
    }

    pub fn kind(&self) -> PatternKind {
        let k = nodes(&self.node)
            .find(|n| {
                matches!(
                    n.kind(),
                    WildcardPat | BindPat | SyntaxKind::CtorPat | ConstPat
                )
            })
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
    let kind = nodes(node).find(|n| {
        matches!(
            n.kind(),
            WildcardPat | BindPat | SyntaxKind::CtorPat | ConstPat
        )
    });
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
    pub args: SmallVec<[(TokenKey, Option<Type>); 4]>,
    pub body: Expr,
    pub ret_ty: Option<Type>,
}

/// A direct function call.
#[derive(Clone, Debug)]
pub struct FuncCall {
    pub name: Path,
    /// Explicit type arguments; `None` means "infer".
    pub ty_args: SmallVec<[Option<Type>; 2]>,
    pub args: SmallVec<[Expr; 4]>,
}

/// A constructor call.
#[derive(Clone, Debug)]
pub struct CtorCall {
    pub name: Path,
    pub ty_args: SmallVec<[Option<Type>; 2]>,
    /// Each argument carries an optional field name (for named ctor syntax).
    pub args: SmallVec<[(Option<TokenKey>, Expr); 4]>,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    ConstExpr(Const),
    ExprSeq(Vec<Expr>),
    /// `if cond { then } else { other }`; `None` for the else branch means it
    /// was omitted, which is sugar for an empty (unit) `else {}`.
    If(Expr, Expr, Option<Expr>),
    Let(Spanned<TokenKey>, Option<(Type, bool)>, Expr),
    Match(Expr, Vec<(Pattern, Expr)>),
    RegionalExpr(Expr),
    Lambda(Box<Lambda>),
    BinOpExpr(BinOp, Expr, Expr),
    UnaryOpExpr(UnaryOp, Expr),
    Cast(Type, Expr),
    /// An *indirect* call: apply a computed callee value to arguments, e.g.
    /// `(|x| x)(1)` or `make_adder(1)(2)`. The callee is an arbitrary [`Expr`]
    /// that must evaluate to a closure; there is no name and no type arguments
    /// (a closure value has nothing to instantiate). Checked by
    /// `infer_closure_call`. Contrast with [`ExprKind::FuncCallExpr`].
    CallExpr(Expr, SmallVec<[Expr; 4]>),
    AccessChain(Expr, SmallVec<[Access; 2]>),
    Assign(Expr, Access, Expr),
    Var(Path),
    /// A *direct* call to a named function: a path applied to arguments, with
    /// optional explicit type arguments, e.g. `f(1)` or `id<i32>(x)`. The callee
    /// is a [`Path`] (usually) resolved to a top-level function prototype (static
    /// dispatch), so it carries `ty_args` for instantiating that function's
    /// generics. Checked by `infer_func_call`. Contrast with
    /// [`ExprKind::CallExpr`], whose callee is a first-class value.
    ///
    /// This form is *syntactic*: the parser emits it for any `name(args)`, so a
    /// local variable bound to a closure — `let var = |x| x; var(0)` — also lands
    /// here, since whether `var` is a function or a binding is a scope question
    /// the parser cannot answer. Resolving function-vs-local-closure is therefore
    /// deferred to `infer_func_call` at callee-resolution time, where a local
    /// binding should shadow a same-named function and be dispatched indirectly
    /// (as an [`ExprKind::CallExpr`] on a `Var` callee).
    /// TODO: not yet wired up — `infer_func_call` currently only consults the
    /// function table, so calling a local closure by name is rejected.
    FuncCallExpr(Box<FuncCall>),
    CtorCallExpr(Box<CtorCall>),
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

    /// Project the node's immediate children into a typed [`ExprKind`]. This
    /// re-walks the direct children on every call (no memoization).
    ///
    /// Ideally we could cache the projection keyed by the resolved node, so a
    /// repeated `kind()` on the same `Expr` is free. We don't do that yet: it is
    /// unclear whether the elaborator actually re-reads the same node's fields
    /// (today each is visited about once), and whether the re-walk — a handful
    /// of bounds-checked child scans, no subtree traversal — is ever a real
    /// bottleneck. Add a cache only once a profile says it matters.
    pub fn kind(&self) -> ExprKind {
        let node = &self.node;
        match node.kind() {
            LiteralExpr => {
                ExprKind::ConstExpr(constant(tokens(node).next().expect("literal token")))
            }
            BlockExpr => ExprKind::ExprSeq(expr_children(node).map(Expr::new).collect()),
            IfExpr => {
                let mut parts = expr_children(node);
                let cond = Expr::new(parts.next().expect("condition"));
                let then = Expr::new(parts.next().expect("then branch"));
                let other = parts.next().map(Expr::new);
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
            RegionalExpr => ExprKind::RegionalExpr(Expr::new(
                expr_children(node).next().expect("regional body"),
            )),
            LambdaExpr => {
                let args = child_node(node, LambdaParamList)
                    .map(|list| {
                        nodes(list)
                            .filter(|n| n.kind() == LambdaParam)
                            .map(|param| {
                                let name = tokens(param)
                                    .find(|t| t.kind().is_ident_like())
                                    .expect("lambda parameter name");
                                let ty =
                                    nodes(param).find(|n| is_type_kind(n.kind())).map(Type::new);
                                (key(name), ty)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let ret_ty = child_node(node, RetType)
                    .map(|r| Type::new(nodes(r).find(|n| is_type_kind(n.kind())).expect("ret")));
                let body = Expr::new(expr_children(node).last().expect("lambda body"));
                ExprKind::Lambda(Box::new(Lambda { args, body, ret_ty }))
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
                child_node(node, PathKind).expect("function path");
                let args = child_node(node, ArgList)
                    .map(|l| expr_children(l).map(Expr::new).collect())
                    .unwrap_or_default();
                ExprKind::FuncCallExpr(Box::new(FuncCall {
                    name: call_path_of(node),
                    ty_args: type_args_of(node),
                    args,
                }))
            }
            CtorCallExpr => {
                child_node(node, PathKind).expect("constructor path");
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
                                let value =
                                    Expr::new(expr_children(arg).next().expect("arg value"));
                                (field, value)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                ExprKind::CtorCallExpr(Box::new(CtorCall {
                    name: call_path_of(node),
                    ty_args: type_args_of(node),
                    args,
                }))
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

/// An outer item attribute such as `#[transform_anchor]` or
/// `#[ffi(import)]`. Bare arguments land in `args`; `key = "value"` pairs
/// land in `values` (strings decoded).
#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: TokenKey,
    pub args: Vec<TokenKey>,
    pub values: Vec<(TokenKey, String)>,
    pub span: Span,
}

/// An opaque foreign source block (`[{ ... }]`); `body` includes the
/// surrounding braces.
#[derive(Clone, Debug)]
pub struct SourceBlock {
    pub body: String,
    pub body_span: Span,
}

/// A function definition. `params`/`return` carry the `[flex]` flag as a bool.
#[derive(Clone, Debug)]
pub struct Function {
    pub visibility: Visibility,
    pub name: TokenKey,
    /// Each generic is `(name, bounds)`.
    pub generics: SmallVec<[(TokenKey, Vec<Path>); 2]>,
    pub params: SmallVec<[(TokenKey, Type, bool); 4]>,
    pub return_type: Option<(Type, bool)>,
    pub is_regional: bool,
    pub body: Option<Expr>,
    /// An opaque foreign body (`[{ ... }];`), for `#[ffi(import)]` functions.
    pub foreign_body: Option<SourceBlock>,
}

#[derive(Clone, Debug)]
pub enum RecordFields {
    /// Each named field is `(name, type, is_mutable, visibility)`.
    Named(Vec<Spanned<(TokenKey, Type, bool, Visibility)>>),
    /// Each unnamed field is `(type, is_mutable, visibility)`.
    Unnamed(Vec<Spanned<(Type, bool, Visibility)>>),
    Variants(Vec<Spanned<(TokenKey, SmallVec<[Type; 2]>)>>),
    /// A field-less declaration (`struct V<T>;`): an opaque (FFI) record.
    Opaque,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub name: TokenKey,
    pub ty_params: SmallVec<[(TokenKey, Vec<Path>); 2]>,
    pub fields: RecordFields,
    pub kind: RecordKind,
    pub visibility: Visibility,
    pub default_cap: Capability,
    /// `#[repr(fixed)]`: uniform max-arm box sizing for an enum instead of the
    /// default per-constructor sizing. Only meaningful for enums.
    pub repr_fixed: bool,
}

/// An `extern "C" trampoline` declaration. The ABI and symbol strings are
/// decoded (escapes resolved), so they are owned.
#[derive(Clone, Debug)]
pub struct ExternTrampoline {
    pub name: String,
    pub abi: String,
    pub func: Path,
    pub ty_args: SmallVec<[Type; 2]>,
}

/// An opaque MLIR transform script. `body` includes its surrounding braces.
#[derive(Clone, Debug)]
pub struct TransformScript {
    pub body: String,
    pub body_span: Span,
}

/// A foreign source block item, `extern "rust" [{ ... }];` — opaque source
/// shared by the file's FFI functions (imports, helper items, ...).
#[derive(Clone, Debug)]
pub struct ExternSource {
    pub abi: String,
    pub block: SourceBlock,
}

/// A file-scoped path abbreviation, `import p (as n)?;`: `name` is the
/// identifier the binding introduces (the path's last segment when no rename
/// is given) and `path` its target. Imports are always private; `visibility`
/// is kept so the elaborator can reject `pub import` with a real diagnostic.
#[derive(Clone, Debug)]
pub struct ImportDecl {
    pub visibility: Visibility,
    pub name: Spanned<TokenKey>,
    pub path: Path,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Function(Function),
    Record(Record),
    /// `mod` declaration: `(visibility, name)`.
    Mod(Visibility, TokenKey),
    ExternTrampoline(ExternTrampoline),
    ExternSource(ExternSource),
    Transform(TransformScript),
    Import(ImportDecl),
    Impl(ImplBlock),
}

/// An `impl<T: B> Type<args> { fn ... }` block: the block-level generics, the
/// target type head with its applied arguments, and the member functions.
#[derive(Clone, Debug)]
pub struct ImplBlock {
    /// Parses for uniformity with the other items; `pub impl` is rejected at
    /// elaboration (members carry their own visibility).
    pub visibility: Visibility,
    /// Each block-level generic is `(name, bounds)`.
    pub generics: SmallVec<[(TokenKey, Vec<Path>); 2]>,
    pub target: Path,
    pub target_args: SmallVec<[Type; 2]>,
    pub members: Vec<Spanned<Function>>,
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

    pub fn attributes(&self) -> Vec<Attribute> {
        nodes(&self.node)
            .filter(|node| node.kind() == AttrList)
            .map(attribute_of)
            .collect()
    }

    pub fn kind(&self) -> StmtKind {
        let node = &self.node;
        match node.kind() {
            FnStmt => StmtKind::Function(function_of(node)),
            StructStmt => StmtKind::Record(record_of(node, RecordKind::StructKind)),
            EnumStmt => StmtKind::Record(record_of(node, RecordKind::EnumKind)),
            ModStmt => StmtKind::Mod(visibility_of(node), key(name_after(node, ModKw))),
            ExternTrampolineStmt => StmtKind::ExternTrampoline(extern_of(node)),
            ExternSourceStmt => StmtKind::ExternSource(extern_source_of(node)),
            TransformStmt => StmtKind::Transform(transform_of(node)),
            ImportStmt => StmtKind::Import(import_of(node)),
            ImplStmt => StmtKind::Impl(impl_of(node)),
            k => unreachable!("unexpected statement node {k:?}"),
        }
    }
}

fn attribute_of(node: &ResolvedNode) -> Attribute {
    // Everything between the brackets, keeping `=` and string literals so
    // `key = "value"` pairs can be recognized positionally.
    let toks: Vec<_> = tokens(node)
        .filter(|token| {
            token.kind().is_ident_like()
                || token.kind() == SyntaxKind::Eq
                || token.kind() == StringLit
        })
        .collect();
    let mut args = Vec::new();
    let mut values = Vec::new();
    let mut idx = 1;
    while idx < toks.len() {
        let tok = toks[idx];
        if toks
            .get(idx + 1)
            .is_some_and(|next| next.kind() == SyntaxKind::Eq)
        {
            // `key = "value"`; a missing/ill-typed value already produced a
            // parse error, so silently skipping here is fine.
            if let Some(value) = toks.get(idx + 2).filter(|value| value.kind() == StringLit) {
                values.push((key(tok), unescape_string(value.text())));
            }
            idx += 3;
        } else {
            if tok.kind().is_ident_like() {
                args.push(key(tok));
            }
            idx += 1;
        }
    }
    Attribute {
        name: key(toks.first().expect("attribute name")),
        args,
        values,
        span: node_span(node),
    }
}

fn visibility_of(node: &ResolvedNode) -> Visibility {
    if tokens(node).any(|t| t.kind() == PubKw) {
        Visibility::Public
    } else {
        Visibility::Private
    }
}

/// Per-field visibility: the `pub` marker lives inside a nested VisFlag node
/// (never as a direct token, which would collide with a field named `pub`).
fn visibility_of_field(field: &ResolvedNode) -> Visibility {
    if child_node(field, VisFlag).is_some() {
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
    let params = child_node(node, ParamList).map_or_else(SmallVec::new, |list| {
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
        foreign_body: source_block_of(node),
    }
}

/// The `[{ ... }]` block directly under `node`, if any, with the `[`/`]`
/// stripped (the braces stay, so the text is a Rust block / item list).
fn source_block_of(node: &ResolvedNode) -> Option<SourceBlock> {
    let token = tokens(node).find(|t| t.kind() == RawMlirLiteral)?;
    let text = token.text();
    let span = token_span(token);
    Some(SourceBlock {
        body: text[1..text.len() - 1].to_owned(),
        body_span: Span {
            start: span.start + 1,
            end: span.end - 1,
        },
    })
}

fn impl_of(node: &ResolvedNode) -> ImplBlock {
    let target_ty = nodes(node)
        .find(|n| n.kind() == PathType)
        .expect("impl target type");
    let target = path_of(child_node(target_ty, PathKind).expect("target path"));
    let target_args = child_node(target_ty, TypeArgList)
        .map(|l| {
            nodes(l)
                .filter(|n| is_type_kind(n.kind()))
                .map(Type::new)
                .collect()
        })
        .unwrap_or_default();
    let members = nodes(node)
        .filter(|n| n.kind() == FnStmt)
        .map(|f| spanned(f, function_of(f)))
        .collect();
    ImplBlock {
        visibility: visibility_of(node),
        generics: generics_of(node),
        target,
        target_args,
        members,
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
                    // A `pub` marker sits inside a VisFlag node, so the
                    // direct-token scan still finds the field name.
                    let name = tokens(f)
                        .find(|t| t.kind().is_ident_like())
                        .expect("field name");
                    let flag = child_node(f, FieldFlag).is_some();
                    let vis = visibility_of_field(f);
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    spanned(f, (key(name), Type::new(ty), flag, vis))
                })
                .collect(),
        )
    } else if let Some(unnamed) = child_node(node, UnnamedFields) {
        RecordFields::Unnamed(
            nodes(unnamed)
                .filter(|n| n.kind() == UnnamedField)
                .map(|f| {
                    let flag = child_node(f, FieldFlag).is_some();
                    let vis = visibility_of_field(f);
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    spanned(f, (Type::new(ty), flag, vis))
                })
                .collect(),
        )
    } else if child_node(node, VariantList).is_none() {
        RecordFields::Opaque
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
    // `#[repr(fixed)]`: the only outer attribute the surface recognizes today.
    // An `AttrList` child holds the bracketed tokens verbatim (`repr` then its
    // `(fixed)` args); any other attribute is ignored here.
    let repr_fixed = nodes(node)
        .filter(|n| n.kind() == SyntaxKind::AttrList)
        .any(|attr| {
            let idents: Vec<&str> = tokens(attr)
                .filter(|t| t.kind().is_ident_like())
                .map(|t| t.text())
                .collect();
            idents.first() == Some(&"repr") && idents[1..].contains(&"fixed")
        });
    Record {
        name: key(name_after(node, intro)),
        ty_params: generics_of(node),
        fields,
        kind,
        visibility: visibility_of(node),
        default_cap,
        repr_fixed,
    }
}

fn extern_of(node: &ResolvedNode) -> ExternTrampoline {
    let mut strings = tokens(node).filter(|t| t.kind() == StringLit);
    let abi = unescape_string(strings.next().expect("ABI string").text());
    let sym = unescape_string(strings.next().expect("symbol string").text());
    let func = child_node(node, PathKind).expect("trampoline target path");
    let ty_args = child_node(node, TypeArgList)
        .map(|l| {
            nodes(l)
                .filter(|n| is_type_kind(n.kind()))
                .map(Type::new)
                .collect()
        })
        .unwrap_or_default();
    ExternTrampoline {
        name: sym,
        abi,
        func: path_of(func),
        ty_args,
    }
}

fn extern_source_of(node: &ResolvedNode) -> ExternSource {
    let abi = unescape_string(
        tokens(node)
            .find(|t| t.kind() == StringLit)
            .expect("ABI string")
            .text(),
    );
    ExternSource {
        abi,
        block: source_block_of(node).expect("source block"),
    }
}

fn import_of(node: &ResolvedNode) -> ImportDecl {
    let path_node = child_node(node, PathKind).expect("import target path");
    let path = path_of(path_node);
    // The bound name: the renaming identifier after `as` when present,
    // otherwise the target path's last segment.
    let name_token = tokens(node)
        .skip_while(|t| t.kind() != AsKw)
        .skip(1) // the `as` token itself is identifier-like
        .find(|t| t.kind().is_ident_like())
        .unwrap_or_else(|| {
            tokens(path_node)
                .filter(|t| t.kind().is_ident_like())
                .last()
                .expect("non-empty path")
        });
    let ns = token_span(name_token);
    ImportDecl {
        visibility: visibility_of(node),
        name: Spanned {
            value: key(name_token),
            start: ns.start,
            end: ns.end,
        },
        path,
    }
}

fn transform_of(node: &ResolvedNode) -> TransformScript {
    let token = tokens(node)
        .find(|token| token.kind() == RawMlirLiteral)
        .expect("transform literal");
    let text = token.text();
    let span = token_span(token);
    TransformScript {
        body: text[1..text.len() - 1].to_owned(),
        body_span: Span {
            start: span.start + 1,
            end: span.end - 1,
        },
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

/// View the root of a REPL expression parse
/// ([`reussir_syntax::parse_repl`] with [`reussir_syntax::ReplInputKind::Expr`])
/// as a typed expression. The root is a brace-less `BlockExpr`, so its
/// [`Expr::kind`] is an [`ExprKind::ExprSeq`] and `let x = 1; x + 1` inputs
/// get block semantics.
pub fn repl_expr(root: &ResolvedNode) -> Expr {
    assert_eq!(root.kind(), BlockExpr, "expected a REPL expression root");
    Expr::new(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> reussir_syntax::Parse {
        let parse = reussir_syntax::parse(source);
        assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
        parse
    }

    /// `prim_type` and the capability match re-encode the parser's
    /// `PRIM_TYPES`/`CAPABILITIES` sets by hand, each ending in
    /// `unreachable!`. Forcing every name the parser recognizes through the
    /// surface view turns a missed mirror entry into a test failure instead
    /// of a compiler panic on valid user input.
    #[test]
    fn every_primitive_type_reaches_the_surface_view() {
        for prim in reussir_syntax::PRIM_TYPES {
            let parse = parse(&format!("fn f(x: {prim}) {{ }}"));
            let prog = program(&parse.root);
            let StmtKind::Function(f) = prog[0].kind() else {
                panic!("expected a function for `{prim}`");
            };
            // `Type::kind` is the lazy view that runs `prim_type`.
            f.params[0].1.kind();
        }
    }

    #[test]
    fn impl_block_projects_visibility_target_generics_members() {
        let parse = parse(
            "impl<T: Num + Sync> Box<T> {\n\
                 pub fn get(self: Self) -> T { 1 }\n\
                 regional fn touch(self: [flex] Self) { 2 }\n\
             }",
        );
        let prog = program(&parse.root);
        let resolver = parse.resolver();
        let StmtKind::Impl(ib) = prog[0].kind() else {
            panic!("expected an impl block");
        };
        assert_eq!(ib.visibility, Visibility::Private);
        assert_eq!(resolver.resolve(ib.target.basename), "Box");
        assert!(ib.target.segments.is_empty());
        assert_eq!(ib.target_args.len(), 1);
        assert_eq!(ib.generics.len(), 1);
        let (g, bounds) = &ib.generics[0];
        assert_eq!(resolver.resolve(*g), "T");
        assert_eq!(bounds.len(), 2);

        assert_eq!(ib.members.len(), 2);
        let get = &ib.members[0].value;
        assert_eq!(resolver.resolve(get.name), "get");
        assert_eq!(get.visibility, Visibility::Public);
        assert!(!get.is_regional);
        let (pname, _, pflex) = &get.params[0];
        assert_eq!(resolver.resolve(*pname), "self");
        assert!(!pflex);
        let touch = &ib.members[1].value;
        assert_eq!(resolver.resolve(touch.name), "touch");
        assert_eq!(touch.visibility, Visibility::Private);
        assert!(touch.is_regional);
        let (_, _, rflex) = &touch.params[0];
        assert!(rflex, "the `[flex]` receiver flag projects");
    }

    #[test]
    fn every_capability_reaches_the_surface_view() {
        for cap in reussir_syntax::CAPABILITIES {
            let parse = parse(&format!("struct [{cap}] S {{ x: i64 }}"));
            let prog = program(&parse.root);
            let StmtKind::Record(r) = prog[0].kind() else {
                panic!("expected a record for `{cap}`");
            };
            // Every parser-recognized capability must map onto the enum;
            // whether it is *valid* as a record default is semi's business.
            let _ = r.default_cap;
        }
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

    #[test]
    fn views_imports() {
        let parse = parse(
            "import core::intrinsic::math;\nimport core::intrinsic::array as arr;\nimport core::intrinsic::math::sqrt as rt;",
        );
        let prog = program(&parse.root);
        assert_eq!(prog.len(), 3);
        let cases = [("math", "math"), ("arr", "array"), ("rt", "sqrt")];
        for (stmt, (name, basename)) in prog.iter().zip(cases) {
            let StmtKind::Import(decl) = stmt.kind() else {
                panic!("expected an import");
            };
            assert_eq!(parse.resolver().resolve(decl.name.value), name);
            assert_eq!(parse.resolver().resolve(decl.path.basename), basename);
            assert_eq!(decl.visibility, Visibility::Private);
        }
    }

    #[test]
    fn views_attributes_and_transform_scripts() {
        let source = "#[transform_anchor]\nfn f() {}\ntransform [{\n  transform.yield\n}];";
        let parse = parse(source);
        let prog = program(&parse.root);

        let attrs = prog[0].attributes();
        assert_eq!(attrs.len(), 1);
        assert_eq!(parse.resolver().resolve(attrs[0].name), "transform_anchor");
        assert!(attrs[0].args.is_empty());

        let StmtKind::Transform(script) = prog[1].kind() else {
            panic!("expected a transform script");
        };
        assert_eq!(script.body, "{\n  transform.yield\n}");
        assert_eq!(
            &source[script.body_span.start as usize..script.body_span.end as usize],
            script.body
        );
    }
}
