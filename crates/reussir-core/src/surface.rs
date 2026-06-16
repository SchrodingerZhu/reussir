//! The surface syntax tree — a typed, `serde`-deserializable mirror of the JSON
//! the parser (`reussir-syntax`) emits.
//!
//! The JSON follows aeson's `defaultOptions`: sum types are
//! `{"tag": "Ctor", "contents": …}` (a bare value for one field, an array for
//! several, `contents` omitted for nullary); all-nullary sums are bare strings;
//! record constructors inline their fields next to `"tag"`; `Maybe` is `null`;
//! spans wrap a value as `{"spanValue": …, "spanStartOffset": n, "spanEndOffset": n}`.
//!
//! Most of this maps onto serde's adjacently-tagged enums; the two enums with a
//! mix of inline-record and tagged constructors ([`StmtKind`], [`PatternKind`])
//! get hand-written `Deserialize` impls.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// A character-offset span into the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// A value carrying its source span, decoded from aeson's span wrapper.
#[derive(Clone, Debug, Deserialize)]
pub struct Spanned<T> {
    #[serde(rename = "spanValue")]
    pub value: T,
    #[serde(rename = "spanStartOffset")]
    pub start: u32,
    #[serde(rename = "spanEndOffset")]
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

/// A dotted name, e.g. `std::collections::List`.
#[derive(Clone, Debug, Deserialize)]
pub struct Path {
    #[serde(rename = "pathBasename")]
    pub basename: String,
    #[serde(rename = "pathSegments")]
    pub segments: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

/// The surface capability written on a record declaration.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum Capability {
    Value,
    Shared,
    Regional,
    Flex,
    Rigid,
    Field,
    Unspecified,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum RecordKind {
    StructKind,
    EnumKind,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

// ===== types =====

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum IntegralType {
    Signed(u16),
    Unsigned(u16),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum FpType {
    #[serde(rename = "IEEEFloat")]
    Ieee(u16),
    BFloat16,
    Float8,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum TypeKind {
    TypeIntegral(IntegralType),
    #[serde(rename = "TypeFP")]
    TypeFp(FpType),
    TypeBool,
    TypeStr,
    TypeUnit,
    /// A named type applied to arguments: `[path, [args]]`.
    TypeExpr(Path, Vec<Type>),
    /// A closure type: `[[arg types], ret]`.
    TypeArrow(Vec<Type>, Type),
}

/// A span-wrapped type (`{"tag":"TypeSpanned","contents":{spanValue,…}}`).
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum Type {
    #[serde(rename = "TypeSpanned")]
    Spanned(Spanned<Box<TypeKind>>),
}

impl Type {
    pub fn kind(&self) -> &TypeKind {
        let Type::Spanned(s) = self;
        &s.value
    }

    pub fn span(&self) -> Span {
        let Type::Spanned(s) = self;
        s.span()
    }
}

// ===== constants =====

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum Const {
    ConstInt(i64),
    ConstDouble(f64),
    ConstString(String),
    ConstBool(bool),
}

// ===== patterns =====

#[derive(Clone, Debug, Deserialize)]
pub struct Pattern {
    #[serde(rename = "patKind")]
    pub kind: PatternKind,
    #[serde(rename = "patGuard")]
    pub guard: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum PatternKind {
    Wildcard,
    Const(Const),
    Bind(String),
    Ctor(CtorPat),
}

/// A constructor pattern; its fields are inlined next to `"tag"` in the JSON.
#[derive(Clone, Debug, Deserialize)]
pub struct CtorPat {
    #[serde(rename = "patCtorPath")]
    pub path: Path,
    #[serde(rename = "patCtorArgs")]
    pub args: Vec<PatArg>,
    #[serde(rename = "patCtorHasEllipsis")]
    pub has_ellipsis: bool,
    #[serde(rename = "patCtorIsNamed")]
    pub is_named: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PatArg {
    #[serde(rename = "patCtorArgField")]
    pub field: Option<String>,
    #[serde(rename = "patCtorArgKind")]
    pub kind: PatternKind,
}

impl<'de> Deserialize<'de> for PatternKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        let tag = v
            .get("tag")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("pattern: missing tag"))?;
        let contents = || v.get("contents").cloned().unwrap_or(Value::Null);
        match tag {
            "WildcardPat" => Ok(PatternKind::Wildcard),
            "ConstPat" => Ok(PatternKind::Const(from_value(contents())?)),
            "BindPat" => Ok(PatternKind::Bind(from_value(contents())?)),
            "CtorPat" => Ok(PatternKind::Ctor(from_value(v)?)),
            other => Err(D::Error::custom(format!("unknown pattern tag {other}"))),
        }
    }
}

// ===== expressions =====

/// A lambda's payload (`{"args":…,"body":…,"retTy":…}`).
#[derive(Clone, Debug, Deserialize)]
pub struct Lambda {
    pub args: Vec<(String, Option<Type>)>,
    pub body: Expr,
    #[serde(rename = "retTy")]
    pub ret_ty: Option<Type>,
}

/// A direct function call.
#[derive(Clone, Debug, Deserialize)]
pub struct FuncCall {
    #[serde(rename = "funcCallName")]
    pub name: Path,
    /// Explicit type arguments; `null` (→ `None`) means "infer".
    #[serde(rename = "funcCallTyArgs")]
    pub ty_args: Vec<Option<Type>>,
    #[serde(rename = "funcCallArgs")]
    pub args: Vec<Expr>,
}

/// A constructor call.
#[derive(Clone, Debug, Deserialize)]
pub struct CtorCall {
    #[serde(rename = "ctorName")]
    pub name: Path,
    #[serde(rename = "ctorTyArgs")]
    pub ty_args: Vec<Option<Type>>,
    /// Each argument carries an optional field name (for named ctor syntax).
    #[serde(rename = "ctorArgs")]
    pub args: Vec<(Option<String>, Expr)>,
}

/// A field access in a chain or an assignment target.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum Access {
    Named(String),
    Unnamed(i64),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum ExprKind {
    ConstExpr(Const),
    ExprSeq(Vec<Expr>),
    If(Expr, Expr, Expr),
    Let(Spanned<String>, Option<(Type, bool)>, Expr),
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

/// A span-wrapped expression.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum Expr {
    #[serde(rename = "SpannedExpr")]
    Spanned(Spanned<Box<ExprKind>>),
}

impl Expr {
    pub fn kind(&self) -> &ExprKind {
        let Expr::Spanned(s) = self;
        &s.value
    }

    pub fn span(&self) -> Span {
        let Expr::Spanned(s) = self;
        s.span()
    }
}

// ===== statements =====

/// A function definition. `params`/`return` carry the `[flex]` flag as a bool.
#[derive(Clone, Debug, Deserialize)]
pub struct Function {
    #[serde(rename = "funcVisibility")]
    pub visibility: Visibility,
    #[serde(rename = "funcName")]
    pub name: String,
    /// Each generic is `(name, bounds)`.
    #[serde(rename = "funcGenerics")]
    pub generics: Vec<(String, Vec<Path>)>,
    #[serde(rename = "funcParams")]
    pub params: Vec<(String, Type, bool)>,
    #[serde(rename = "funcReturnType")]
    pub return_type: Option<(Type, bool)>,
    #[serde(rename = "funcIsRegional")]
    pub is_regional: bool,
    #[serde(rename = "funcBody")]
    pub body: Option<Expr>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum RecordFields {
    Named(Vec<Spanned<(String, Type, bool)>>),
    Unnamed(Vec<Spanned<(Type, bool)>>),
    Variants(Vec<Spanned<(String, Vec<Type>)>>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct Record {
    #[serde(rename = "recordName")]
    pub name: String,
    #[serde(rename = "recordTyParams")]
    pub ty_params: Vec<(String, Vec<Path>)>,
    #[serde(rename = "recordFields")]
    pub fields: RecordFields,
    #[serde(rename = "recordKind")]
    pub kind: RecordKind,
    #[serde(rename = "recordVisibility")]
    pub visibility: Visibility,
    #[serde(rename = "recordDefaultCap")]
    pub default_cap: Capability,
}

/// An `extern "C" trampoline` declaration; its fields are inlined in the JSON.
#[derive(Clone, Debug, Deserialize)]
pub struct ExternTrampoline {
    #[serde(rename = "etsName")]
    pub name: String,
    #[serde(rename = "etsABI")]
    pub abi: String,
    #[serde(rename = "etsFunc")]
    pub func: Path,
    #[serde(rename = "etsFuncTyArgs")]
    pub ty_args: Vec<Type>,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Function(Function),
    Record(Record),
    /// `mod` declaration: `(visibility, name)`.
    Mod(Visibility, String),
    ExternTrampoline(ExternTrampoline),
}

impl<'de> Deserialize<'de> for StmtKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        let tag = v
            .get("tag")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("stmt: missing tag"))?;
        let contents = || v.get("contents").cloned().unwrap_or(Value::Null);
        match tag {
            "FunctionStmt" => Ok(StmtKind::Function(from_value(contents())?)),
            "RecordStmt" => Ok(StmtKind::Record(from_value(contents())?)),
            "ModStmt" => {
                let (vis, name): (Visibility, String) = from_value(contents())?;
                Ok(StmtKind::Mod(vis, name))
            }
            "ExternTrampolineStmt" => Ok(StmtKind::ExternTrampoline(from_value(v)?)),
            other => Err(D::Error::custom(format!("unknown stmt tag {other}"))),
        }
    }
}

/// A span-wrapped statement.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tag", content = "contents")]
pub enum Stmt {
    #[serde(rename = "SpannedStmt")]
    Spanned(Spanned<Box<StmtKind>>),
}

impl Stmt {
    pub fn kind(&self) -> &StmtKind {
        let Stmt::Spanned(s) = self;
        &s.value
    }

    pub fn span(&self) -> Span {
        let Stmt::Spanned(s) = self;
        s.span()
    }
}

/// A whole parsed source file: a list of statements.
pub type Program = Vec<Stmt>;

/// Deserialize a [`Program`] from the parser's JSON value.
pub fn program_from_json(value: Value) -> Result<Program, serde_json::Error> {
    serde_json::from_value(value)
}

/// Helper: re-deserialize an owned [`Value`] into `T`, mapping the error into a
/// `serde::de::Error` so it composes inside hand-written `Deserialize` impls.
fn from_value<T, E>(value: Value) -> Result<T, E>
where
    T: serde::de::DeserializeOwned,
    E: serde::de::Error,
{
    serde_json::from_value(value).map_err(E::custom)
}

// ===== direct lowering from the lossless CST =====
//
// The elaborator's primary input path: build the typed AST straight from the
// parser's CST, with no JSON in between. The serialized path
// ([`program_from_json`]) exists too, and a test cross-checks the two agree.

mod lower {
    use reussir_syntax::ast::unescape_string;
    use reussir_syntax::diagnostics::SourceMap;
    use reussir_syntax::kind::{ResolvedNode, ResolvedToken, SyntaxKind, SyntaxKind::*};

    // Bare names resolve to `SyntaxKind` variants (used pervasively in matches);
    // surface types whose names collide with a kind are aliased.
    use super::{
        Access, BinOp, Const, CtorCall, Expr, ExprKind, ExternTrampoline, FpType, FuncCall,
        Function, IntegralType, Lambda, PatternKind, Program, Record, RecordFields, RecordKind,
        Span, Spanned, Stmt, StmtKind, Type, TypeKind, UnaryOp, Visibility,
    };
    use super::{
        Capability as AstCap, CtorPat as AstCtorPat, PatArg as AstPatArg, Path as AstPath,
        Pattern as AstPattern,
    };

    /// Lower a parsed `SourceFile` root into the typed surface program.
    pub fn program(root: &ResolvedNode, map: &SourceMap) -> Program {
        assert_eq!(root.kind(), SourceFile, "expected a SourceFile root");
        let l = Lowerer { map };
        root.children().map(|s| l.stmt(s)).collect()
    }

    struct Lowerer<'m> {
        map: &'m SourceMap,
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

    impl Lowerer<'_> {
        fn node_span(&self, node: &ResolvedNode) -> Span {
            let r = node.text_range();
            let (start, end) = self.map.char_span((r.start().into(), r.end().into()));
            Span { start, end }
        }

        fn token_span(&self, token: &ResolvedToken) -> Span {
            let r = token.text_range();
            let (start, end) = self.map.char_span((r.start().into(), r.end().into()));
            Span { start, end }
        }

        fn spanned<T>(&self, node: &ResolvedNode, value: T) -> Spanned<T> {
            let s = self.node_span(node);
            Spanned {
                value,
                start: s.start,
                end: s.end,
            }
        }

        // ----- statements -----

        fn stmt(&self, node: &ResolvedNode) -> Stmt {
            let kind = match node.kind() {
                FnStmt => StmtKind::Function(self.function(node)),
                StructStmt => StmtKind::Record(self.record(node, RecordKind::StructKind)),
                EnumStmt => StmtKind::Record(self.record(node, RecordKind::EnumKind)),
                ModStmt => {
                    let name = self.name_after(node, ModKw);
                    StmtKind::Mod(self.visibility(node), name.text().to_owned())
                }
                ExternTrampolineStmt => StmtKind::ExternTrampoline(self.extern_trampoline(node)),
                k => unreachable!("unexpected statement node {k:?}"),
            };
            Stmt::Spanned(self.spanned(node, Box::new(kind)))
        }

        fn visibility(&self, node: &ResolvedNode) -> Visibility {
            if tokens(node).any(|t| t.kind() == PubKw) {
                Visibility::Public
            } else {
                Visibility::Private
            }
        }

        fn name_after<'n>(&self, node: &'n ResolvedNode, kw: SyntaxKind) -> &'n ResolvedToken {
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

        fn generics(&self, node: &ResolvedNode) -> Vec<(String, Vec<AstPath>)> {
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
                        .filter(|n| n.kind() == Path)
                        .map(|p| self.path(p))
                        .collect();
                    (name.text().to_owned(), bounds)
                })
                .collect()
        }

        fn function(&self, node: &ResolvedNode) -> Function {
            let body = expr_children(node).last().map(|b| self.expr(b));
            let return_type = child_node(node, RetType).map(|r| {
                let flex = child_node(r, FlexFlag).is_some();
                let ty = nodes(r)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("return type");
                (self.type_(ty), flex)
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
                        (name.text().to_owned(), self.type_(ty), flex)
                    })
                    .collect()
            });
            Function {
                visibility: self.visibility(node),
                name: self.name_after(node, FnKw).text().to_owned(),
                generics: self.generics(node),
                params,
                return_type,
                is_regional: tokens(node).any(|t| t.kind() == RegionalKw),
                body,
            }
        }

        fn record(&self, node: &ResolvedNode, kind: RecordKind) -> Record {
            let intro = if kind == RecordKind::StructKind {
                StructKw
            } else {
                EnumKw
            };
            let default_cap = child_node(node, SyntaxKind::Capability)
                .and_then(|c| {
                    tokens(c).find(|t| t.kind().is_ident_like() || t.kind() == RegionalKw)
                })
                .map_or(AstCap::Shared, |t| match t.text() {
                    "shared" => AstCap::Shared,
                    "value" => AstCap::Value,
                    "flex" => AstCap::Flex,
                    "rigid" => AstCap::Rigid,
                    "field" => AstCap::Field,
                    "regional" => AstCap::Regional,
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
                            self.spanned(f, (name.text().to_owned(), self.type_(ty), flag))
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
                            self.spanned(f, (self.type_(ty), flag))
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
                                .map(|t| self.type_(t))
                                .collect();
                            self.spanned(v, (name.text().to_owned(), tys))
                        })
                        .collect(),
                )
            };
            Record {
                name: self.name_after(node, intro).text().to_owned(),
                ty_params: self.generics(node),
                fields,
                kind,
                visibility: self.visibility(node),
                default_cap,
            }
        }

        fn extern_trampoline(&self, node: &ResolvedNode) -> ExternTrampoline {
            let mut strings = tokens(node).filter(|t| t.kind() == StringLit);
            let abi = unescape_string(strings.next().expect("ABI string").text());
            let sym = unescape_string(strings.next().expect("symbol string").text());
            let func = child_node(node, Path).expect("trampoline target path");
            let ty_args = child_node(node, TypeArgList)
                .map(|l| {
                    nodes(l)
                        .filter(|n| is_type_kind(n.kind()))
                        .map(|t| self.type_(t))
                        .collect()
                })
                .unwrap_or_default();
            ExternTrampoline {
                name: sym,
                abi,
                func: self.path(func),
                ty_args,
            }
        }

        // ----- paths and types -----

        fn path(&self, node: &ResolvedNode) -> AstPath {
            let mut segments: Vec<String> = tokens(node)
                .filter(|t| t.kind().is_ident_like())
                .map(|t| t.text().to_owned())
                .collect();
            let basename = segments.pop().expect("non-empty path");
            AstPath { basename, segments }
        }

        fn type_(&self, node: &ResolvedNode) -> Type {
            let kind = match node.kind() {
                PrimType => prim_type(tokens(node).next().expect("prim type token").text()),
                PathType => {
                    let path = child_node(node, Path).expect("type path");
                    let args = child_node(node, TypeArgList)
                        .map(|l| {
                            nodes(l)
                                .filter(|n| is_type_kind(n.kind()))
                                .map(|t| self.type_(t))
                                .collect()
                        })
                        .unwrap_or_default();
                    TypeKind::TypeExpr(self.path(path), args)
                }
                ArrowType => {
                    let mut children = nodes(node).filter(|n| is_type_kind(n.kind()));
                    let lhs = children.next().expect("arrow lhs");
                    let rhs = children.next().expect("arrow rhs");
                    let args = if lhs.kind() == ParenTypeList {
                        nodes(lhs)
                            .filter(|n| is_type_kind(n.kind()))
                            .map(|t| self.type_(t))
                            .collect()
                    } else {
                        vec![self.type_(lhs)]
                    };
                    TypeKind::TypeArrow(args, self.type_(rhs))
                }
                ParenTypeList => {
                    let inner = nodes(node)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("parenthesized type");
                    return self.type_(inner);
                }
                k => unreachable!("unexpected type node {k:?}"),
            };
            Type::Spanned(self.spanned(node, Box::new(kind)))
        }

        // ----- patterns -----

        fn pattern(&self, node: &ResolvedNode) -> AstPattern {
            let kind = nodes(node)
                .find(|n| matches!(n.kind(), WildcardPat | BindPat | CtorPat | ConstPat))
                .expect("pattern kind");
            let guard = child_node(node, PatGuard)
                .map(|g| self.expr(expr_children(g).next().expect("guard expression")));
            AstPattern {
                kind: self.pattern_kind(kind),
                guard,
            }
        }

        fn pattern_kind(&self, node: &ResolvedNode) -> PatternKind {
            match node.kind() {
                WildcardPat => PatternKind::Wildcard,
                ConstPat => {
                    PatternKind::Const(self.constant(tokens(node).next().expect("constant token")))
                }
                BindPat => {
                    let path = child_node(node, Path).expect("binding path");
                    PatternKind::Bind(tokens(path).next().expect("binding name").text().to_owned())
                }
                CtorPat => {
                    let path = child_node(node, Path).expect("constructor path");
                    let (args, has_ellipsis, is_named) = match child_node(node, PatArgList) {
                        None => (Vec::new(), false, false),
                        Some(list) => {
                            let named = tokens(list).next().is_some_and(|t| t.kind() == LBrace);
                            let ellipsis = child_node(list, PatRest).is_some();
                            let args = nodes(list)
                                .filter(|n| n.kind() == PatArg)
                                .map(|arg| self.pat_arg(arg))
                                .collect();
                            (args, ellipsis, named)
                        }
                    };
                    PatternKind::Ctor(AstCtorPat {
                        path: self.path(path),
                        args,
                        has_ellipsis,
                        is_named,
                    })
                }
                k => unreachable!("unexpected pattern node {k:?}"),
            }
        }

        fn pat_arg(&self, node: &ResolvedNode) -> AstPatArg {
            let field = tokens(node)
                .find(|t| t.kind().is_ident_like())
                .map(|t| t.text().to_owned());
            let kind = nodes(node)
                .find(|n| matches!(n.kind(), WildcardPat | BindPat | CtorPat | ConstPat));
            let kind = match kind {
                Some(k) => self.pattern_kind(k),
                None => PatternKind::Bind(field.clone().expect("shorthand field name")),
            };
            AstPatArg { field, kind }
        }

        // ----- expressions -----

        fn expr(&self, node: &ResolvedNode) -> Expr {
            let kind = match node.kind() {
                ParenExpr => {
                    return self.expr(expr_children(node).next().expect("inner expression"));
                }
                LiteralExpr => {
                    ExprKind::ConstExpr(self.constant(tokens(node).next().expect("literal token")))
                }
                BlockExpr => ExprKind::ExprSeq(expr_children(node).map(|e| self.expr(e)).collect()),
                IfExpr => {
                    let mut parts = expr_children(node);
                    let cond = self.expr(parts.next().expect("condition"));
                    let then = self.expr(parts.next().expect("then branch"));
                    let other = self.expr(parts.next().expect("else branch"));
                    ExprKind::If(cond, then, other)
                }
                LetExpr => {
                    let name = self.let_name(node).expect("let binding name");
                    let flex = child_node(node, FlexFlag).is_some();
                    let ty = nodes(node)
                        .find(|n| is_type_kind(n.kind()))
                        .map(|t| (self.type_(t), flex));
                    let value = expr_children(node).last().expect("bound value");
                    let name_span = self.token_span(name);
                    ExprKind::Let(
                        Spanned {
                            value: name.text().to_owned(),
                            start: name_span.start,
                            end: name_span.end,
                        },
                        ty,
                        self.expr(value),
                    )
                }
                MatchExpr => {
                    let scrutinee = self.expr(expr_children(node).next().expect("scrutinee"));
                    let arms = nodes(node)
                        .filter(|n| n.kind() == MatchArm)
                        .map(|arm| {
                            let pat = child_node(arm, Pattern).expect("arm pattern");
                            let body = expr_children(arm).next().expect("arm body");
                            (self.pattern(pat), self.expr(body))
                        })
                        .collect();
                    ExprKind::Match(scrutinee, arms)
                }
                RegionalExpr => ExprKind::RegionalExpr(
                    self.expr(expr_children(node).next().expect("regional body")),
                ),
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
                                        .map(|t| self.type_(t));
                                    (name.text().to_owned(), ty)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let ret_ty = child_node(node, RetType).map(|r| {
                        self.type_(nodes(r).find(|n| is_type_kind(n.kind())).expect("ret"))
                    });
                    let body = self.expr(expr_children(node).last().expect("lambda body"));
                    ExprKind::Lambda(Lambda { args, body, ret_ty })
                }
                BinExpr => {
                    let op = tokens(node)
                        .find_map(|t| binary_op(t.kind()))
                        .expect("binary operator");
                    let mut operands = expr_children(node);
                    let lhs = self.expr(operands.next().expect("lhs"));
                    let rhs = self.expr(operands.next().expect("rhs"));
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
                    let operand = self.expr(expr_children(node).next().expect("operand"));
                    ExprKind::UnaryOpExpr(op, operand)
                }
                CastExpr => {
                    let operand = expr_children(node).next().expect("cast operand");
                    let ty = nodes(node)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("cast type");
                    ExprKind::Cast(self.type_(ty), self.expr(operand))
                }
                CallExpr => {
                    let callee = self.expr(expr_children(node).next().expect("callee"));
                    let args = child_node(node, ArgList)
                        .map(|l| expr_children(l).map(|e| self.expr(e)).collect())
                        .unwrap_or_default();
                    ExprKind::CallExpr(callee, args)
                }
                AccessChain => {
                    let base = self.expr(expr_children(node).next().expect("access base"));
                    ExprKind::AccessChain(base, self.access_segs(node))
                }
                AssignExpr => {
                    let mut operands = expr_children(node);
                    let lhs = self.expr(operands.next().expect("assignment target"));
                    let rhs = self.expr(operands.next().expect("assignment value"));
                    let access = self
                        .access_segs(node)
                        .into_iter()
                        .next()
                        .expect("assignment field");
                    ExprKind::Assign(lhs, access, rhs)
                }
                VarExpr => ExprKind::Var(self.path(child_node(node, Path).expect("variable path"))),
                FuncCallExpr => {
                    let path = child_node(node, Path).expect("function path");
                    let args = child_node(node, ArgList)
                        .map(|l| expr_children(l).map(|e| self.expr(e)).collect())
                        .unwrap_or_default();
                    ExprKind::FuncCallExpr(FuncCall {
                        name: self.path(path),
                        ty_args: self.type_args(node),
                        args,
                    })
                }
                CtorCallExpr => {
                    let path = child_node(node, Path).expect("constructor path");
                    let args = child_node(node, CtorArgList)
                        .map(|l| {
                            nodes(l)
                                .filter(|n| n.kind() == CtorArg)
                                .map(|arg| {
                                    let named = tokens(arg).any(|t| t.kind() == Colon);
                                    let field = if named {
                                        tokens(arg)
                                            .find(|t| t.kind().is_ident_like())
                                            .map(|t| t.text().to_owned())
                                    } else {
                                        None
                                    };
                                    let value =
                                        self.expr(expr_children(arg).next().expect("arg value"));
                                    (field, value)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    ExprKind::CtorCallExpr(CtorCall {
                        name: self.path(path),
                        ty_args: self.type_args(node),
                        args,
                    })
                }
                k => unreachable!("unexpected expression node {k:?}"),
            };
            Expr::Spanned(self.spanned(node, Box::new(kind)))
        }

        fn let_name<'n>(&self, node: &'n ResolvedNode) -> Option<&'n ResolvedToken> {
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

        fn type_args(&self, node: &ResolvedNode) -> Vec<Option<Type>> {
            let Some(list) = child_node(node, TypeArgList) else {
                return Vec::new();
            };
            nodes(list)
                .filter(|n| is_type_kind(n.kind()) || n.kind() == InferType)
                .map(|n| {
                    if n.kind() == InferType {
                        None
                    } else {
                        Some(self.type_(n))
                    }
                })
                .collect()
        }

        fn access_segs(&self, node: &ResolvedNode) -> Vec<Access> {
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
                        k if k.is_ident_like() => out.push(Access::Named(t.text().to_owned())),
                        k => unreachable!("unexpected access token {k:?}"),
                    }
                }
            }
            out
        }

        fn constant(&self, token: &ResolvedToken) -> Const {
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
        let int = |t| TypeKind::TypeIntegral(t);
        let fp = |t| TypeKind::TypeFp(t);
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
}

pub use lower::program as lower;

#[cfg(test)]
mod tests {
    use super::*;

    /// Lower a source string directly from the CST (the primary path), and
    /// assert it agrees with the serialized (JSON) path so neither can drift.
    fn surface_of(source: &str) -> Program {
        let parse = reussir_syntax::parse(source);
        assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
        let map = reussir_syntax::diagnostics::SourceMap::new(source);

        let direct = lower(&parse.root, &map);
        let serialized = program_from_json(parse.to_json(&map)).expect("surface deserialization");
        // The two input paths must produce identical trees.
        assert_eq!(
            format!("{direct:#?}"),
            format!("{serialized:#?}"),
            "direct and serialized surface trees disagree"
        );
        direct
    }

    #[test]
    fn round_trips_a_function_with_calls() {
        let prog = surface_of(
            "fn f(x: i32, y: i32) -> i32 {\n    let z = x * 2 + y % 3;\n    if (z >= 0 && !(z == 4)) { id<i32>(z) } else { -z }\n}",
        );
        assert_eq!(prog.len(), 1);
        let StmtKind::Function(f) = prog[0].kind() else {
            panic!("expected a function");
        };
        assert_eq!(f.name, "f");
        assert_eq!(f.params.len(), 2);
        assert!(f.return_type.is_some());
        assert!(f.body.is_some());
    }

    #[test]
    fn round_trips_enum_match_and_ctor() {
        let prog = surface_of(
            "enum List<T> { Nil, Cons(T, List<T>) }\nfn h(a: List<i32>) -> i32 {\n    match a {\n        List::Nil => 0,\n        List::Cons(x, xs) => x\n    }\n}",
        );
        assert_eq!(prog.len(), 2);
        let StmtKind::Record(r) = prog[0].kind() else {
            panic!("expected a record");
        };
        assert_eq!(r.name, "List");
        assert_eq!(r.kind, RecordKind::EnumKind);
        assert!(matches!(r.fields, RecordFields::Variants(_)));
    }

    #[test]
    fn round_trips_regional_and_struct() {
        let prog = surface_of(
            "struct [regional] L<T> { v: T, next: [field] L<T> }\nregional fn push<T>(c : [flex] L<T>, e : T) { c->next := Nullable::NonNull{c} }",
        );
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
