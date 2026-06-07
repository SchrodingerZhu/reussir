use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WithSpan<T> {
    pub value: T,
    pub span: Span,
}

impl<T> WithSpan<T> {
    pub fn new(value: T, start: usize, end: usize) -> Self {
        Self {
            value,
            span: Span::new(start, end),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Identifier(pub String);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Path {
    pub basename: Identifier,
    pub segments: Vec<Identifier>,
}

impl Path {
    pub fn from_parts(parts: Vec<Identifier>) -> Self {
        let mut parts = parts;
        let basename = parts
            .pop()
            .expect("path construction requires at least one identifier");
        Self {
            basename,
            segments: parts,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Shared,
    Value,
    Flex,
    Rigid,
    Field,
    Regional,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegralType {
    Signed(i16),
    Unsigned(i16),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatingPointType {
    Ieee(i16),
    BFloat16,
    Float8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Type {
    Expr { path: Path, args: Vec<Type> },
    Integral(IntegralType),
    Float(FloatingPointType),
    Bool,
    Str,
    Unit,
    Arrow { args: Vec<Type>, ret: Box<Type> },
    Bottom,
    Spanned(WithSpan<Box<Type>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Constant {
    Int(i64),
    Double(String),
    String(String),
    Bool(bool),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Named(Identifier),
    Unnamed(i64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub kind: PatternKind,
    pub guard: Option<Box<Expr>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternKind {
    Wildcard,
    Bind(Identifier),
    Ctor {
        path: Path,
        args: Vec<PatternCtorArg>,
        has_ellipsis: bool,
        is_named: bool,
    },
    Const(Constant),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternCtorArg {
    pub field: Option<Identifier>,
    pub kind: PatternKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CtorCall {
    pub name: Path,
    pub ty_args: Vec<Option<Type>>,
    pub args: Vec<(Option<Identifier>, Expr)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FuncCall {
    pub name: Path,
    pub ty_args: Vec<Option<Type>>,
    pub args: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LambdaExpr {
    pub args: Vec<(Identifier, Option<Type>)>,
    pub body: Box<Expr>,
    pub ret_ty: Option<Type>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    Const(Constant),
    BinOp {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Cast {
        ty: Type,
        expr: Box<Expr>,
    },
    Let {
        name: WithSpan<Identifier>,
        ty: Option<(Type, bool)>,
        value: Box<Expr>,
    },
    Seq(Vec<Expr>),
    Lambda(LambdaExpr),
    Match {
        scrutinee: Box<Expr>,
        cases: Vec<(Pattern, Expr)>,
    },
    Var(Path),
    FuncCall(FuncCall),
    Regional(Box<Expr>),
    CtorCall(CtorCall),
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    AccessChain {
        base: Box<Expr>,
        accesses: Vec<Access>,
    },
    Spanned(WithSpan<Box<Expr>>),
    Assign {
        base: Box<Expr>,
        access: Access,
        value: Box<Expr>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordFields {
    Named(Vec<WithSpan<(Identifier, Type, bool)>>),
    Unnamed(Vec<WithSpan<(Type, bool)>>),
    Variants(Vec<WithSpan<(Identifier, Vec<Type>)>>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Struct,
    Enum,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub name: Identifier,
    pub ty_params: Vec<(Identifier, Vec<Path>)>,
    pub fields: RecordFields,
    pub kind: RecordKind,
    pub visibility: Visibility,
    pub default_cap: Capability,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub visibility: Visibility,
    pub name: Identifier,
    pub generics: Vec<(Identifier, Vec<Path>)>,
    pub params: Vec<(Identifier, Type, bool)>,
    pub return_type: Option<(Type, bool)>,
    pub is_regional: bool,
    pub body: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stmt {
    Function(Function),
    Record(Record),
    ExternTrampoline {
        name: Identifier,
        abi: String,
        func: Path,
        func_ty_args: Vec<Type>,
    },
    Mod {
        visibility: Visibility,
        name: Identifier,
    },
    Spanned(WithSpan<Box<Stmt>>),
}

pub type Program = Vec<Stmt>;
