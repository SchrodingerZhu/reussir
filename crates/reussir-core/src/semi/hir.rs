//! The Semi HIR: a typed, still-polymorphic intermediate representation.
//!
//! Every [`Expr`] carries its [`Ty`], which may contain generics or holes.
//! Pattern matches are compiled to a [`DecisionTree`].

use reussir_syntax::kind::TokenKey;

use crate::semi::ty::{DefId, GenericId, Ty};
use crate::surface::Span;
use crate::utils::string::StringToken;

// The textual HIR ser/de, as submodules of the IR they (de)serialize: the
// `pprint` serializer ([`print`]), the owned grammar AST ([`raw`]) the lalrpop
// [`grammar`] builds, and the re-intern pass ([`build`]).
pub mod build;
pub mod print;
pub mod raw;

// The lalrpop-generated parser (from `semi/hir/grammar.lalrpop`), fed by the
// shared [`crate::ir_lex`] logos lexer.
lalrpop_util::lalrpop_mod!(
    #[allow(clippy::all, dead_code, unused_imports)]
    pub grammar,
    "/semi/hir/grammar.rs"
);

/// A local variable, unique within a function body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarId(pub u32);

/// A unique id for an expression node (used by later phases for keying).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(pub u32);

/// Arithmetic / boolean binary operators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
}

/// Comparison operators (all produce `bool`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

/// A type-checked expression.
#[derive(Clone, Debug)]
pub struct Expr<'tcx> {
    pub kind: ExprKind<'tcx>,
    pub ty: Ty<'tcx>,
    pub span: Option<Span>,
    pub id: ExprId,
}

/// The closure form: captured variables, parameters, and the body.
///
/// Lowering convention (load-bearing): the runtime closure has no separate
/// capture environment. Captures and parameters share a single input list with
/// the captures as the *leading* inputs, so a closure lowers to a value of
/// closure type over `(captures ++ params) -> ret`. Each captured value is then
/// supplied with one `closure.apply` at construction time, leaving a
/// user-visible closure of type `(params) -> ret`. The order of `captures` is
/// therefore significant and must be preserved through monomorphization and
/// lowering.
#[derive(Clone, Debug)]
pub struct ClosureExpr<'tcx> {
    pub captures: Vec<(VarId, Ty<'tcx>)>,
    pub params: Vec<(VarId, Ty<'tcx>)>,
    pub body: Box<Expr<'tcx>>,
}

#[derive(Clone, Debug)]
pub enum ExprKind<'tcx> {
    /// An interned string literal.
    GlobalStr(StringToken),
    /// An integer literal (its type is a `Num`-bounded hole until solved).
    ConstInt(i128),
    /// A floating-point literal.
    ConstFloat(f64),
    ConstBool(bool),
    Negate(Box<Expr<'tcx>>),
    Not(Box<Expr<'tcx>>),
    Arith(Box<Expr<'tcx>>, ArithOp, Box<Expr<'tcx>>),
    Cmp(Box<Expr<'tcx>>, CmpOp, Box<Expr<'tcx>>),
    Cast(Box<Expr<'tcx>>, Ty<'tcx>),
    If(Box<Expr<'tcx>>, Box<Expr<'tcx>>, Box<Expr<'tcx>>),
    Var(VarId),
    /// A `region-run` body; the result has been frozen to a rigid capability.
    RegionRun(Box<Expr<'tcx>>),
    /// A nested field projection (a path of field indices).
    Proj(Box<Expr<'tcx>>, Vec<u32>),
    Match(Box<Expr<'tcx>>, DecisionTree<'tcx>),
    /// Assign `src` into field `field` of the flex record `dst`.
    Assign(Box<Expr<'tcx>>, u32, Box<Expr<'tcx>>),
    Let {
        var: VarId,
        name: TokenKey,
        span: Option<Span>,
        value: Box<Expr<'tcx>>,
    },
    Seq(Vec<Expr<'tcx>>),
    /// A direct function call with (possibly hole) type arguments.
    FuncCall {
        target: DefId,
        ty_args: Vec<Ty<'tcx>>,
        args: Vec<Expr<'tcx>>,
        regional: bool,
    },
    /// A struct (compound) constructor call.
    CompoundCall {
        target: DefId,
        ty_args: Vec<Ty<'tcx>>,
        args: Vec<Expr<'tcx>>,
    },
    /// An enum variant constructor call.
    VariantCall {
        target: DefId,
        ty_args: Vec<Ty<'tcx>>,
        variant: usize,
        args: Vec<Expr<'tcx>>,
    },
    /// `Nullable::NonNull{e}` (Some) or `Nullable::Null` (None).
    NullableCall(Option<Box<Expr<'tcx>>>),
    Closure(ClosureExpr<'tcx>),
    ClosureCall {
        target: Box<Expr<'tcx>>,
        args: Vec<Expr<'tcx>>,
    },
    /// Error-recovery placeholder.
    Poison,
}

// ===== decision trees (compiled pattern matches) =====

/// A path from the scrutinee to a sub-value: a sequence of field indices.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PatVarRef(pub Vec<u32>);

/// A compiled pattern match.
#[derive(Clone, Debug)]
pub enum DecisionTree<'tcx> {
    /// Non-exhaustive: no arm covers some input.
    Uncovered,
    /// Unreachable arm.
    Unreachable,
    /// A matched arm body, with pattern bindings mapped to scrutinee paths.
    Leaf {
        body: Box<Expr<'tcx>>,
        bindings: Vec<(VarId, PatVarRef)>,
    },
    /// A guard: take `success` if `guard` holds, else `failure`.
    Guard {
        bindings: Vec<(VarId, PatVarRef)>,
        guard: Box<Expr<'tcx>>,
        success: Box<DecisionTree<'tcx>>,
        failure: Box<DecisionTree<'tcx>>,
    },
    /// Switch on the value at `scrutinee`.
    Switch {
        scrutinee: PatVarRef,
        cases: SwitchCases<'tcx>,
    },
}

#[derive(Clone, Debug)]
pub enum SwitchCases<'tcx> {
    Int {
        cases: Vec<(i128, DecisionTree<'tcx>)>,
        default: Box<DecisionTree<'tcx>>,
    },
    Bool {
        if_true: Box<DecisionTree<'tcx>>,
        if_false: Box<DecisionTree<'tcx>>,
    },
    /// One sub-tree per variant index of the scrutinee's enum.
    Ctor(Vec<DecisionTree<'tcx>>),
    String {
        cases: Vec<(StringToken, DecisionTree<'tcx>)>,
        default: Box<DecisionTree<'tcx>>,
    },
    Nullable {
        non_null: Box<DecisionTree<'tcx>>,
        null: Box<DecisionTree<'tcx>>,
    },
}

/// A type-checked function, the unit of Semi output.
#[derive(Clone, Debug)]
pub struct Function<'tcx> {
    pub def: DefId,
    pub name: TokenKey,
    /// Source visibility; carried so the Full phase can give exported (`pub`)
    /// instances external linkage and keep internal ones private.
    pub visibility: crate::surface::Visibility,
    pub generics: Vec<(TokenKey, GenericId)>,
    /// Generics used at a `[flex]` position; their instantiations must be
    /// regional records (checked at monomorphization). See
    /// [`crate::semi::ctxt::FuncProto::regional_generics`].
    pub regional_generics: Vec<GenericId>,
    pub params: Vec<(TokenKey, VarId, Ty<'tcx>)>,
    pub return_ty: Ty<'tcx>,
    pub is_regional: bool,
    pub body: Option<Expr<'tcx>>,
    pub span: Option<Span>,
}
