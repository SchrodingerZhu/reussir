//! The Semi HIR: a typed, still-polymorphic intermediate representation.
//!
//! Every [`Expr`] carries its [`Ty`], which may contain generics or holes.
//! Pattern matches are compiled to a [`DecisionTree`].

use crate::semi::ty::{GenericId, Ty};
use crate::surface::Span;
use crate::utils::string::StringToken;

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
        name: String,
        span: Option<Span>,
        value: Box<Expr<'tcx>>,
    },
    Seq(Vec<Expr<'tcx>>),
    /// A direct function call with (possibly hole) type arguments.
    FuncCall {
        target: String,
        ty_args: Vec<Ty<'tcx>>,
        args: Vec<Expr<'tcx>>,
        regional: bool,
    },
    /// A struct (compound) constructor call.
    CompoundCall {
        target: String,
        ty_args: Vec<Ty<'tcx>>,
        args: Vec<Expr<'tcx>>,
    },
    /// An enum variant constructor call.
    VariantCall {
        target: String,
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
    pub name: String,
    pub generics: Vec<(String, GenericId)>,
    pub params: Vec<(String, VarId, Ty<'tcx>)>,
    pub return_ty: Ty<'tcx>,
    pub is_regional: bool,
    pub body: Option<Expr<'tcx>>,
    pub span: Option<Span>,
}
