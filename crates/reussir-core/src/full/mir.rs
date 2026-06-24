//! The Full *MIR*: the monomorphized, symbol-indexed mid-level IR.
//!
//! Where the Semi [`hir`](crate::semi::hir) keeps generics open and dispatches
//! calls by `(DefId, ty_args)`, the MIR is **fully ground and generics-erased**:
//! every call names its callee by an interned [`Symbol`] (the v0 mangled name),
//! no node carries type arguments, and no type contains a `Generic`/`Hole`.
//! Types survive only as ground [`Ty`] layout data on nodes, never as a dispatch
//! key. This is the representation the backend lowers from — `call @symbol` is a
//! direct read — and the one the ownership pass will thread its inc/dec/drop/
//! reuse operations through.
//!
//! The node tree is **arena-allocated and `Copy`**, like the type interner:
//! children are `&'tcx Expr` and lists are `&'tcx [Expr]` / `&'tcx [u32]`,
//! allocated through [`TyCtxt::alloc`](crate::semi::ty::TyCtxt::alloc). Produced
//! by [`crate::full::mono`], which *lowers* the Semi HIR into it (it is not the
//! HIR with ground types — it is a distinct tree).

use lasso::{Rodeo, Spur};
use reussir_syntax::kind::TokenKey;

use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::ty::Ty;
use crate::surface::{Span, Visibility};
use crate::utils::string::StringToken;

// The textual MIR ser/de, as submodules of the IR they (de)serialize: the
// `pprint` serializer ([`print`]), the owned grammar AST ([`raw`]) the
// lalrpop [`grammar`] builds, and the re-intern pass ([`build`]).
pub mod build;
pub mod print;
pub mod raw;

// The lalrpop-generated parser (from `full/mir/grammar.lalrpop`), fed by the
// shared [`crate::ir_lex`] logos lexer.
lalrpop_util::lalrpop_mod!(
    #[allow(clippy::all, dead_code, unused_imports)]
    pub grammar,
    "/full/mir/grammar.rs"
);

/// An interned linker symbol (a v0 mangled name). `Copy`; resolve it to text via
/// [`Program::symbol`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub Spur);

/// A path from a scrutinee to a sub-value: a sequence of field indices,
/// arena-allocated (the MIR counterpart of [`crate::semi::hir::PatVarRef`]).
pub type Path<'tcx> = &'tcx [u32];

/// The whole monomorphized program: ground functions, the ground record
/// instances their layouts need, and exported trampolines — all keyed by
/// [`Symbol`]. The owning [`Rodeo`] resolves those symbols back to text.
pub struct Program<'tcx> {
    pub functions: Vec<Function<'tcx>>,
    pub records: Vec<RecordInstance<'tcx>>,
    pub trampolines: Vec<Trampoline>,
    /// Interner backing every [`Symbol`] in this program.
    pub symbols: Rodeo,
}

impl<'tcx> Program<'tcx> {
    /// Resolve an interned symbol to its mangled text.
    pub fn symbol(&self, sym: Symbol) -> &str {
        self.symbols.resolve(&sym.0)
    }
}

/// A monomorphized function: a ground signature and body under its v0 symbol.
#[derive(Clone, Debug)]
pub struct Function<'tcx> {
    pub symbol: Symbol,
    pub visibility: Visibility,
    pub is_regional: bool,
    pub params: Vec<Param<'tcx>>,
    pub return_ty: Ty<'tcx>,
    /// `None` for a declared-but-undefined function.
    pub body: Option<&'tcx Expr<'tcx>>,
}

/// A function parameter: source name, local variable, and ground type.
#[derive(Clone, Copy, Debug)]
pub struct Param<'tcx> {
    pub name: TokenKey,
    pub var: VarId,
    pub ty: Ty<'tcx>,
}

/// A ground record instance whose layout the backend must materialize. Keyed by
/// [`Symbol`]; the ground type carries the fields needed to compute that layout.
#[derive(Clone, Copy, Debug)]
pub struct RecordInstance<'tcx> {
    pub symbol: Symbol,
    /// The ground record type (`def` + args + capability) for layout.
    pub ty: Ty<'tcx>,
}

/// An exported C-ABI trampoline aliasing a concrete function instance.
#[derive(Clone, Debug)]
pub struct Trampoline {
    pub export: Symbol,
    pub abi: String,
    pub target: Symbol,
}

/// A monomorphized expression: a ground type, the structure, and a source span.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'tcx> {
    pub kind: ExprKind<'tcx>,
    pub ty: Ty<'tcx>,
    pub span: Option<Span>,
}

/// The closure form; see [`crate::semi::hir::ClosureExpr`] for the
/// captures-as-leading-inputs lowering convention, which the MIR preserves.
#[derive(Clone, Copy, Debug)]
pub struct ClosureExpr<'tcx> {
    pub captures: &'tcx [(VarId, Ty<'tcx>)],
    pub params: &'tcx [(VarId, Ty<'tcx>)],
    pub body: &'tcx Expr<'tcx>,
}

#[derive(Clone, Copy, Debug)]
pub enum ExprKind<'tcx> {
    GlobalStr(StringToken),
    ConstInt(i128),
    ConstFloat(f64),
    ConstBool(bool),
    Var(VarId),
    Negate(&'tcx Expr<'tcx>),
    Not(&'tcx Expr<'tcx>),
    Arith(&'tcx Expr<'tcx>, ArithOp, &'tcx Expr<'tcx>),
    Cmp(&'tcx Expr<'tcx>, CmpOp, &'tcx Expr<'tcx>),
    Cast(&'tcx Expr<'tcx>, Ty<'tcx>),
    If(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>),
    /// A `region-run` body; its result has been frozen to a rigid capability.
    RegionRun(&'tcx Expr<'tcx>),
    /// A nested field projection (a path of field indices).
    Proj(&'tcx Expr<'tcx>, Path<'tcx>),
    /// Assign `src` into field `field` of the flex record `dst`.
    Assign(&'tcx Expr<'tcx>, u32, &'tcx Expr<'tcx>),
    Let {
        var: VarId,
        name: TokenKey,
        value: &'tcx Expr<'tcx>,
    },
    Seq(&'tcx [Expr<'tcx>]),
    /// A direct call to a ground function instance, by interned symbol.
    Call {
        callee: Symbol,
        args: &'tcx [Expr<'tcx>],
        regional: bool,
    },
    /// A struct (compound) constructor for a ground record instance.
    Ctor {
        record: Symbol,
        args: &'tcx [Expr<'tcx>],
    },
    /// An enum variant constructor for a ground record instance.
    Variant {
        record: Symbol,
        variant: usize,
        args: &'tcx [Expr<'tcx>],
    },
    /// `Nullable::NonNull{e}` (Some) or `Nullable::Null` (None).
    NullableCall(Option<&'tcx Expr<'tcx>>),
    Closure(ClosureExpr<'tcx>),
    ClosureCall {
        target: &'tcx Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    },
    Match(&'tcx Expr<'tcx>, DecisionTree<'tcx>),
    /// Error-recovery placeholder.
    Poison,
}

/// A pattern binding: a local variable bound to a scrutinee [`Path`].
pub type Binding<'tcx> = (VarId, Path<'tcx>);

/// A compiled pattern match, mirroring [`crate::semi::hir::DecisionTree`] but
/// arena-allocated over MIR expressions.
#[derive(Clone, Copy, Debug)]
pub enum DecisionTree<'tcx> {
    Uncovered,
    Unreachable,
    Leaf {
        body: &'tcx Expr<'tcx>,
        bindings: &'tcx [Binding<'tcx>],
    },
    Guard {
        bindings: &'tcx [Binding<'tcx>],
        guard: &'tcx Expr<'tcx>,
        success: &'tcx DecisionTree<'tcx>,
        failure: &'tcx DecisionTree<'tcx>,
    },
    Switch {
        scrutinee: Path<'tcx>,
        cases: SwitchCases<'tcx>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum SwitchCases<'tcx> {
    Int {
        cases: &'tcx [(i128, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
    },
    Bool {
        if_true: &'tcx DecisionTree<'tcx>,
        if_false: &'tcx DecisionTree<'tcx>,
    },
    Ctor(&'tcx [DecisionTree<'tcx>]),
    String {
        cases: &'tcx [(StringToken, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
    },
    Nullable {
        non_null: &'tcx DecisionTree<'tcx>,
        null: &'tcx DecisionTree<'tcx>,
    },
}
