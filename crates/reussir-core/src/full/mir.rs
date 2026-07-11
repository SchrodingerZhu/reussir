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
use reussir_syntax::source::FileId;

use crate::literal::{FloatLit, Integer};
use crate::semi::ctxt::DefaultCap;
use crate::semi::hir::{ArithOp, CmpOp, ExprId, VarId};
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
    pub string_literals: Vec<(StringToken, String)>,
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
    /// The file the function's spans index in the compilation's source cache
    /// (drives per-function debug locations). Serialized in the textual MIR as
    /// `in <id>` against the dump's source-file table, so the association
    /// survives a dump/re-parse round trip.
    pub file: FileId,
}

/// A function parameter: source name, local variable, and ground type.
#[derive(Clone, Copy, Debug)]
pub struct Param<'tcx> {
    pub name: TokenKey,
    pub var: VarId,
    pub ty: Ty<'tcx>,
}

/// A ground record instance whose layout the backend materializes. Keyed by
/// [`Symbol`]; the [`layout`](Self::layout) carries the ground member types so a
/// program parsed from textual MIR is lowerable without re-consulting the
/// elaborator (the [`ty`](Self::ty) is the nominal record handle used to match a
/// record-typed value back to its instance).
#[derive(Clone, Copy, Debug)]
pub struct RecordInstance<'tcx> {
    pub symbol: Symbol,
    /// The nominal ground record type (`def` + args), capability-canonicalized.
    pub ty: Ty<'tcx>,
    /// The capability the record declares by default — selects the lowering
    /// (only `Value` records lower today; `Shared`/`Regional` await the rc work).
    pub default_cap: DefaultCap,
    /// `#[repr(fixed)]`: pin uniform max-arm box sizing for this variant instead
    /// of the default per-constructor sizing (`header + arm[k]`). Only ever set
    /// for enums; always `false` for compounds. Threaded through to the dialect
    /// variant record type's `fixed` flag.
    pub repr_fixed: bool,
    /// The ground field layout.
    pub layout: RecordLayout<'tcx>,
}

/// A ground record's shape: a struct's ordered fields, or an enum's variants.
#[derive(Clone, Copy, Debug)]
pub enum RecordLayout<'tcx> {
    Compound(&'tcx [Member<'tcx>]),
    Variant(&'tcx [VariantDef<'tcx>]),
}

/// One compound field: its ground type and whether it is a mutable `[field]`
/// link (only regional records carry these).
#[derive(Clone, Copy, Debug)]
pub struct Member<'tcx> {
    pub ty: Ty<'tcx>,
    pub is_field: bool,
    /// The source field name (interned), for debug info. `None` for a tuple
    /// (unnamed) field, or when the layout was rebuilt from textual MIR (which
    /// does not carry field names). Not part of the textual round-trip.
    pub name: Option<Symbol>,
}

/// One enum variant: its source name, the mangled symbol of its payload record
/// (the enum path with the variant nested as a final segment — see
/// [`crate::full::mangle::Mangler::mangle_variant`]), and ordered field types.
/// The payload symbol is carried so a program rebuilt from textual MIR names the
/// per-case record identically without re-running the mangler.
#[derive(Clone, Copy, Debug)]
pub struct VariantDef<'tcx> {
    pub name: Symbol,
    pub symbol: Symbol,
    pub fields: &'tcx [Ty<'tcx>],
}

/// An exported C-ABI trampoline aliasing a concrete function instance.
#[derive(Clone, Debug)]
pub struct Trampoline {
    pub export: Symbol,
    pub abi: String,
    pub target: Symbol,
}

/// A monomorphized expression: a ground type, the structure, and a source span.
///
/// The [`id`](Self::id) is a per-program-unique **anchor** the ownership pass
/// keys its inc/dec/drop placement on (see [`crate::full::ownership`]); the MIR
/// itself stays immutable, so analyses record their results in side-tables keyed
/// by this id rather than rewriting the tree.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'tcx> {
    pub id: ExprId,
    pub kind: ExprKind<'tcx>,
    pub ty: Ty<'tcx>,
    pub span: Option<Span>,
}

/// A monotonic source of fresh [`ExprId`]s, threaded through whichever pass
/// constructs MIR — monomorphization ([`crate::full::mono`]), textual re-intern
/// ([`mir::build`](self::build)), or the test-only builder. Centralizing id
/// assignment here is what makes "one expr, one stable anchor" hold regardless of
/// who built the tree; ids are *not* printed (they are regenerated
/// deterministically on parse), so they never enter the textual round-trip.
#[derive(Default)]
pub struct ExprIdGen {
    next: u32,
}

impl ExprIdGen {
    /// Hand out the next id. Post-order at the construction site (children are
    /// built before their parent), but the pass only relies on uniqueness.
    pub fn fresh(&mut self) -> ExprId {
        let id = ExprId(self.next);
        self.next += 1;
        id
    }
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
    /// A Unicode scalar value, stored as its 32-bit code point.
    ConstChar(u32),
    /// An integer literal, arbitrary-precision (arena-allocated to keep the
    /// node `Copy`); range-checked against its ground type at
    /// monomorphization and emitted at full width by codegen.
    ConstInt(&'tcx Integer),
    /// A floating-point literal, still the *exact* decimal value: codegen
    /// performs the single correctly-rounded conversion to the ground format.
    ConstFloat(&'tcx FloatLit),
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
    /// A `core::intrinsic::<family>::<fn>` call (see [`crate::intrinsic`]).
    Intrinsic {
        op: crate::intrinsic::IntrinsicOp,
        args: &'tcx [Expr<'tcx>],
    },
    Closure(ClosureExpr<'tcx>),
    ClosureCall {
        target: &'tcx Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    },
    /// A built-in operation on a statically shaped array, mirroring
    /// [`crate::semi::hir::ExprKind::ArrayOp`]: the resolved op, its value
    /// operands, and — for `Tabulate`/`Fold` — an inline kernel. The kernel is
    /// not a closure: its body references enclosing bindings directly
    /// (read-only borrows for the op's duration) and codegen inlines it into
    /// the element loop nest.
    ArrayOp {
        op: crate::intrinsic::ArrayFn,
        args: &'tcx [Expr<'tcx>],
        kernel: Option<&'tcx Kernel<'tcx>>,
    },
    Match(&'tcx Expr<'tcx>, DecisionTree<'tcx>),
    /// Error-recovery placeholder.
    Poison,
}

/// The inline kernel of an [`ArrayOp`](ExprKind::ArrayOp); see
/// [`crate::semi::hir::Kernel`]. The type checker guarantees the body is
/// rc-free (plain-typed throughout, except `get` reads of enclosing arrays).
#[derive(Clone, Copy, Debug)]
pub struct Kernel<'tcx> {
    pub params: &'tcx [(VarId, Ty<'tcx>)],
    pub body: &'tcx Expr<'tcx>,
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
        cases: &'tcx [(&'tcx Integer, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
    },
    Bool {
        if_true: &'tcx DecisionTree<'tcx>,
        if_false: &'tcx DecisionTree<'tcx>,
    },
    Char {
        cases: &'tcx [(u32, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
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
