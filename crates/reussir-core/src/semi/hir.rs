//! The Semi HIR: a typed, still-polymorphic intermediate representation.
//!
//! Every [`Expr`] carries its [`Ty`], which may contain generics or holes.
//! Pattern matches are compiled to a [`DecisionTree`].

use reussir_syntax::kind::TokenKey;
use reussir_syntax::source::FileId;

use crate::literal::{FloatLit, Integer};
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

/// The qualified path of a bound's trait in the textual HIR/.rri, as owned
/// segments — structural, so consumers resolve by segment rather than
/// re-splitting a joined string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundPath {
    pub segments: Vec<String>,
}

impl BoundPath {
    /// The trait's own name — the last segment.
    pub fn basename(&self) -> &str {
        self.segments.last().expect("a bound path is never empty")
    }
}

impl std::fmt::Display for BoundPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.segments.join("::"))
    }
}

/// One bound on a generic binder (`$0 (T): Num + Sync`) as it round-trips
/// through the textual HIR and `.rri`. An enum so future bound kinds
/// (capability bounds, …) extend it structurally; the only kind today is a
/// trait named by qualified path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bound {
    Trait(BoundPath),
}

impl Bound {
    /// The trait path for resolution; every current bound kind carries one.
    pub fn trait_path(&self) -> &BoundPath {
        match self {
            Bound::Trait(path) => path,
        }
    }
}

impl std::fmt::Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bound::Trait(path) => path.fmt(f),
        }
    }
}

/// A local variable, unique within a function body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarId(pub u32);

/// A unique id for an expression node (used by later phases for keying).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(pub u32);

/// Arithmetic / boolean / bitwise binary operators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
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
    /// A Unicode scalar value, stored as its 32-bit code point.
    ConstChar(u32),
    /// An integer literal (its type is a `Num`-bounded hole until solved).
    /// Arbitrary-precision: range-checked against its ground type at
    /// monomorphization, never truncated before.
    ConstInt(&'tcx Integer),
    /// A floating-point literal, exact ([`crate::literal::FloatLit`]): rounded
    /// to its ground format only in codegen.
    ConstFloat(&'tcx FloatLit),
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
    /// A trait-method call whose receiver is a bare in-scope generic, so the
    /// impl is unknowable until monomorphization grounds `Self` and selects
    /// it. A call with any *ground-headed* receiver is resolved at check
    /// time to the unique impl's method as an ordinary [`ExprKind::FuncCall`] —
    /// coherence guarantees the choice — so this variant never reaches
    /// codegen; monomorphization rewrites it.
    TraitCall {
        /// The trait's path-keyed identity (stable across serialization,
        /// unlike the session-dense `TraitId`).
        trait_def: DefId,
        /// Index into the trait's method table (= the impl's method order).
        method: u32,
        /// `[Self] ++ trait args ++ the method's own generics`.
        ty_args: Vec<Ty<'tcx>>,
        /// The receiver first, then the remaining arguments.
        args: Vec<Expr<'tcx>>,
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
    /// A `core::intrinsic::<family>::<fn>` call: the resolved op (family + its
    /// immediates) and the value operands (the node's own type is the result).
    Intrinsic {
        op: crate::intrinsic::IntrinsicOp,
        args: Vec<Expr<'tcx>>,
    },
    Closure(ClosureExpr<'tcx>),
    ClosureCall {
        target: Box<Expr<'tcx>>,
        args: Vec<Expr<'tcx>>,
    },
    /// A built-in operation on a statically shaped array (issue #344): the
    /// resolved op and its value operands (see [`crate::intrinsic::ArrayFn`]
    /// for each op's operand list). `Tabulate`/`Fold` take their kernel as an
    /// ordinary closure-typed operand (the last one); the optimizer inlines a
    /// literal lambda into the loop nest (loop-carried beta reduction).
    ArrayOp {
        op: crate::intrinsic::ArrayFn,
        args: Vec<Expr<'tcx>>,
    },
    /// A built-in operation on a transient tensor
    /// (docs/design/tensor-kernels.md): the resolved op and its value
    /// operands (see [`crate::intrinsic::TensorFn`]).
    TensorOp {
        op: crate::intrinsic::TensorFn,
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
        cases: Vec<(&'tcx Integer, DecisionTree<'tcx>)>,
        default: Box<DecisionTree<'tcx>>,
    },
    Bool {
        if_true: Box<DecisionTree<'tcx>>,
        if_false: Box<DecisionTree<'tcx>>,
    },
    Char {
        cases: Vec<(u32, DecisionTree<'tcx>)>,
        default: Box<DecisionTree<'tcx>>,
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

/// A trait item in print-ready form: paths and names pre-rendered. Both the
/// elaborator (from its `TraitDb`, via [`trait_texts`]) and the HIR-text
/// parser produce this shape, so one printer serves emission and the
/// parse-side round trip; the extern reload resolves it back into a session
/// `TraitDb`.
///
/// [`trait_texts`]: crate::semi::hir::trait_texts
#[derive(Clone, Debug)]
pub struct TraitText<'tcx> {
    /// The trait's def — identity for interface filtering and reload; not
    /// itself printed (the path is).
    pub def: DefId,
    pub is_pub: bool,
    /// `#[sealed]`: no user impls; serialized so a consumer session
    /// enforces the same rule.
    pub sealed: bool,
    pub path: String,
    /// Binders; `[0]` is the implicit `Self`. Names are display-only.
    pub generics: Vec<(GenericId, Option<String>)>,
    /// Super-references: rendered path + full args (`args[0]` = `Self`).
    pub supers: Vec<(String, Vec<Ty<'tcx>>)>,
    pub methods: Vec<TraitMethodText<'tcx>>,
    pub file: FileId,
    pub span: Option<Span>,
}

/// A trait method signature in print-ready form (receiver at `params[0]`).
#[derive(Clone, Debug)]
pub struct TraitMethodText<'tcx> {
    pub regional: bool,
    pub name: String,
    pub generics: Vec<(GenericId, Option<String>)>,
    /// `None` is an associated function (no receiver).
    pub receiver: Option<crate::semi::traits::def::ReceiverForm>,
    pub params: Vec<Ty<'tcx>>,
    pub ret: Ty<'tcx>,
    pub span: Option<Span>,
}

/// An impl item in print-ready form: trait path + full args
/// (`args[0]` = the self type) and method paths in trait-method order.
#[derive(Clone, Debug)]
pub struct ImplText<'tcx> {
    /// The implemented trait's def (identity only, like [`TraitText::def`]).
    pub trait_def: DefId,
    pub generics: Vec<(GenericId, Option<String>)>,
    pub trait_path: String,
    pub args: Vec<Ty<'tcx>>,
    pub methods: Vec<String>,
    pub file: FileId,
    pub span: Option<Span>,
}

/// Render the user-declared traits and impls of `db` print-ready. Sealed
/// traits (the builtins) and their impls are skipped: they are re-registered
/// from the compiler itself in every session, never shipped.
pub fn trait_texts<'tcx>(
    db: &crate::semi::traits::TraitDb<'tcx>,
    defs: &crate::semi::resolve::DefTable,
    resolver: &dyn reussir_syntax::kind::Resolver<TokenKey>,
) -> (Vec<TraitText<'tcx>>, Vec<ImplText<'tcx>>) {
    let path_of = |def: DefId| {
        defs.path(def)
            .0
            .iter()
            .map(|k| resolver.resolve(*k))
            .collect::<Vec<_>>()
            .join("::")
    };
    let mut traits = Vec::new();
    for id in (0..db.traits_len() as u32).map(crate::semi::traits::TraitId) {
        let t = db.trait_def(id);
        // Compiler-provided declarations (the construction-time builtins,
        // the site-less comparison fallback, and the scalar impls below)
        // never serialize: every session re-registers its own. A *declared*
        // sealed trait — `core`'s numeric tower — ships, `#[sealed]` and
        // all.
        if t.compiler_provided() {
            continue;
        }
        let mut generics = vec![(t.self_param, Some("Self".to_string()))];
        generics.extend(
            t.params
                .iter()
                .map(|&(name, g)| (g, Some(resolver.resolve(name).to_string()))),
        );
        traits.push(TraitText {
            def: t.def,
            is_pub: t.visibility == crate::surface::Visibility::Public,
            sealed: t.sealed,
            path: path_of(t.def),
            generics,
            supers: t
                .supertraits
                .iter()
                .map(|s| (path_of(db.trait_def(s.trait_id).def), s.args.clone()))
                .collect(),
            methods: t
                .methods
                .iter()
                .map(|m| TraitMethodText {
                    regional: m.is_regional,
                    name: resolver.resolve(m.name).to_string(),
                    generics: m
                        .generics
                        .iter()
                        .map(|&(name, g)| (g, Some(resolver.resolve(name).to_string())))
                        .collect(),
                    receiver: m.receiver,
                    params: m.params.clone(),
                    ret: m.ret,
                    span: m.span,
                })
                .collect(),
            file: t.file,
            span: t.span,
        });
    }
    let mut impls = Vec::new();
    for imp in db.impls() {
        // Compiler-provided impls (the no-`core` fallback's) never
        // serialize; source-declared builtin impls — including those of the
        // sealed numeric tower — ship like any other.
        if imp.compiler_provided() {
            continue;
        }
        impls.push(ImplText {
            trait_def: db.trait_def(imp.trait_ref.trait_id).def,
            generics: imp.generics.iter().map(|&g| (g, None)).collect(),
            trait_path: path_of(db.trait_def(imp.trait_ref.trait_id).def),
            args: imp.trait_ref.args.clone(),
            methods: imp.methods.iter().map(|&d| path_of(d)).collect(),
            file: imp.file,
            span: imp.span,
        });
    }
    (traits, impls)
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
    /// The file the function is declared in; its spans (and its body's) index
    /// that file in the compilation's source cache.
    pub file: FileId,
}
