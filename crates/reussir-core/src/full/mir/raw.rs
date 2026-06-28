//! The owned, context-free AST the textual-IR grammar produces.
//!
//! The lalrpop grammar builds this (plain `String`/`Vec`/`Box`, no interners or
//! arena), and a separate re-intern pass ([`crate::full::mir::build`]) rebuilds the
//! arena-allocated MIR from it. Decoupling keeps the grammar actions trivial and
//! the interning logic in ordinary, testable Rust.
//!
//! Every node is `{ kind, ty }`: the grammar reads an explicit `: ty` for every
//! value node, so the re-intern pass takes each type straight from the text and
//! never invents one. The two structural exceptions — `let` (always `unit`) and
//! a `Seq` (its last item's type) — are filled by the grammar actions and are
//! correct by construction. This makes the round trip value-sound, not merely
//! text-faithful.

/// A whole program: record instances (with their ground layout), functions,
/// exported trampolines.
#[derive(Clone, Debug)]
pub struct Program {
    pub records: Vec<RecordDecl>,
    pub funcs: Vec<Func>,
    pub trampolines: Vec<Tramp>,
}

/// One top-level item, as the grammar yields them before partitioning.
#[derive(Clone, Debug)]
pub enum Item {
    Record(RecordDecl),
    Func(Func),
    Tramp(Tramp),
}

/// A ground record declaration: its v0 symbol, the capability it declares by
/// default, the nominal record type (path + ground args), and its field layout.
/// The layout is what makes the textual MIR self-contained — a parsed program
/// carries enough to lower records without re-consulting the elaborator.
#[derive(Clone, Debug)]
pub struct RecordDecl {
    pub symbol: String,
    pub default_cap: DefCap,
    pub ty: Ty,
    pub body: RecordBody,
}

/// A record's ground shape.
#[derive(Clone, Debug)]
pub enum RecordBody {
    /// A struct: ordered fields.
    Compound(Vec<Member>),
    /// An enum: one compound sub-record per variant, in declaration order.
    Variant(Vec<Variant>),
}

/// One compound field: its ground type, and whether it is a mutable `[field]`
/// link (only regional records carry these).
#[derive(Clone, Debug)]
pub struct Member {
    pub is_field: bool,
    pub ty: Ty,
}

/// One enum variant: its source name and ordered field types.
#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Ty>,
}

/// The capability a record declares by default (mirrors
/// [`crate::semi::ctxt::DefaultCap`]).
#[derive(Clone, Copy, Debug)]
pub enum DefCap {
    Value,
    Shared,
    Regional,
}

impl Program {
    /// Partition a flat item list into a [`Program`].
    pub fn from_items(items: Vec<Item>) -> Program {
        let mut p = Program {
            records: Vec::new(),
            funcs: Vec::new(),
            trampolines: Vec::new(),
        };
        for item in items {
            match item {
                Item::Record(r) => p.records.push(r),
                Item::Func(f) => p.funcs.push(f),
                Item::Tramp(t) => p.trampolines.push(t),
            }
        }
        p
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub is_pub: bool,
    pub regional: bool,
    pub symbol: String,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub var: u32,
    pub name: String,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct Tramp {
    pub abi: String,
    pub export: String,
    pub target: String,
}

#[derive(Clone, Debug)]
pub enum Ty {
    Signed(u16),
    Unsigned(u16),
    Ieee(u16),
    BFloat16,
    Float8,
    Bool,
    Str,
    Unit,
    Bottom,
    Nullable(Box<Ty>),
    /// `[cap] path<args>` — a regional/value record type.
    Record {
        cap: Cap,
        path: String,
        args: Vec<Ty>,
    },
    Closure {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },
}

/// The per-use capability prefix printed before a record type (absent = value).
#[derive(Clone, Copy, Debug)]
pub enum Cap {
    None,
    Flex,
    Rigid,
    Regional,
}

/// The four raw words of an interned `StringToken`.
pub type StrTag = [u64; 4];

/// A typed expression node: every node carries its [`Ty`], so the re-intern pass
/// never invents a type — the printed `kind : ty` annotation is read back
/// verbatim, making the text round trip a soundness proof. The two exceptions
/// (`Let`/`Assign` are `unit`, `Seq` is its last item's type) are filled by the
/// grammar actions, not printed, and are correct by construction.
#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: Box<Kind>,
    pub ty: Ty,
}

impl Expr {
    pub fn new(kind: Kind, ty: Ty) -> Expr {
        Expr {
            kind: Box::new(kind),
            ty,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Kind {
    ConstInt(i128),
    ConstFloat(f64),
    ConstBool(bool),
    /// An interned string literal, as its four raw `StringToken` words.
    GlobalStr([u64; 4]),
    Var(u32),
    Poison,
    Negate(Expr),
    Not(Expr),
    Arith(Expr, ArithOp, Expr),
    Cmp(Expr, CmpOp, Expr),
    Cast(Expr, Ty),
    If(Expr, Expr, Expr),
    RegionRun(Expr),
    Proj(Expr, Vec<u32>),
    Assign(Expr, u32, Expr),
    Let {
        var: u32,
        name: String,
        value: Expr,
    },
    Seq(Vec<Expr>),
    Call {
        regional: bool,
        symbol: String,
        args: Vec<Expr>,
    },
    Ctor {
        symbol: String,
        args: Vec<Expr>,
    },
    Variant {
        symbol: String,
        variant: usize,
        args: Vec<Expr>,
    },
    NullableCall(Option<Expr>),
    ClosureCall {
        target: Expr,
        args: Vec<Expr>,
    },
    Closure {
        captures: Vec<(u32, Ty)>,
        params: Vec<(u32, Ty)>,
        body: Expr,
    },
    Match(Expr, Box<Tree>),
}

#[derive(Clone, Copy, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
}

#[derive(Clone, Copy, Debug)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

/// A scrutinee path (`scrut.0.1`) — the field indices after `scrut`.
pub type Path = Vec<u32>;

/// A pattern binding: a local variable bound to a scrutinee path.
pub type Binding = (u32, Path);

#[derive(Clone, Debug)]
pub enum Tree {
    Uncovered,
    Unreachable,
    Leaf {
        bindings: Vec<Binding>,
        body: Expr,
    },
    Guard {
        bindings: Vec<Binding>,
        guard: Expr,
        success: Box<Tree>,
        failure: Box<Tree>,
    },
    Switch {
        scrutinee: Path,
        cases: Cases,
    },
}

#[derive(Clone, Debug)]
pub enum Cases {
    Int {
        cases: Vec<(i128, Tree)>,
        default: Box<Tree>,
    },
    Bool {
        if_true: Box<Tree>,
        if_false: Box<Tree>,
    },
    Ctor(Vec<Tree>),
    Str {
        cases: Vec<([u64; 4], Tree)>,
        default: Box<Tree>,
    },
    Nullable {
        non_null: Box<Tree>,
        null: Box<Tree>,
    },
}

/// A switch-arm label, as the grammar reads it before partitioning into a typed
/// [`Cases`]. The printer emits homogeneous labels per switch (all `#i`, all int,
/// …), with a `_` wildcard standing for the default arm.
#[derive(Clone, Copy, Debug)]
pub enum Label {
    Int(i128),
    Ctor(usize),
    Str([u64; 4]),
    Bool(bool),
    NonNull,
    Null,
    Wildcard,
}

/// Reassemble a `switch`'s arms into the typed [`Cases`], keyed off the first
/// non-wildcard label. (`unwrap`s assume printer-shaped input, as the textual IR
/// is machine-emitted.)
pub fn build_switch(scrutinee: Path, arms: Vec<(Label, Tree)>) -> Tree {
    let kind = arms
        .iter()
        .map(|(l, _)| *l)
        .find(|l| !matches!(l, Label::Wildcard));
    let cases = match kind {
        Some(Label::Bool(_)) => {
            let (mut if_true, mut if_false) = (None, None);
            for (l, t) in arms {
                match l {
                    Label::Bool(true) => if_true = Some(Box::new(t)),
                    Label::Bool(false) => if_false = Some(Box::new(t)),
                    _ => {}
                }
            }
            Cases::Bool {
                if_true: if_true.unwrap(),
                if_false: if_false.unwrap(),
            }
        }
        Some(Label::Ctor(_)) => {
            let mut v: Vec<(usize, Tree)> = arms
                .into_iter()
                .filter_map(|(l, t)| match l {
                    Label::Ctor(i) => Some((i, t)),
                    _ => None,
                })
                .collect();
            v.sort_by_key(|(i, _)| *i);
            Cases::Ctor(v.into_iter().map(|(_, t)| t).collect())
        }
        Some(Label::NonNull | Label::Null) => {
            let (mut non_null, mut null) = (None, None);
            for (l, t) in arms {
                match l {
                    Label::NonNull => non_null = Some(Box::new(t)),
                    Label::Null => null = Some(Box::new(t)),
                    _ => {}
                }
            }
            Cases::Nullable {
                non_null: non_null.unwrap(),
                null: null.unwrap(),
            }
        }
        Some(Label::Str(_)) => {
            let (mut cases, mut default) = (Vec::new(), None);
            for (l, t) in arms {
                match l {
                    Label::Str(s) => cases.push((s, t)),
                    Label::Wildcard => default = Some(Box::new(t)),
                    _ => {}
                }
            }
            Cases::Str {
                cases,
                default: default.unwrap(),
            }
        }
        // Int, or a lone wildcard.
        _ => {
            let (mut cases, mut default) = (Vec::new(), None);
            for (l, t) in arms {
                match l {
                    Label::Int(n) => cases.push((n, t)),
                    Label::Wildcard => default = Some(Box::new(t)),
                    _ => {}
                }
            }
            Cases::Int {
                cases,
                default: default.unwrap(),
            }
        }
    };
    Tree::Switch { scrutinee, cases }
}
