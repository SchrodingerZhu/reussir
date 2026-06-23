//! The owned, context-free AST the textual-IR grammar produces.
//!
//! The lalrpop grammar builds this (plain `String`/`Vec`/`Box`, no interners or
//! arena), and a separate re-intern pass ([`crate::full::ir_build`]) rebuilds the
//! arena-allocated MIR from it. Decoupling keeps the grammar actions trivial and
//! the interning logic in ordinary, testable Rust.
//!
//! Type annotations are carried where the printer emits them (`5 : i32`,
//! `(a + b) : t`, `call(..) : t`); nodes the printer leaves unannotated (vars,
//! blocks) get a placeholder type at re-intern time, which is sound for the
//! text-faithful round trip (those positions are never re-printed).

/// A whole program: record instances, functions, exported trampolines.
#[derive(Clone, Debug)]
pub struct Program {
    pub records: Vec<String>,
    pub funcs: Vec<Func>,
    pub trampolines: Vec<Tramp>,
}

/// One top-level item, as the grammar yields them before partitioning.
#[derive(Clone, Debug)]
pub enum Item {
    Record(String),
    Func(Func),
    Tramp(Tramp),
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
                Item::Record(s) => p.records.push(s),
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

#[derive(Clone, Debug)]
pub enum Expr {
    ConstInt(i128, Ty),
    ConstFloat(f64, Ty),
    ConstBool(bool),
    /// An interned string literal, as its four raw `StringToken` words.
    GlobalStr([u64; 4]),
    Var(u32),
    Poison,
    Negate(Box<Expr>),
    Not(Box<Expr>),
    Arith(Box<Expr>, ArithOp, Box<Expr>, Ty),
    Cmp(Box<Expr>, CmpOp, Box<Expr>, Ty),
    Cast(Box<Expr>, Ty),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    RegionRun(Box<Expr>),
    Proj(Box<Expr>, Vec<u32>),
    Assign(Box<Expr>, u32, Box<Expr>),
    Let {
        var: u32,
        name: String,
        ty: Ty,
        value: Box<Expr>,
    },
    Seq(Vec<Expr>),
    Call {
        regional: bool,
        symbol: String,
        args: Vec<Expr>,
        ty: Ty,
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
    NullableCall(Option<Box<Expr>>),
    ClosureCall {
        target: Box<Expr>,
        args: Vec<Expr>,
    },
    Closure {
        captures: Vec<u32>,
        params: Vec<(u32, Ty)>,
        body: Box<Expr>,
    },
    Match(Box<Expr>, Box<Tree>),
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
