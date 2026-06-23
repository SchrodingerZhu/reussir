//! The owned, context-free AST the HIR grammar (`semi/hir_ir.lalrpop`) produces.
//!
//! Mirrors [`crate::full::ir_raw`] (the MIR side) and reuses its operator/
//! capability enums, but the call forms carry a `#path` + type-argument list
//! (the HIR is pre-mono and item-keyed, not symbol-keyed) and types/functions
//! may mention generics (`$n`) and holes (`?n`).

pub use crate::full::ir_raw::{ArithOp, Cap, CmpOp};

/// The whole HIR program: the elaborated functions.
#[derive(Clone, Debug)]
pub struct Program {
    pub funcs: Vec<Func>,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub is_pub: bool,
    pub regional: bool,
    pub path: String,
    pub generics: Vec<Generic>,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Option<Expr>,
}

/// A generic parameter binder: its id and whether it sits at a `[flex]` position
/// (and so must be instantiated regionally).
#[derive(Clone, Copy, Debug)]
pub struct Generic {
    pub id: u32,
    pub regional: bool,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub var: u32,
    pub name: String,
    pub ty: Ty,
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
    Generic(u32),
    Hole(u32),
    Nullable(Box<Ty>),
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
    FuncCall {
        regional: bool,
        path: String,
        ty_args: Vec<Ty>,
        args: Vec<Expr>,
        ty: Ty,
    },
    CompoundCall {
        path: String,
        ty_args: Vec<Ty>,
        args: Vec<Expr>,
        ty: Ty,
    },
    VariantCall {
        path: String,
        ty_args: Vec<Ty>,
        variant: usize,
        args: Vec<Expr>,
        ty: Ty,
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

/// A scrutinee path (the field indices after `scrut`).
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

/// A switch-arm label (see [`crate::full::ir_raw::Label`]).
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

/// Reassemble a `switch`'s arms into the typed [`Cases`] (see the MIR twin,
/// [`crate::full::ir_raw::build_switch`]).
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
