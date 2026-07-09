//! The owned, context-free AST the HIR grammar (`semi/hir/grammar.lalrpop`) produces.
//!
//! Mirrors [`crate::full::mir::raw`] (the MIR side) and reuses its operator/
//! capability enums, but the call forms carry a `#path` + type-argument list
//! (the HIR is pre-mono and item-keyed, not symbol-keyed) and types/functions
//! may mention generics (`$n`). A fully elaborated HIR has no inference
//! holes, so the textual form does not represent them.

pub use crate::full::mir::raw::{
    ArithOp, Cap, CmpOp, FileEntry, FloatLit, Integer, Span, StringEntry, char_scalar, float_lit,
    float_path_segs, small_u32, small_u64, small_usize, string_entries,
};

/// The whole HIR program: enough to resume into monomorphization — the record
/// declarations and trampoline roots mono needs, plus the elaborated functions.
#[derive(Clone, Debug)]
pub struct Program {
    pub files: Vec<FileEntry>,
    pub strings: Vec<StringEntry>,
    pub records: Vec<Record>,
    pub trampolines: Vec<Tramp>,
    pub funcs: Vec<Func>,
}

/// One top-level item, as the grammar yields them before partitioning.
#[derive(Clone, Debug)]
pub enum Item {
    File(FileEntry),
    String(StringEntry),
    Record(Record),
    Tramp(Tramp),
    Func(Func),
}

impl Program {
    pub fn from_items(items: Vec<Item>) -> Program {
        let mut p = Program {
            files: Vec::new(),
            strings: Vec::new(),
            records: Vec::new(),
            trampolines: Vec::new(),
            funcs: Vec::new(),
        };
        for item in items {
            match item {
                Item::File(f) => p.files.push(f),
                Item::String(s) => p.strings.push(s),
                Item::Record(r) => p.records.push(r),
                Item::Tramp(t) => p.trampolines.push(t),
                Item::Func(f) => p.funcs.push(f),
            }
        }
        p
    }
}

/// A record declaration carrying what mono reads: its path/kind/default cap, its
/// generic parameters (with the `regional` ones marked), and its field layout.
/// Field types are serialized (struct field *names* are not — mono does not read
/// them) so the resumed HIR resolves ground record layouts identically.
#[derive(Clone, Debug)]
pub struct Record {
    pub default_cap: DefaultCap,
    /// `#[repr(fixed)]`: uniform max-arm box sizing for an enum. Only ever set
    /// for enums (`struct` is always `false`).
    pub repr_fixed: bool,
    pub kind: RecordKind,
    pub path: String,
    pub generics: Vec<Generic>,
    /// The file-table id the record's span indexes (`in <id>`).
    pub file: Option<u32>,
    pub span: Option<Span>,
    pub body: RecordBody,
}

/// A record's field layout in the HIR form.
#[derive(Clone, Debug)]
pub enum RecordBody {
    /// A struct's ordered fields (names dropped; mono does not read them).
    Compound(Vec<Member>),
    /// An enum's variants, in declaration order.
    Variant(Vec<Variant>),
}

/// One compound field: an optional source name (absent for a tuple field), a
/// `[field]`-mutability marker, and its type.
#[derive(Clone, Debug)]
pub struct Member {
    pub name: Option<String>,
    pub is_field: bool,
    pub ty: Ty,
}

/// One enum variant: its source name and ordered field types.
#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Ty>,
}

#[derive(Clone, Copy, Debug)]
pub enum DefaultCap {
    Value,
    Shared,
    Regional,
}

#[derive(Clone, Copy, Debug)]
pub enum RecordKind {
    Struct,
    Enum,
}

/// An exported trampoline root: its C name/abi and the ground internal target.
#[derive(Clone, Debug)]
pub struct Tramp {
    pub abi: String,
    pub name: String,
    pub target: String,
    pub ty_args: Vec<Ty>,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub is_pub: bool,
    pub regional: bool,
    pub path: String,
    pub generics: Vec<Generic>,
    pub params: Vec<Param>,
    pub ret: Ty,
    /// The file-table id the function's spans index (`in <id>`).
    pub file: Option<u32>,
    pub span: Option<Span>,
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
    Char,
    Unit,
    Bottom,
    Generic(u32),
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

/// A typed HIR node — every node carries its [`Ty`] (see the MIR twin
/// [`crate::full::mir::raw::Expr`]); the re-intern pass reads it rather than
/// inventing one. `Let` is the lone structural exception (always `unit`).
#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: Box<Kind>,
    pub ty: Ty,
    /// The node's source span (`[start..end]`), byte offsets into the owning
    /// function's file.
    pub span: Option<Span>,
}

impl Expr {
    pub fn new(kind: Kind, ty: Ty) -> Expr {
        Expr {
            kind: Box::new(kind),
            ty,
            span: None,
        }
    }

    pub fn with_span(mut self, span: Option<Span>) -> Expr {
        self.span = span;
        self
    }
}

#[derive(Clone, Debug)]
pub enum Kind {
    ConstInt(Integer),
    ConstFloat(FloatLit),
    ConstBool(bool),
    /// A Unicode scalar value, stored as its 32-bit code point.
    ConstChar(u32),
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
        /// The binding name's own span (distinct from the node's).
        name_span: Option<Span>,
        value: Expr,
    },
    Seq(Vec<Expr>),
    FuncCall {
        regional: bool,
        path: String,
        ty_args: Vec<Ty>,
        args: Vec<Expr>,
    },
    CompoundCall {
        path: String,
        ty_args: Vec<Ty>,
        args: Vec<Expr>,
    },
    VariantCall {
        path: String,
        ty_args: Vec<Ty>,
        variant: usize,
        args: Vec<Expr>,
    },
    NullableCall(Option<Expr>),
    /// `intrinsic#<family>#<name>#<imm>(args…)` — a `core::intrinsic` call.
    Intrinsic {
        family: String,
        name: String,
        imm: u32,
        args: Vec<Expr>,
    },
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
        cases: Vec<(Integer, Tree)>,
        default: Box<Tree>,
    },
    Bool {
        if_true: Box<Tree>,
        if_false: Box<Tree>,
    },
    Char {
        cases: Vec<(u32, Tree)>,
        default: Box<Tree>,
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

/// A switch-arm label (see [`crate::full::mir::raw::Label`]).
#[derive(Clone, Debug)]
pub enum Label {
    Int(Integer),
    Ctor(usize),
    Str([u64; 4]),
    Bool(bool),
    Char(u32),
    NonNull,
    Null,
    Wildcard,
}

/// Reassemble a `switch`'s arms into the typed [`Cases`] (see the MIR twin,
/// [`crate::full::mir::raw::build_switch`]).
pub fn build_switch(scrutinee: Path, arms: Vec<(Label, Tree)>) -> Tree {
    let kind = arms
        .iter()
        .map(|(l, _)| l)
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
        Some(Label::Char(_)) => {
            let (mut cases, mut default) = (Vec::new(), None);
            for (l, t) in arms {
                match l {
                    Label::Char(c) => cases.push((c, t)),
                    Label::Wildcard => default = Some(Box::new(t)),
                    _ => {}
                }
            }
            Cases::Char {
                cases,
                default: default.unwrap(),
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
