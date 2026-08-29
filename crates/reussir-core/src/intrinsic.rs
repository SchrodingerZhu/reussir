//! The built-in `core::intrinsic` operation families.
//!
//! Reussir source reaches these as `core::intrinsic::<family>::<name>(…)` — the
//! package name `core` is reserved, so the path cannot collide with user code.
//! [`IntrinsicOp`] is the resolved family-independent representation carried by
//! HIR and MIR; each family defines its own names, checking, ownership contract,
//! and code generation.

/// The shape of a math intrinsic's value arguments (the fast-math flag is an
/// extra trailing argument in the surface form, not counted here).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MathKind {
    /// One float in, same float out (`sqrt`, `sin`, …).
    Unary,
    /// One float in, `bool` out (`isnan`, …).
    Check,
    /// Two floats in, same float out (`powf`, `atan2`, `copysign`).
    Binary,
    /// Fused multiply-add: three floats in, same float out.
    Fma,
    /// Float base and `i32` exponent in, float out.
    Fpowi,
    /// One integer in, same integer out (`ctpop`, `ctlz`, `cttz`). Bounded
    /// by `Integral` rather than `FloatingPoint`, and takes no fast-math
    /// flag — the MLIR integer math ops have no such attribute.
    IntUnary,
}

macro_rules! math_fns {
    ($($variant:ident => $name:literal : $kind:ident),* $(,)?) => {
        /// A surfaced math intrinsic; [`Self::as_str`] is both the surface
        /// name and the MLIR `math.<name>` mnemonic.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum MathFn {
            $($variant),*
        }

        impl MathFn {
            /// Parse a surface / textual-IR name.
            pub fn parse(name: &str) -> Option<MathFn> {
                match name {
                    $($name => Some(MathFn::$variant),)*
                    _ => None,
                }
            }

            /// The surface name, also the `math.<name>` op mnemonic.
            pub fn as_str(self) -> &'static str {
                match self {
                    $(MathFn::$variant => $name),*
                }
            }

            pub fn kind(self) -> MathKind {
                match self {
                    $(MathFn::$variant => MathKind::$kind),*
                }
            }

            /// Every surfaced math intrinsic, for coverage checks.
            pub const ALL: &'static [MathFn] = &[$(MathFn::$variant),*];
        }
    };
}

math_fns! {
    Absf => "absf": Unary,
    Acos => "acos": Unary,
    Acosh => "acosh": Unary,
    Asin => "asin": Unary,
    Asinh => "asinh": Unary,
    Atan => "atan": Unary,
    Atanh => "atanh": Unary,
    Cbrt => "cbrt": Unary,
    Ceil => "ceil": Unary,
    Cos => "cos": Unary,
    Cosh => "cosh": Unary,
    Erf => "erf": Unary,
    Erfc => "erfc": Unary,
    Exp => "exp": Unary,
    Exp2 => "exp2": Unary,
    Expm1 => "expm1": Unary,
    Floor => "floor": Unary,
    Log10 => "log10": Unary,
    Log1p => "log1p": Unary,
    Log2 => "log2": Unary,
    Round => "round": Unary,
    Roundeven => "roundeven": Unary,
    Rsqrt => "rsqrt": Unary,
    Sin => "sin": Unary,
    Sinh => "sinh": Unary,
    Sqrt => "sqrt": Unary,
    Tan => "tan": Unary,
    Tanh => "tanh": Unary,
    Trunc => "trunc": Unary,
    Isfinite => "isfinite": Check,
    Isinf => "isinf": Check,
    Isnan => "isnan": Check,
    Isnormal => "isnormal": Check,
    Atan2 => "atan2": Binary,
    Copysign => "copysign": Binary,
    Powf => "powf": Binary,
    Fma => "fma": Fma,
    Fpowi => "fpowi": Fpowi,
    Ctpop => "ctpop": IntUnary,
    Ctlz => "ctlz": IntUnary,
    Cttz => "cttz": IntUnary,
}

impl MathKind {
    /// How many *value* arguments the intrinsic takes (the constant fast-math
    /// flag is one more).
    pub fn value_args(self) -> usize {
        match self {
            MathKind::Unary | MathKind::Check | MathKind::IntUnary => 1,
            MathKind::Binary | MathKind::Fpowi => 2,
            MathKind::Fma => 3,
        }
    }
}

/// A fast-math flag set, the constant trailing `i32` argument of every math
/// intrinsic. Bit-compatible with MLIR's `arith.fastmath` bits; `0` is none
/// and `127` (all bits) is `fast`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FastMath(pub u32);

impl FastMath {
    pub const MAX: u32 = 127;

    const NAMES: [(u32, &'static str); 7] = [
        (1, "reassoc"),
        (2, "nnan"),
        (4, "ninf"),
        (8, "nsz"),
        (16, "arcp"),
        (32, "contract"),
        (64, "afn"),
    ];

    /// The MLIR `#arith.fastmath<…>` attribute text, or `None` for an empty
    /// flag set (MLIR's default — no attribute needed).
    pub fn mlir_attr(self) -> Option<String> {
        if self.0 == 0 {
            return None;
        }
        if self.0 == Self::MAX {
            return Some("#arith.fastmath<fast>".to_string());
        }
        let names: Vec<&str> = Self::NAMES
            .iter()
            .filter(|(bit, _)| self.0 & bit != 0)
            .map(|&(_, name)| name)
            .collect();
        Some(format!("#arith.fastmath<{}>", names.join(",")))
    }
}

/// A built-in operation on a statically shaped array (`TyKind::Array`, issue
/// #344), spelled `core::intrinsic::array::<fn>`. Unlike [`IntrinsicOp`],
/// these carry their own HIR/MIR node (`ExprKind::ArrayOp`) because
/// `Tabulate`/`Fold` hold an inline kernel body, which the generic
/// `Intrinsic` node cannot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArrayFn {
    /// `splat<[T; e…]>(v)`: every element is `v`. Value operands: `[v]`.
    Splat,
    /// `tabulate<[T; e…]>(|i, …| e)`: element at each index tuple is the
    /// kernel applied to the (row-major) indices. No value operands.
    Tabulate,
    /// `get(a, i, …)`: checked element read. Value operands: `[a, i…]`; `a`
    /// is borrowed, not consumed.
    Get,
    /// `set(a, i, …, v)`: checked element write, consuming `a` and returning
    /// the updated array (copy-on-write when shared). Value operands:
    /// `[a, i…, v]`.
    Set,
    /// `fold(a, init, |acc, x| e)`: row-major reduction. Value operands:
    /// `[a, init]`; `a` is borrowed, not consumed.
    Fold,
    /// `dim(a, k)`: the `k`-th extent as `u64`. Value operands: `[a, k]`;
    /// `a` is borrowed, not consumed. The interesting case is a dynamic
    /// dimension (`[T; ?, 4]`), whose extent lives in the box header; on a
    /// fully static array it folds to a constant.
    Dim,
}

/// A built-in operation on a shared mutable [`Cell`](crate::semi::ty::TyKind::Cell),
/// spelled `core::intrinsic::cell::<fn>` and dispatched on the operand's
/// cell kind. `alloc`/`get`/`set` work on every kind; `rmw` on every kind
/// but a plain `Cell` (the exclusive in-use region, an atomic CAS-retry, or
/// a lock's critical section); `in_use` observes the exclusive guard flag;
/// `rdlock` runs a read transaction over an `RwLock` cell. The operation
/// itself does not record the kind — operand and result types carry it, the
/// checker enforces the matrix, and the dialect verifier re-checks it after
/// lowering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CellFn {
    /// `alloc(value)`: move `value` into a fresh cell.
    Alloc,
    /// `get(cell)`: clone the value currently stored in `cell`.
    Get,
    /// `set(cell, value)`: replace and drop the old value.
    Set,
    /// `rmw(cell, update)`: move the value through `update` and store its
    /// result. Every kind but a plain `Cell` and an `Atomic` cell (the
    /// atomic CAS-retry region must be effect-free, which a closure call is
    /// not; dedicated atomic arithmetic intrinsics are future work).
    Rmw,
    /// `in_use(cell)`: whether the element is currently moved out into an
    /// active `rmw` updater. Exclusive cells (`RefCell`) only.
    InUse,
    /// `rdlock(cell, reader)`: run `reader` over a clone of the element
    /// inside the shared read lock. `RwLock` cells only.
    Rdlock,
}

impl CellFn {
    /// Parse a surface / textual-IR name.
    pub fn parse(name: &str) -> Option<CellFn> {
        match name {
            "alloc" => Some(CellFn::Alloc),
            "get" => Some(CellFn::Get),
            "set" => Some(CellFn::Set),
            "rmw" => Some(CellFn::Rmw),
            "in_use" => Some(CellFn::InUse),
            "rdlock" => Some(CellFn::Rdlock),
            _ => None,
        }
    }

    /// The surface name, also used by the textual IR.
    pub fn as_str(self) -> &'static str {
        match self {
            CellFn::Alloc => "alloc",
            CellFn::Get => "get",
            CellFn::Set => "set",
            CellFn::Rmw => "rmw",
            CellFn::InUse => "in_use",
            CellFn::Rdlock => "rdlock",
        }
    }

    /// Whether this operation is available on a cell of `kind` — the §4.2
    /// access matrix. Lowering re-derives the kind from the monomorphized
    /// operand type; the dialect verifier re-checks after lowering.
    pub fn supported_on(self, kind: crate::semi::ty::CellKind) -> bool {
        use crate::semi::ty::CellKind;
        match self {
            CellFn::Alloc | CellFn::Get | CellFn::Set => true,
            CellFn::Rmw => kind != CellKind::Plain && kind != CellKind::Atomic,
            CellFn::InUse => kind == CellKind::Exclusive,
            CellFn::Rdlock => kind == CellKind::Rwlock,
        }
    }

    /// Which operands support this operation, for diagnostics.
    pub fn requirement(self) -> &'static str {
        match self {
            CellFn::Alloc | CellFn::Get | CellFn::Set => "any cell",
            CellFn::Rmw => {
                "an exclusive or lock-guarded cell (`RefCell`, `Mutex`, \
                 `FlatLock`, or `RwLock`)"
            }
            CellFn::InUse => "an exclusive cell (`RefCell`)",
            CellFn::Rdlock => "an `RwLock` cell",
        }
    }
}

impl ArrayFn {
    /// Parse a surface / textual-IR name.
    pub fn parse(name: &str) -> Option<ArrayFn> {
        match name {
            "splat" => Some(ArrayFn::Splat),
            "tabulate" => Some(ArrayFn::Tabulate),
            "get" => Some(ArrayFn::Get),
            "set" => Some(ArrayFn::Set),
            "fold" => Some(ArrayFn::Fold),
            "dim" => Some(ArrayFn::Dim),
            _ => None,
        }
    }

    /// The surface name, also used by the textual IR (`array#<name>`).
    pub fn as_str(self) -> &'static str {
        match self {
            ArrayFn::Splat => "splat",
            ArrayFn::Tabulate => "tabulate",
            ArrayFn::Get => "get",
            ArrayFn::Set => "set",
            ArrayFn::Fold => "fold",
            ArrayFn::Dim => "dim",
        }
    }

    /// Whether the op takes an inline kernel body.
    pub fn has_kernel(self) -> bool {
        matches!(self, ArrayFn::Tabulate | ArrayFn::Fold)
    }
}

/// A built-in operation on a transient tensor (`TyKind::Tensor`), spelled
/// `core::intrinsic::tensor::<fn>`. Like [`ArrayFn`], these carry their own
/// HIR/MIR node (`ExprKind::TensorOp`). The boundary contract
/// (docs/design/tensor-kernels.md): `of` roots a reference to the backing
/// array for the tensor's lifetime, `materialize` is the only way a
/// tensor's data escapes back into the rc world.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorFn {
    /// `of(a)`: view an array as a tensor. Value operands: `[a]`; consumes
    /// the reference (the tensor owns the rooted dup).
    Of,
    /// `materialize(t)`: build a fresh array from the tensor's value.
    /// Value operands: `[t]`.
    Materialize,
    /// `dim(t, k)`: the `k`-th extent as `u64`. Value operands: `[t, k]`.
    Dim,
}

impl TensorFn {
    /// Parse a surface / textual-IR name.
    pub fn parse(name: &str) -> Option<TensorFn> {
        match name {
            "of" => Some(TensorFn::Of),
            "materialize" => Some(TensorFn::Materialize),
            "dim" => Some(TensorFn::Dim),
            _ => None,
        }
    }

    /// The surface name, also used by the textual IR (`tensor#<name>`).
    pub fn as_str(self) -> &'static str {
        match self {
            TensorFn::Of => "of",
            TensorFn::Materialize => "materialize",
            TensorFn::Dim => "dim",
        }
    }
}

/// A resolved intrinsic operation: which family, which op within it, and the
/// family's compile-time immediates. One generic `Intrinsic` IR node carries
/// this through HIR/MIR, so the node itself — and its traversal, ownership, and
/// monomorphization — stay family-agnostic. Adding a family (`arith`, `atomic`,
/// …) adds a variant here plus its own checking and codegen; the shared IR
/// plumbing is untouched.
///
/// The textual IR spells an op as `intrinsic#<family>#<name>#<imm>(args…)`,
/// where `<imm>` is the family's packed immediate ([`Self::imm`]); [`Self::parse`]
/// rebuilds the typed op from that triple.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntrinsicOp {
    /// `core::intrinsic::panic::panic()`: aborts execution and inhabits the
    /// contextually expected result type.
    Panic,
    /// `core::intrinsic::math::<func>`, with an `arith.fastmath` flag.
    Math { func: MathFn, flag: u32 },
    /// `core::intrinsic::cell::<func>`; cell operations have no immediates
    /// and dispatch on the operand's cell flavor rather than encoding it.
    Cell { func: CellFn },
    /// `core::intrinsic::str::<func>`; string operations have no immediates.
    Str { func: StrFn },
}

/// A surfaced `str` intrinsic; [`Self::as_str`] is both the surface name
/// and the `reussir.str.<name>` mnemonic family it lowers through.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StrFn {
    /// `len(s)`: the byte length.
    Len,
    /// `byte_at(s, index)`: the byte at `index`, `0` when out of bounds.
    ByteAt,
    /// `slice(s, offset)`: the suffix from byte `offset`, clamped to empty
    /// when the offset passes the end.
    Slice,
}

impl StrFn {
    /// Parse a surface / textual-IR name.
    pub fn parse(name: &str) -> Option<StrFn> {
        match name {
            "len" => Some(StrFn::Len),
            "byte_at" => Some(StrFn::ByteAt),
            "slice" => Some(StrFn::Slice),
            _ => None,
        }
    }

    /// The surface name, also used by the textual IR.
    pub fn as_str(self) -> &'static str {
        match self {
            StrFn::Len => "len",
            StrFn::ByteAt => "byte_at",
            StrFn::Slice => "slice",
        }
    }
}

impl IntrinsicOp {
    /// The family segment under `core::intrinsic::` (`math`, …).
    pub fn family(self) -> &'static str {
        match self {
            IntrinsicOp::Panic => "panic",
            IntrinsicOp::Math { .. } => "math",
            IntrinsicOp::Cell { .. } => "cell",
            IntrinsicOp::Str { .. } => "str",
        }
    }

    /// The op's name within its family (also the dialect mnemonic).
    pub fn name(self) -> &'static str {
        match self {
            IntrinsicOp::Panic => "panic",
            IntrinsicOp::Math { func, .. } => func.as_str(),
            IntrinsicOp::Cell { func, .. } => func.as_str(),
            IntrinsicOp::Str { func, .. } => func.as_str(),
        }
    }

    /// The family's packed immediate, as printed in the textual IR.
    pub fn imm(self) -> u32 {
        match self {
            IntrinsicOp::Panic => 0,
            IntrinsicOp::Math { flag, .. } => flag,
            IntrinsicOp::Cell { .. } => 0,
            IntrinsicOp::Str { .. } => 0,
        }
    }

    /// Rebuild a typed op from a textual-IR `(family, name, imm)` triple.
    /// `None` if the family or name is unknown.
    pub fn parse(family: &str, name: &str, imm: u32) -> Option<IntrinsicOp> {
        match family {
            "panic" if name == "panic" && imm == 0 => Some(IntrinsicOp::Panic),
            "math" => Some(IntrinsicOp::Math {
                func: MathFn::parse(name)?,
                flag: imm,
            }),
            "cell" if imm == 0 => Some(IntrinsicOp::Cell {
                func: CellFn::parse(name)?,
            }),
            "str" if imm == 0 => Some(IntrinsicOp::Str {
                func: StrFn::parse(name)?,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_op_round_trips_through_textual_triple() {
        for op in [
            IntrinsicOp::Panic,
            IntrinsicOp::Math {
                func: MathFn::Sqrt,
                flag: 0,
            },
            IntrinsicOp::Math {
                func: MathFn::Fma,
                flag: 127,
            },
            IntrinsicOp::Cell {
                func: CellFn::Alloc,
            },
            IntrinsicOp::Cell { func: CellFn::Rmw },
            IntrinsicOp::Cell {
                func: CellFn::InUse,
            },
        ] {
            assert_eq!(
                IntrinsicOp::parse(op.family(), op.name(), op.imm()),
                Some(op)
            );
        }
        assert_eq!(IntrinsicOp::parse("math", "nope", 0), None);
        assert_eq!(IntrinsicOp::parse("panic", "panic", 1), None);
        assert_eq!(IntrinsicOp::parse("panic", "abort", 0), None);
        assert_eq!(IntrinsicOp::parse("cell", "get", 1), None);
        assert!(
            IntrinsicOp::parse("cell", "in_use", 0).is_some(),
            "one textual family; the operand type carries the flavor"
        );
        assert_eq!(IntrinsicOp::parse("refcell", "in_use", 0), None);
        assert_eq!(IntrinsicOp::parse("atomic", "sqrt", 0), None);
    }

    #[test]
    fn names_round_trip() {
        for f in [
            MathFn::Sqrt,
            MathFn::Isnan,
            MathFn::Powf,
            MathFn::Fma,
            MathFn::Fpowi,
        ] {
            assert_eq!(MathFn::parse(f.as_str()), Some(f));
        }
        assert_eq!(MathFn::parse("log"), None);
        assert_eq!(MathFn::parse("sincos"), None);
        for f in [
            CellFn::Alloc,
            CellFn::Get,
            CellFn::Set,
            CellFn::Rmw,
            CellFn::InUse,
        ] {
            assert_eq!(CellFn::parse(f.as_str()), Some(f));
        }
    }

    #[test]
    fn fastmath_attr_spellings() {
        assert_eq!(FastMath(0).mlir_attr(), None);
        assert_eq!(
            FastMath(127).mlir_attr().as_deref(),
            Some("#arith.fastmath<fast>")
        );
        assert_eq!(
            FastMath(6).mlir_attr().as_deref(),
            Some("#arith.fastmath<nnan,ninf>")
        );
        assert_eq!(
            FastMath(1).mlir_attr().as_deref(),
            Some("#arith.fastmath<reassoc>")
        );
    }
}
