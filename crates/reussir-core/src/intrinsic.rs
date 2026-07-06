//! The built-in `core` package's math intrinsics.
//!
//! Reussir source reaches them as `core::intrinsic::math::<name>(args…, flag)`
//! — the package name `core` is reserved, so the path can never collide with
//! user code. Each intrinsic maps 1:1 onto an MLIR `math` dialect op; the
//! trailing argument is always a **constant** `i32` fast-math flag
//! ([`FastMath`]), and the value operands share one `FloatingPoint`-bounded
//! type (given as an optional explicit type argument, or inferred).
//!
//! The surfaced set matches the reference frontend: the float unary group,
//! the `is*` classification checks (returning `bool`), the float binary
//! group, `fma`, and `fpowi` (float base, `i32` exponent).

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
}

impl MathKind {
    /// How many *value* arguments the intrinsic takes (the constant fast-math
    /// flag is one more).
    pub fn value_args(self) -> usize {
        match self {
            MathKind::Unary | MathKind::Check => 1,
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
/// #344). Unlike [`IntrinsicOp`], these are not spelled under `core::intrinsic`
/// — `Splat`/`Tabulate` are `Array::<fn>` path calls and the rest are
/// method-style calls on an array-typed receiver — and they carry their own
/// HIR/MIR node (`ExprKind::ArrayOp`) because `Tabulate`/`Fold` hold an inline
/// kernel body, which the generic `Intrinsic` node cannot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArrayFn {
    /// `Array::splat(v)`: every element is `v`. Value operands: `[v]`.
    Splat,
    /// `Array::tabulate(|i, …| e)`: element at each index tuple is the kernel
    /// applied to the (row-major) indices. No value operands.
    Tabulate,
    /// `a.get(i, …)`: checked element read. Value operands: `[a, i…]`; `a` is
    /// borrowed, not consumed.
    Get,
    /// `a.set(i, …, v)`: checked element write, consuming `a` and returning
    /// the updated array (copy-on-write when shared). Value operands:
    /// `[a, i…, v]`.
    Set,
    /// `a.fold(init, |acc, x| e)`: row-major reduction. Value operands:
    /// `[a, init]`; `a` is borrowed, not consumed.
    Fold,
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
        }
    }

    /// Whether the op takes an inline kernel body.
    pub fn has_kernel(self) -> bool {
        matches!(self, ArrayFn::Tabulate | ArrayFn::Fold)
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
    /// `core::intrinsic::math::<func>`, with an `arith.fastmath` flag.
    Math { func: MathFn, flag: u32 },
}

impl IntrinsicOp {
    /// The family segment under `core::intrinsic::` (`math`, …).
    pub fn family(self) -> &'static str {
        match self {
            IntrinsicOp::Math { .. } => "math",
        }
    }

    /// The op's name within its family (also the dialect mnemonic).
    pub fn name(self) -> &'static str {
        match self {
            IntrinsicOp::Math { func, .. } => func.as_str(),
        }
    }

    /// The family's packed immediate, as printed in the textual IR.
    pub fn imm(self) -> u32 {
        match self {
            IntrinsicOp::Math { flag, .. } => flag,
        }
    }

    /// Rebuild a typed op from a textual-IR `(family, name, imm)` triple.
    /// `None` if the family or name is unknown.
    pub fn parse(family: &str, name: &str, imm: u32) -> Option<IntrinsicOp> {
        match family {
            "math" => Some(IntrinsicOp::Math {
                func: MathFn::parse(name)?,
                flag: imm,
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
            IntrinsicOp::Math {
                func: MathFn::Sqrt,
                flag: 0,
            },
            IntrinsicOp::Math {
                func: MathFn::Fma,
                flag: 127,
            },
        ] {
            assert_eq!(
                IntrinsicOp::parse(op.family(), op.name(), op.imm()),
                Some(op)
            );
        }
        assert_eq!(IntrinsicOp::parse("math", "nope", 0), None);
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
