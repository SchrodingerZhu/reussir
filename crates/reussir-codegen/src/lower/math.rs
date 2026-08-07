//! Building `math.<fn>` dialect ops for the `core::intrinsic::math`
//! intrinsics, through melior's generated ODS builders (`ods::math`).
//!
//! Every surfaced intrinsic is one op over scalar float operands with an
//! optional `fastmath` attribute; the shapes differ only in operand count,
//! operand names, and whether the result type is inferable
//! (`SameOperandsAndResultType`) or must be given (the `bool` classification
//! checks, and `fpowi` whose exponent operand differs).

use reussir_backend::melior::Context;
use reussir_backend::melior::dialect::ods::math;
use reussir_backend::melior::ir::attribute::Attribute;
use reussir_backend::melior::ir::{Location, Operation, Type, Value};
use reussir_core::intrinsic::MathFn;

/// Build the `math` op for `func` over `operands` (already checked to the
/// intrinsic's arity), with `fastmath` when the flag is non-empty. `result`
/// is used only by the shapes whose result is not inferable.
pub(super) fn math_operation<'c>(
    context: &'c Context,
    func: MathFn,
    operands: &[Value<'c, '_>],
    result: Type<'c>,
    fastmath: Option<Attribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    // One arm per op: the generated builders are distinct types, so the
    // dispatch cannot be a table — the macros keep each shape to one line.
    macro_rules! unary {
        ($builder:ident) => {{
            let b = math::$builder::new(context, location).operand(operands[0]);
            match fastmath {
                Some(fm) => b.fastmath(fm).build().into(),
                None => b.build().into(),
            }
        }};
    }
    macro_rules! check {
        ($builder:ident) => {{
            let b = math::$builder::new(context, location)
                .result(result)
                .operand(operands[0]);
            match fastmath {
                Some(fm) => b.fastmath(fm).build().into(),
                None => b.build().into(),
            }
        }};
    }
    macro_rules! binary {
        ($builder:ident) => {{
            let b = math::$builder::new(context, location)
                .lhs(operands[0])
                .rhs(operands[1]);
            match fastmath {
                Some(fm) => b.fastmath(fm).build().into(),
                None => b.build().into(),
            }
        }};
    }

    match func {
        MathFn::Absf => unary!(AbsFOperationBuilder),
        MathFn::Acos => unary!(AcosOperationBuilder),
        MathFn::Acosh => unary!(AcoshOperationBuilder),
        MathFn::Asin => unary!(AsinOperationBuilder),
        MathFn::Asinh => unary!(AsinhOperationBuilder),
        MathFn::Atan => unary!(AtanOperationBuilder),
        MathFn::Atanh => unary!(AtanhOperationBuilder),
        MathFn::Cbrt => unary!(CbrtOperationBuilder),
        MathFn::Ceil => unary!(CeilOperationBuilder),
        MathFn::Cos => unary!(CosOperationBuilder),
        MathFn::Cosh => unary!(CoshOperationBuilder),
        MathFn::Erf => unary!(ErfOperationBuilder),
        MathFn::Erfc => unary!(ErfcOperationBuilder),
        MathFn::Exp => unary!(ExpOperationBuilder),
        MathFn::Exp2 => unary!(Exp2OperationBuilder),
        MathFn::Expm1 => unary!(ExpM1OperationBuilder),
        MathFn::Floor => unary!(FloorOperationBuilder),
        MathFn::Log10 => unary!(Log10OperationBuilder),
        MathFn::Log1p => unary!(Log1pOperationBuilder),
        MathFn::Log2 => unary!(Log2OperationBuilder),
        MathFn::Round => unary!(RoundOperationBuilder),
        MathFn::Roundeven => unary!(RoundEvenOperationBuilder),
        MathFn::Rsqrt => unary!(RsqrtOperationBuilder),
        MathFn::Sin => unary!(SinOperationBuilder),
        MathFn::Sinh => unary!(SinhOperationBuilder),
        MathFn::Sqrt => unary!(SqrtOperationBuilder),
        MathFn::Tan => unary!(TanOperationBuilder),
        MathFn::Tanh => unary!(TanhOperationBuilder),
        MathFn::Trunc => unary!(TruncOperationBuilder),
        MathFn::Isfinite => check!(IsFiniteOperationBuilder),
        MathFn::Isinf => check!(IsInfOperationBuilder),
        MathFn::Isnan => check!(IsNaNOperationBuilder),
        MathFn::Isnormal => check!(IsNormalOperationBuilder),
        MathFn::Atan2 => binary!(Atan2OperationBuilder),
        MathFn::Copysign => binary!(CopySignOperationBuilder),
        MathFn::Powf => binary!(PowFOperationBuilder),
        MathFn::Fma => {
            let b = math::FmaOperationBuilder::new(context, location)
                .a(operands[0])
                .b(operands[1])
                .c(operands[2]);
            match fastmath {
                Some(fm) => b.fastmath(fm).build().into(),
                None => b.build().into(),
            }
        }
        MathFn::Fpowi => {
            let b = math::FPowIOperationBuilder::new(context, location)
                .result(result)
                .lhs(operands[0])
                .rhs(operands[1]);
            match fastmath {
                Some(fm) => b.fastmath(fm).build().into(),
                None => b.build().into(),
            }
        }
        // The integer ops carry no fastmath attribute (the checker never
        // passes one), and their result type is inferable.
        MathFn::Ctpop => math::CtPopOperationBuilder::new(context, location)
            .operand(operands[0])
            .build()
            .into(),
        MathFn::Ctlz => math::CountLeadingZerosOperationBuilder::new(context, location)
            .operand(operands[0])
            .build()
            .into(),
        MathFn::Cttz => math::CountTrailingZerosOperationBuilder::new(context, location)
            .operand(operands[0])
            .build()
            .into(),
    }
}
