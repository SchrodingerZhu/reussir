//! `math` dialect constructors: unary and binary math operations.

use crate::context::{Context, Type};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// A unary `math` operation, e.g. `math.sqrt`, `math.sin`, `math.exp`. The
    /// `fastmath` property defaults to `none` and is left implicit.
    pub fn math_unary(&self, name: &str, value: Value<'a>, ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op(name).operand(value).result(ty).build_one()
    }

    /// A binary `math` operation, e.g. `math.powf`, `math.atan2`, `math.copysign`.
    pub fn math_binary(
        &self,
        name: &str,
        lhs: Value<'a>,
        rhs: Value<'a>,
        ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        self.op(name)
            .operand(lhs)
            .operand(rhs)
            .result(ty)
            .build_one()
    }
}
