//! `arith` dialect constructors: arithmetic, comparison, conversion, constants.

use crate::context::{Context, Type};
use crate::dialects::CmpIPredicate;
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// A binary `arith` operation whose operands and result share one type, e.g.
    /// `arith.addi`, `arith.subi`, `arith.muli`, `arith.addf`, `arith.divf`.
    pub fn arith_binary(
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

    /// An integer constant: `arith.constant value : ty`.
    pub fn arith_constant_int(&self, value: i64, ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("arith.constant")
            .attrs(self.attr_dict(&[("value", self.attr_int(value, ty))]))
            .result(ty)
            .build_one()
    }

    /// An integer comparison producing an `i1`: `arith.cmpi pred, lhs, rhs`.
    pub fn arith_cmpi(
        &self,
        pred: CmpIPredicate,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let i64 = self.int(64);
        let pred_attr = self.attr_int(pred as i64, i64);
        self.op("arith.cmpi")
            .attrs(self.attr_dict(&[("predicate", pred_attr)]))
            .operand(lhs)
            .operand(rhs)
            .result(self.bool_ty())
            .build_one()
    }

    /// A `cond ? a : b` selection: `arith.select cond, a, b`.
    pub fn arith_select(
        &self,
        cond: Value<'a>,
        a: Value<'a>,
        b: Value<'a>,
        ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        self.op("arith.select")
            .operand(cond)
            .operand(a)
            .operand(b)
            .result(ty)
            .build_one()
    }

    /// A single-operand `arith` conversion, e.g. `arith.extsi`, `arith.trunci`,
    /// `arith.sitofp`, `arith.fptosi`, `arith.bitcast`. The result type names the
    /// target type.
    pub fn arith_cast(&self, name: &str, value: Value<'a>, to: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op(name).operand(value).result(to).build_one()
    }
}
