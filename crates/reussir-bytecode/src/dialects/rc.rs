//! `reussir.rc.*` constructors: reference-counted box operations.

use crate::context::{Context, Type};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// `reussir.rc.inc` — increment a reference count. Consumes nothing.
    pub fn rc_inc(&self, rc: Value<'a>) -> Op<'a> {
        self.op("reussir.rc.inc").operand(rc).build_zero()
    }

    /// `reussir.rc.dec` — decrement a reference count.
    pub fn rc_dec(&self, rc: Value<'a>) -> Op<'a> {
        self.op("reussir.rc.dec").operand(rc).build_zero()
    }

    /// `reussir.rc.borrow` — borrow an rc pointer to obtain an access reference.
    pub fn rc_borrow(&self, rc: Value<'a>, ref_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.rc.borrow")
            .operand(rc)
            .result(ref_ty)
            .build_one()
    }

    /// `reussir.rc.freeze` — freeze a flex rc pointer into a rigid one.
    pub fn rc_freeze(&self, rc: Value<'a>, frozen_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.rc.freeze")
            .operand(rc)
            .result(frozen_ty)
            .build_one()
    }

    /// `reussir.rc.is_unique` — test whether an rc pointer is uniquely owned.
    pub fn rc_is_unique(&self, rc: Value<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.rc.is_unique")
            .operand(rc)
            .result(self.bool_ty())
            .build_one()
    }

    /// `reussir.rc.fetch` — read the current reference count as an `index`.
    pub fn rc_fetch(&self, rc: Value<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.rc.fetch")
            .operand(rc)
            .result(self.index())
            .build_one()
    }

    /// `reussir.rc.create` — wrap a value into a reference-counted box. The
    /// optional `token` and `region` operands are encoded with the
    /// `operandSegmentSizes` property the variadic-segment op requires.
    pub fn rc_create(
        &self,
        value: Value<'a>,
        token: Option<Value<'a>>,
        region: Option<Value<'a>>,
        rc_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let mut builder = self.op("reussir.rc.create").operand(value);
        if let Some(t) = token {
            builder = builder.operand(t);
        }
        if let Some(r) = region {
            builder = builder.operand(r);
        }
        let segments = self.attr_dense_i32(&[1, token.is_some() as i32, region.is_some() as i32]);
        builder
            .attrs(self.attr_dict(&[("operandSegmentSizes", segments)]))
            .result(rc_ty)
            .build_one()
    }
}
