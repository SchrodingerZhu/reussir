//! `reussir.ref.*` constructors: reference (access pointer) operations.

use crate::context::{Context, Type};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// `reussir.ref.load` — load the value behind a reference.
    pub fn ref_load(&self, reference: Value<'a>, value_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.ref.load")
            .operand(reference)
            .result(value_ty)
            .build_one()
    }

    /// `reussir.ref.store` — store a value through a reference with field
    /// capability.
    pub fn ref_store(&self, reference: Value<'a>, value: Value<'a>) -> Op<'a> {
        self.op("reussir.ref.store")
            .operand(reference)
            .operand(value)
            .build_zero()
    }

    /// `reussir.ref.project` — project to a field reference by index.
    pub fn ref_project(
        &self,
        reference: Value<'a>,
        index: i64,
        field_ref_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let idx = self.attr_int(index, self.index());
        self.op("reussir.ref.project")
            .attrs(self.attr_dict(&[("index", idx)]))
            .operand(reference)
            .result(field_ref_ty)
            .build_one()
    }

    /// `reussir.ref.spilled` — spill a value to a stack reference.
    pub fn ref_spilled(&self, value: Value<'a>, ref_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.ref.spilled")
            .operand(value)
            .result(ref_ty)
            .build_one()
    }

    /// `reussir.ref.drop` — destruct the element behind a reference in place.
    /// `variant` selects the active variant arm when the referent is a variant.
    pub fn ref_drop(&self, reference: Value<'a>, variant: Option<i64>) -> Op<'a> {
        let mut builder = self.op("reussir.ref.drop").operand(reference);
        if let Some(v) = variant {
            let attr = self.attr_int(v, self.index());
            builder = builder.attrs(self.attr_dict(&[("variant", attr)]));
        }
        builder.build_zero()
    }

    /// `reussir.ref.acquire` — acquire ownership of the element behind a
    /// reference in place.
    pub fn ref_acquire(&self, reference: Value<'a>, variant: Option<i64>) -> Op<'a> {
        let mut builder = self.op("reussir.ref.acquire").operand(reference);
        if let Some(v) = variant {
            let attr = self.attr_int(v, self.index());
            builder = builder.attrs(self.attr_dict(&[("variant", attr)]));
        }
        builder.build_zero()
    }
}
