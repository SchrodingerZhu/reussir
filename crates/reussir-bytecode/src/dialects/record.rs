//! `reussir.record.*` constructors: record construction, projection, dispatch.

use crate::context::{Context, Type};
use crate::ir::{Op, Region, Value};

impl<'a> Context<'a> {
    /// `reussir.record.compound` — build a compound record from field values.
    pub fn record_compound(
        &self,
        fields: &[Value<'a>],
        record_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        self.op("reussir.record.compound")
            .operands(fields)
            .result(record_ty)
            .build_one()
    }

    /// `reussir.record.extract` — extract a field from a record by index.
    pub fn record_extract(
        &self,
        record: Value<'a>,
        index: i64,
        field_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let idx = self.attr_int(index, self.index());
        self.op("reussir.record.extract")
            .attrs(self.attr_dict(&[("index", idx)]))
            .operand(record)
            .result(field_ty)
            .build_one()
    }

    /// `reussir.record.variant` — build a tagged variant record.
    pub fn record_variant(
        &self,
        tag: i64,
        value: Value<'a>,
        variant_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let tag_attr = self.attr_int(tag, self.index());
        self.op("reussir.record.variant")
            .attrs(self.attr_dict(&[("tag", tag_attr)]))
            .operand(value)
            .result(variant_ty)
            .build_one()
    }

    /// `reussir.record.tag` — read the active tag of a variant behind a
    /// reference, as an `index`.
    pub fn record_tag(&self, variant_ref: Value<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.record.tag")
            .operand(variant_ref)
            .result(self.index())
            .build_one()
    }

    /// `reussir.record.dispatch` — structured branch over a variant's tag. One
    /// region per entry in `tag_sets`; each region's entry block takes a
    /// reference to the corresponding compound and ends with `reussir.scf.yield`.
    pub fn record_dispatch(
        &self,
        variant_ref: Value<'a>,
        tag_sets: &[&[i64]],
        regions: &[Region<'a>],
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        let set_attrs: Vec<_> = tag_sets.iter().map(|s| self.attr_dense_i64(s)).collect();
        let tag_sets_attr = self.attr_array(&set_attrs);
        let mut builder = self
            .op("reussir.record.dispatch")
            .attrs(self.attr_dict(&[("tagSets", tag_sets_attr)]))
            .operand(variant_ref)
            .results(result_types);
        for r in regions {
            builder = builder.region(*r);
        }
        builder.build()
    }
}
