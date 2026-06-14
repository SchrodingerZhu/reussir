//! `scf` (structured control flow) dialect constructors.

use crate::context::{Context, Type};
use crate::ir::{Op, Region, Value};

impl<'a> Context<'a> {
    /// An `scf.if` with `then`/`else` regions. The regions' `scf.yield`
    /// terminators must yield values matching `result_types`.
    pub fn scf_if(
        &self,
        cond: Value<'a>,
        result_types: &[Type<'a>],
        then_region: Region<'a>,
        else_region: Region<'a>,
    ) -> (Op<'a>, &'a [Value<'a>]) {
        self.op("scf.if")
            .operand(cond)
            .results(result_types)
            .region(then_region)
            .region(else_region)
            .build()
    }

    /// An `scf.yield` terminator carrying the given values.
    pub fn scf_yield(&self, values: &[Value<'a>]) -> Op<'a> {
        self.op("scf.yield").operands(values).build_zero()
    }

    /// `scf.index_switch` — switch over an `index` value. `case_regions` is
    /// parallel to `cases`; `default_region` handles unmatched values. Each
    /// region ends with `scf.yield`.
    ///
    /// Structurally the default region is emitted first, then the case regions,
    /// matching the operation's ODS declaration order (the textual form prints
    /// the default last).
    pub fn scf_index_switch(
        &self,
        value: Value<'a>,
        cases: &[i64],
        case_regions: &[Region<'a>],
        default_region: Region<'a>,
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        debug_assert_eq!(
            case_regions.len(),
            cases.len(),
            "index_switch needs one region per case value"
        );
        let cases_attr = self.attr_dense_i64(cases);
        let mut builder = self
            .op("scf.index_switch")
            .attrs(self.attr_dict(&[("cases", cases_attr)]))
            .operand(value)
            .results(result_types)
            .region(default_region);
        for r in case_regions {
            builder = builder.region(*r);
        }
        builder.build()
    }
}
