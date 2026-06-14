//! `reussir.region.*` constructors: region (arena) scopes.

use crate::context::{Context, Type};
use crate::ir::{Op, Region, Value};

impl<'a> Context<'a> {
    /// `reussir.region.run` — run a region scope. The body's entry block takes a
    /// single `!reussir.region` argument and ends with `reussir.region.yield`.
    pub fn region_run(
        &self,
        body: Region<'a>,
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        self.op("reussir.region.run")
            .results(result_types)
            .region(body)
            .build()
    }

    /// `reussir.region.yield` — yield from a `reussir.region.run` body.
    pub fn region_yield(&self, values: &[Value<'a>]) -> Op<'a> {
        self.op("reussir.region.yield")
            .operands(values)
            .build_zero()
    }
}
