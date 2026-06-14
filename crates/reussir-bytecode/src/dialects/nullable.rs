//! `reussir.nullable.*` constructors: nullable pointer handling.

use crate::context::{Context, Type};
use crate::ir::{Op, Region, Value};

impl<'a> Context<'a> {
    /// `reussir.nullable.check` — test whether a nullable is null, as `i1`.
    pub fn nullable_check(&self, nullable: Value<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.nullable.check")
            .operand(nullable)
            .result(self.bool_ty())
            .build_one()
    }

    /// `reussir.nullable.create` — build a nullable, optionally from an inner
    /// non-null pointer.
    pub fn nullable_create(
        &self,
        inner: Option<Value<'a>>,
        nullable_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let mut builder = self.op("reussir.nullable.create");
        if let Some(v) = inner {
            builder = builder.operand(v);
        }
        builder.result(nullable_ty).build_one()
    }

    /// `reussir.nullable.dispatch` — branch on a nullable. The `nonnull` region's
    /// entry block takes the unwrapped pointer; the `null` region takes none.
    /// Both end with `reussir.scf.yield`.
    pub fn nullable_dispatch(
        &self,
        nullable: Value<'a>,
        nonnull: Region<'a>,
        null: Region<'a>,
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        self.op("reussir.nullable.dispatch")
            .operand(nullable)
            .results(result_types)
            .region(nonnull)
            .region(null)
            .build()
    }
}
