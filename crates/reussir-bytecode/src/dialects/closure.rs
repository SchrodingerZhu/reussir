//! `reussir.closure.*` constructors: closure creation and application.

use crate::context::{Context, Type};
use crate::ir::{Op, Region, Value};

impl<'a> Context<'a> {
    /// `reussir.closure.create` — build a closure. The body region (isolated
    /// from above) takes the closure's arguments and ends with
    /// `reussir.closure.yield`. An optional allocation `token` and `vtable`
    /// symbol select the outlined form.
    pub fn closure_create(
        &self,
        token: Option<Value<'a>>,
        vtable: Option<&str>,
        body: Region<'a>,
        closure_rc_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        let mut builder = self.op("reussir.closure.create");
        if let Some(t) = token {
            builder = builder.operand(t);
        }
        if let Some(sym) = vtable {
            let attr = self.attr_symbol(sym);
            builder = builder.attrs(self.attr_dict(&[("vtable", attr)]));
        }
        builder
            .region(body)
            .result(closure_rc_ty)
            .isolated(true)
            .build_one()
    }

    /// `reussir.closure.yield` — yield the result from a closure body.
    pub fn closure_yield(&self, values: &[Value<'a>]) -> Op<'a> {
        self.op("reussir.closure.yield")
            .operands(values)
            .build_zero()
    }

    /// `reussir.closure.apply` — partially apply one argument to a closure.
    pub fn closure_apply(
        &self,
        arg: Value<'a>,
        closure: Value<'a>,
        applied_ty: Type<'a>,
    ) -> (Op<'a>, Value<'a>) {
        self.op("reussir.closure.apply")
            .operand(arg)
            .operand(closure)
            .result(applied_ty)
            .build_one()
    }

    /// `reussir.closure.eval` — evaluate a fully applied closure.
    pub fn closure_eval(
        &self,
        closure: Value<'a>,
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        self.op("reussir.closure.eval")
            .operand(closure)
            .results(result_types)
            .build()
    }

    /// `reussir.closure.uniqify` — ensure a closure is uniquely owned.
    pub fn closure_uniqify(&self, closure: Value<'a>, ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.closure.uniqify")
            .operand(closure)
            .result(ty)
            .build_one()
    }
}
