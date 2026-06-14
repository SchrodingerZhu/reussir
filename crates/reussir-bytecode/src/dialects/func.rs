//! `func` dialect constructors.

use crate::context::{Context, Type};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// A direct call: `func.call @callee(args) : (...) -> (...)`.
    pub fn func_call(
        &self,
        callee: &str,
        args: &[Value<'a>],
        result_types: &[Type<'a>],
    ) -> (Op<'a>, &'a [Value<'a>]) {
        let callee_attr = self.attr_symbol(callee);
        self.op("func.call")
            .attrs(self.attr_dict(&[("callee", callee_attr)]))
            .operands(args)
            .results(result_types)
            .build()
    }
}
