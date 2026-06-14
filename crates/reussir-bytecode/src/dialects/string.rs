//! `reussir.str.*` constructors: string globals and operations.

use crate::context::{Context, Type};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// `reussir.str.global` — define a named global string with byte payload.
    pub fn str_global(&self, name: &str, payload: &str) -> Op<'a> {
        let attrs = self.attr_dict(&[
            ("payload", self.attr_str(payload)),
            ("sym_name", self.attr_str(name)),
        ]);
        self.op("reussir.str.global").attrs(attrs).build_zero()
    }

    /// `reussir.str.literal` — reference a global string by symbol, yielding a
    /// global-scope `!reussir.str`.
    pub fn str_literal(&self, sym: &str, str_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        let attr = self.attr_symbol(sym);
        self.op("reussir.str.literal")
            .attrs(self.attr_dict(&[("sym_name", attr)]))
            .result(str_ty)
            .build_one()
    }

    /// `reussir.str.cast` — cast a global string to a local string.
    pub fn str_cast(&self, global_str: Value<'a>, local_str_ty: Type<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.str.cast")
            .operand(global_str)
            .result(local_str_ty)
            .build_one()
    }

    /// `reussir.str.len` — the byte length of a string, as `index`.
    pub fn str_len(&self, s: Value<'a>) -> (Op<'a>, Value<'a>) {
        self.op("reussir.str.len")
            .operand(s)
            .result(self.index())
            .build_one()
    }

    /// `reussir.str.select` — match a string against patterns, yielding the
    /// matched index (`index`) and a found flag (`i1`).
    pub fn str_select(&self, s: Value<'a>, patterns: &[&str]) -> (Op<'a>, &'a [Value<'a>]) {
        let pats: Vec<_> = patterns.iter().map(|p| self.attr_str(p)).collect();
        let patterns_attr = self.attr_array(&pats);
        self.op("reussir.str.select")
            .attrs(self.attr_dict(&[("patterns", patterns_attr)]))
            .operand(s)
            .result(self.index())
            .result(self.bool_ty())
            .build()
    }
}
