//! Ergonomic builders for the most common top-level constructs.
//!
//! These sit on top of the generic [`crate::ir::OpBuilder`] and produce the
//! handful of structural operations every module needs — the module wrapper and
//! function definitions — so a frontend can focus on emitting bodies. Operation
//! bodies are filled by pushing arbitrary operations built from the [`Context`],
//! which keeps the builder open to the full Reussir/arith/scf instruction set
//! without baking each one in here.

use crate::context::{Attr, Context, Type};
use crate::ir::{Op, Value};

/// Builds a `builtin.module` and the functions inside it.
pub struct ModuleBuilder<'c, 'a> {
    ctx: &'c Context<'a>,
    name: &'a str,
    ops: Vec<Op<'a>>,
}

impl<'c, 'a> ModuleBuilder<'c, 'a> {
    /// Start a module with the given symbol name.
    pub fn new(ctx: &'c Context<'a>, name: &str) -> Self {
        ModuleBuilder {
            ctx,
            name: ctx.alloc_str(name),
            ops: Vec::new(),
        }
    }

    /// The context this module is built in.
    pub fn ctx(&self) -> &'c Context<'a> {
        self.ctx
    }

    /// Append an already-built top-level operation (e.g. a global) to the module
    /// body.
    pub fn push(&mut self, op: Op<'a>) {
        self.ops.push(op);
    }

    /// Define a function with the given parameter and result types. The closure
    /// receives a [`FunctionBuilder`] whose entry block already has one argument
    /// per parameter type; it fills the body and must end with a terminator.
    pub fn function(
        &mut self,
        name: &str,
        params: &[Type<'a>],
        results: &[Type<'a>],
        body: impl FnOnce(&mut FunctionBuilder<'c, 'a>),
    ) {
        let mut f = FunctionBuilder::new(self.ctx, name, params, results);
        body(&mut f);
        self.ops.push(f.finish());
    }

    /// Finish building and return the `builtin.module` operation.
    pub fn finish(self) -> Op<'a> {
        let attrs = self
            .ctx
            .attr_dict(&[("sym_name", self.ctx.attr_str(self.name))]);
        let entry = self.ctx.block_unknown_locs(&[], &self.ops);
        let region = self.ctx.region(&[entry]);
        self.ctx
            .op("builtin.module")
            .attrs(attrs)
            .region(region)
            .isolated(true)
            .build_zero()
    }
}

/// Builds a single `func.func` definition.
pub struct FunctionBuilder<'c, 'a> {
    ctx: &'c Context<'a>,
    name: &'a str,
    params: Vec<Type<'a>>,
    results: Vec<Type<'a>>,
    args: Vec<Value<'a>>,
    ops: Vec<Op<'a>>,
    visibility: Option<&'a str>,
}

impl<'c, 'a> FunctionBuilder<'c, 'a> {
    fn new(ctx: &'c Context<'a>, name: &str, params: &[Type<'a>], results: &[Type<'a>]) -> Self {
        let args = params.iter().map(|&ty| ctx.value(ty)).collect();
        FunctionBuilder {
            ctx,
            name: ctx.alloc_str(name),
            params: params.to_vec(),
            results: results.to_vec(),
            args,
            ops: Vec::new(),
            visibility: None,
        }
    }

    /// The context this function is built in.
    pub fn ctx(&self) -> &'c Context<'a> {
        self.ctx
    }

    /// The `i`-th entry block argument (the `i`-th function parameter).
    pub fn arg(&self, i: usize) -> Value<'a> {
        self.args[i]
    }

    /// All entry block arguments.
    pub fn args(&self) -> &[Value<'a>] {
        &self.args
    }

    /// Mark the function `private`.
    pub fn private(&mut self) {
        self.visibility = Some("private");
    }

    /// Append an operation to the function body.
    pub fn push(&mut self, op: Op<'a>) {
        self.ops.push(op);
    }

    /// Append a `func.return` terminator yielding the given values.
    pub fn ret(&mut self, values: &[Value<'a>]) {
        let op = self.ctx.op("func.return").operands(values).build_zero();
        self.ops.push(op);
    }

    fn finish(self) -> Op<'a> {
        let ctx = self.ctx;
        let func_ty = ctx.func_type(&self.params, &self.results);
        let mut entries: Vec<(&str, Attr<'a>)> = vec![
            ("function_type", ctx.attr_type(func_ty)),
            ("sym_name", ctx.attr_str(self.name)),
        ];
        if let Some(vis) = self.visibility {
            entries.push(("sym_visibility", ctx.attr_str(vis)));
        }
        let attrs = ctx.attr_dict(&entries);
        let entry = ctx.block_unknown_locs(&self.args, &self.ops);
        let region = ctx.region(&[entry]);
        ctx.op("func.func")
            .attrs(attrs)
            .region(region)
            .isolated(true)
            .build_zero()
    }
}
