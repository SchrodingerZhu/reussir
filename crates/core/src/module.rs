use crate::{Result, func::FunctionProto};
use std::{fmt::write, io::Write};

use crate::{
    CGContext, Context, Path,
    ir::{Block, Symbol},
    types::Type,
};

#[derive(Debug, Clone)]
pub struct ModuleInstance<'a> {
    pub ctx: &'a Context,
    pub functions: &'a [FunctionInstance<'a>],
}

#[derive(Debug, Clone)]
pub struct FunctionInstance<'a> {
    pub proto: &'a FunctionProto,
    pub symbol: &'a Symbol<'a>,
    pub body: &'a Block<'a>,
}

impl<'a> ModuleInstance<'a> {
    pub fn codegen<W: Write>(&self, ctx: &mut CGContext<W>) -> Result<()> {
        write!(ctx.output, "module @\"{:?}\" {{\n", self.ctx.module())?;
        ctx.identation += 1;
        for function in self.functions {
            function.codegen(ctx)?;
        }
        ctx.identation -= 1;
        writeln!(ctx.output, "}}")?;
        let mut locs = ctx.location_uniqifer.values().copied().collect::<Vec<_>>();
        locs.sort_unstable_by_key(|x| x.0);
        for (idx, attr) in locs.into_iter() {
            (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
            write!(ctx.output, "#loc{} = ", idx)?;
            attr.codegen(ctx)?;
            writeln!(ctx.output)?;
        }
        Ok(())
    }
}

impl<'a> FunctionInstance<'a> {
    pub fn codegen<W: Write>(&self, ctx: &mut CGContext<W>) -> Result<()> {
        (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
        write!(ctx.output, "func.func @\"{:?}\"(", self.symbol.path)?;
        for (i, param) in self.proto.params.iter().enumerate() {
            if i > 0 {
                write!(ctx.output, ", ")?;
            }
            param.codegen(ctx, i)?;
        }
        write!(ctx.output, ") -> ")?;
        self.proto.return_type.codegen(ctx)?;
        write!(ctx.output, " {{\n")?;
        ctx.identation += 1;
        self.body.codegen(ctx)?;
        ctx.identation -= 1;
        (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
        writeln!(ctx.output, "}}")?;
        Ok(())
    }
}
