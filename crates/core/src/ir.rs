use crate::Result;
use std::io::Write;
use ustr::Ustr;

use crate::{
    CGContext, Location, Path,
    literal::{FloatLiteral, IntegerLiteral},
    types::Type,
};

#[derive(Debug, Clone)]
pub struct OutputValue<'a> {
    pub value: ValID,
    pub ty: &'a Type,
    pub name: Option<Ustr>,
}

#[derive(Debug, Clone)]
pub struct Operation<'a> {
    pub location: Option<Location>,
    pub output: Option<&'a OutputValue<'a>>,
    pub kind: OperationKind<'a>,
}

impl Operation<'_> {
    pub fn codegen<W: Write>(&self, ctx: &mut CGContext<W>) -> Result<()> {
        (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
        self.kind.codegen(ctx, self.output)?;
        if let Some(location) = self.location.and_then(|x| ctx.location_to_line_span(x)) {
            write!(ctx.output, " loc(#loc{})", location)?;
        }
        writeln!(ctx.output)?;
        Ok(())
    }
}

impl OperationKind<'_> {
    pub fn codegen<W: Write>(
        &self,
        ctx: &mut CGContext<W>,
        out: Option<&OutputValue<'_>>,
    ) -> Result<()> {
        match self {
            OperationKind::FnCall { target, args } => {
                if let Some(intrinsic) = target.get_intrinsic() {
                    intrinsic.codegen(ctx, *args, out)?;
                    return Ok(());
                }
                if let Some(out) = out {
                    write!(ctx.output, "%{} = ", out.value)?;
                }
                write!(ctx.output, "func.call ")?;
                target.codegen(ctx)?;
                write!(ctx.output, "(")?;
                if let Some(args) = args {
                    for (idx, (value, _)) in args.iter().enumerate() {
                        if idx > 0 {
                            write!(ctx.output, ", ")?;
                        }
                        write!(ctx.output, "%{value}")?;
                    }
                }
                write!(ctx.output, ") : (")?;
                if let Some(args) = args {
                    for (idx, (_, ty)) in args.iter().enumerate() {
                        if idx > 0 {
                            write!(ctx.output, ", ")?;
                        }
                        ty.codegen(ctx)?;
                    }
                }
                write!(ctx.output, ") -> ")?;
                if let Some(out) = out {
                    out.ty.codegen(ctx)?;
                } else {
                    write!(ctx.output, "()")?;
                }
            }
            OperationKind::CtorCall { ty, args } => todo!(),
            OperationKind::VariantCall { ty, variant, args } => todo!(),
            OperationKind::PartialApply { target, arg } => todo!(),
            OperationKind::Evaluate { target } => todo!(),
            OperationKind::FunctionToClosure { target } => todo!(),
            OperationKind::InlineClosure { region } => todo!(),
            OperationKind::ScfIf {
                cond,
                then_region,
                else_region,
            } => {
                if let Some(out) = out {
                    write!(ctx.output, "%{} = ", out.value)?;
                }
                write!(ctx.output, "scf.if %{}", cond)?;
                if let Some(out) = out {
                    write!(ctx.output, " -> ")?;
                    out.ty.codegen(ctx)?;
                }
                write!(ctx.output, " {{\n")?;
                ctx.identation += 1;
                then_region.codegen(ctx)?;
                ctx.identation -= 1;
                if let Some(else_region) = else_region {
                    (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
                    writeln!(ctx.output, "}} else {{")?;
                    ctx.identation += 1;
                    else_region.codegen(ctx)?;
                    ctx.identation -= 1;
                }
                (0..ctx.identation).try_for_each(|_| write!(ctx.output, "\t"))?;
                write!(ctx.output, "}}")?;
            }
            OperationKind::ScfYield(val) => {
                if let Some((value, ty)) = val {
                    write!(ctx.output, "scf.yield %{} : ", value)?;
                    ty.codegen(ctx)?;
                } else {
                    write!(ctx.output, "scf.yield")?;
                }
            }
            OperationKind::Return(val) => {
                write!(ctx.output, "func.return")?;
                if let Some((value, ty)) = val {
                    write!(ctx.output, " %{}", value)?;
                    write!(ctx.output, " : ")?;
                    ty.codegen(ctx)?;
                }
            }
            OperationKind::Switch { cond, cases } => todo!(),
            OperationKind::VariantCast { target, ty } => todo!(),
            OperationKind::Proj { target, field } => todo!(),
            OperationKind::ConstInt(integer_literal) => {
                let out = out.unwrap();
                write!(
                    ctx.output,
                    "%{} = arith.constant {integer_literal} : ",
                    out.value
                )?;
                out.ty.codegen(ctx)?;
            }
            OperationKind::ConstFloat(float_literal) => {
                let out = out.unwrap();
                write!(
                    ctx.output,
                    "%{} = arith.constant {float_literal} : ",
                    out.value
                )?;
                out.ty.codegen(ctx)?;
            }
            OperationKind::ConstantBool(value) => {
                let out = out.unwrap();
                let val = if *value { 1 } else { 0 };
                write!(ctx.output, "%{} = arith.constant {val} : ", out.value)?;
                out.ty.codegen(ctx)?;
            }
            OperationKind::Unit => todo!(),
            OperationKind::Panic(_) => todo!(),
            OperationKind::Poison => todo!(),
        }
        Ok(())
    }
}

pub type ValID = usize;

#[derive(Debug, Clone)]
// We will use structured control flow at this level, so there is no need for a region of multiple blocks.
pub struct Block<'a>(pub &'a [Operation<'a>]);

impl Block<'_> {
    pub fn codegen<W: Write>(&self, ctx: &mut CGContext<W>) -> Result<()> {
        for operation in self.0 {
            operation.codegen(ctx)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Symbol<'a> {
    pub path: Path,
    pub type_params: Option<&'a [&'a Type]>,
}

impl<'a> Symbol<'a> {
    // TODO: we need mangling
    pub fn codegen<W: Write>(&self, ctx: &mut CGContext<W>) -> Result<()> {
        write!(ctx.output, "@\"{:?}", self.path)?;
        if let Some(type_params) = self.type_params {
            write!(ctx.output, "<")?;
            for (i, param) in type_params.iter().enumerate() {
                if i > 0 {
                    write!(ctx.output, ",")?;
                }
                param.codegen(ctx)?;
            }
            write!(ctx.output, ">")?;
        }
        write!(ctx.output, "\"")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum FieldIdentifer {
    Named(Ustr),
    Indexed(usize),
}

#[derive(Debug, Clone)]
pub struct SwitchCase<'a> {
    pub variants: &'a [Ustr],
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub enum OperationKind<'a> {
    FnCall {
        target: &'a Symbol<'a>,
        args: Option<&'a [(ValID, &'a Type)]>,
    },
    CtorCall {
        ty: &'a Symbol<'a>,
        args: Option<&'a [ValID]>,
    },
    VariantCall {
        ty: &'a Symbol<'a>,
        variant: Ustr,
        args: Option<&'a [ValID]>,
    },
    PartialApply {
        target: ValID,
        arg: ValID,
    },
    Evaluate {
        target: ValID,
    },
    FunctionToClosure {
        target: &'a Symbol<'a>,
    },
    InlineClosure {
        region: &'a Block<'a>,
    },
    ScfIf {
        cond: ValID,
        then_region: &'a Block<'a>,
        else_region: Option<&'a Block<'a>>,
    },
    ScfYield(Option<(ValID, &'a Type)>),
    Return(Option<(ValID, &'a Type)>),
    Switch {
        cond: ValID,
        cases: &'a [SwitchCase<'a>],
    },
    VariantCast {
        target: ValID,
        ty: &'a Symbol<'a>,
    },
    Proj {
        target: ValID,
        field: FieldIdentifer,
    },
    ConstInt(&'a IntegerLiteral),
    ConstFloat(&'a FloatLiteral),
    ConstantBool(bool),
    Unit,
    Panic(&'a str),
    Poison,
}
