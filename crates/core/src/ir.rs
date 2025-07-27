use std::fmt::Write;

use ustr::Ustr;

use crate::{
    Location, Path,
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

pub struct CGContext<'a> {
    output: std::fmt::Formatter<'a>,
    identation: usize,
}

impl Operation<'_> {
    pub fn codegen(&self, ctx: &mut CGContext) -> std::fmt::Result {
        (0..ctx.identation).try_for_each(|_| ctx.output.write_char('\t'))?;
        match &self.output {
            Some(OutputValue {
                value,
                ty: _,
                name: _,
            }) => {
                write!(ctx.output, "%{value} = ")?;
                self.kind.codegen(ctx)?;
                write!(ctx.output, " : <todo-print-type>")?;
            }
            None => self.kind.codegen(ctx)?,
        }
        ctx.output.write_char('\n')
    }
}

impl OperationKind<'_> {
    pub fn codegen(&self, ctx: &mut CGContext) -> std::fmt::Result {
        match self {
            OperationKind::FnCall { target, args } => todo!(),
            OperationKind::CtorCall { ty, args } => todo!(),
            OperationKind::VariantCall { ty, variant, args } => todo!(),
            OperationKind::PartialApply { target, arg } => todo!(),
            OperationKind::Evaluate { target } => todo!(),
            OperationKind::FunctionToClosure { target } => todo!(),
            OperationKind::InlineClosure { region } => todo!(),
            OperationKind::Condition {
                cond,
                then_region,
                else_region,
            } => todo!(),
            OperationKind::Yield(_) => todo!(),
            OperationKind::Return(_) => todo!(),
            OperationKind::Switch { cond, cases } => todo!(),
            OperationKind::VariantCast { target, ty } => todo!(),
            OperationKind::Proj { target, field } => todo!(),
            OperationKind::ConstInt(integer_literal) => {
                write!(ctx.output, "arith.constant {integer_literal}")
            }
            OperationKind::ConstFloat(float_literal) => {
                write!(ctx.output, "arith.constant {float_literal}")
            }
            OperationKind::ConstantBool(value) => {
                let val = if *value { 1 } else { 0 };
                write!(ctx.output, "arith.constant {val}")
            }
            OperationKind::Unit => todo!(),
            OperationKind::Panic(_) => todo!(),
            OperationKind::Poison => todo!(),
        }
    }
}

pub type ValID = usize;

#[derive(Debug, Clone)]
// We will use structured control flow at this level, so there is no need for a region of multiple blocks.
pub struct Block<'a>(pub &'a [Operation<'a>]);

#[derive(Debug, Clone)]
pub struct Symbol<'a> {
    pub path: Path,
    pub type_params: Option<&'a [&'a Type]>,
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
        args: Option<&'a [ValID]>,
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
    Condition {
        cond: ValID,
        then_region: &'a Block<'a>,
        else_region: Option<&'a Block<'a>>,
    },
    Yield(Option<ValID>),
    Return(Option<ValID>),
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
