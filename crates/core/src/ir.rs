use std::cell::Cell;

use indexmap::IndexMap;
use ustr::Ustr;

use crate::{
    Location, Path,
    literal::{FloatLiteral, IntegerLiteral},
    types::Type,
};

pub struct OutputValue {
    pub value: ValID,
    pub ty: Type,
    pub name: Option<Ustr>,
}

pub struct Operation<'a> {
    location: Option<Location>,
    output: Option<&'a OutputValue>,
    kind: OperationKind<'a>,
}

pub type ValID = usize;

// We will use structured control flow at this level, so there is no need for a region of multiple blocks.
pub struct Block<'a>(&'a [Operation<'a>]);

pub struct Symbol<'a> {
    pub path: Path,
    pub type_params: Option<&'a [Type]>,
}

pub enum FieldIdentifer {
    Named(Ustr),
    Indexed(usize),
}

pub struct SwitchCase<'a> {
    pub variants: &'a [Ustr],
    pub body: Block<'a>,
}

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
        else_region: &'a Block<'a>,
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
    ConstInt(Box<IntegerLiteral>),
    ConstFloat(Box<FloatLiteral>),
}

pub struct IRBuilder<'a> {
    arena: &'a bumpalo::Bump,
    next_val: ValID,
}
