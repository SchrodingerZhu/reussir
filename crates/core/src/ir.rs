use ustr::Ustr;

use crate::{Location, Path, types::Type};

pub struct TypedValue {
    pub value: ValID,
    pub ty: Type,
}

pub struct Operation {
    location: Option<Location>,
    output: Option<Box<TypedValue>>,
    kind: OperationKind,
}
type TypeBox = Box<Type>;
type SymbolBox = Box<Symbol>;

pub type ValID = usize;

pub struct Region(Box<[Block]>);
// Do I need block arguments?
pub struct Block(Box<[Operation]>);

pub struct Symbol {
    pub path: Path,
    pub type_params: Option<Box<[Type]>>,
}

pub enum FieldIdentifer {
    Named(Ustr),
    Indexed(usize),
}

pub struct SwitchCase {
    pub variants: Box<[Ustr]>,
    pub region: Box<Region>,
}

pub enum OperationKind {
    FnCall {
        target: SymbolBox,
        args: Option<Box<[ValID]>>,
    },
    CtorCall {
        ty: SymbolBox,
        args: Option<Box<[ValID]>>,
    },
    VariantCall {
        ty: SymbolBox,
        variant: Ustr,
        args: Option<Box<[ValID]>>,
    },
    PartialApply {
        target: ValID,
        arg: ValID,
    },
    Evaluate {
        target: ValID,
    },
    FunctionToClosure {
        target: SymbolBox,
    },
    InlineClosure {
        region: Box<Region>,
    },
    Condition {
        cond: ValID,
        then_region: Box<Region>,
        else_region: Box<Region>,
    },
    Yield(Option<ValID>),
    Return(Option<ValID>),
    Switch {
        cond: ValID,
        cases: Box<[SwitchCase]>,
    },
    VariantCast {
        target: ValID,
        ty: SymbolBox,
    },
    Proj {
        target: ValID,
        field: FieldIdentifer,
    },
}
