use ustr::Ustr;

use crate::{Location, Path, types::Type};

pub struct Operation {
    location: Option<Location>,
    output: Option<usize>,
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

pub enum OperationKind {
    FnCall {
        target: SymbolBox,
        args: Option<Box<[ValID]>>,
        ret: Option<TypeBox>,
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
        ret: TypeBox,
    },
    FunctionToClosure {
        target: SymbolBox,
        ret: TypeBox,
    },
    InlineClosure {
        region: Box<Region>,
        ret: TypeBox,
    },
    If {
        cond: ValID,
        then_region: Box<Region>,
        else_region: Box<Region>,
        ret: Option<TypeBox>,
    },
}
