use std::{
    cell::{Cell, RefCell},
    mem::ManuallyDrop,
};

use archery::RcK;
use indexmap::IndexMap;
use rpds::HashTrieMap;
use rustc_hash::FxRandomState;
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
    ConstInt(&'a IntegerLiteral),
    ConstFloat(&'a FloatLiteral),
}

pub type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;

pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
}

pub struct Diagnostic {
    level: DiagnosticLevel,
    message: String,
    location: Option<Location>,
    nested_labels: Option<Box<[(Location, String)]>>,
}

pub struct IRBuilder<'a> {
    arena: &'a bumpalo::Bump,
    next_val: Cell<ValID>,
    value_types: RefCell<IndexMap<ValID, &'a Type>>,
    named_values: RefCell<Map<Ustr, ValID>>,
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl<'a> IRBuilder<'a> {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            value_types: self.value_types.borrow().len(),
            named_values: self.named_values.borrow().clone(),
        }
    }
    fn recover_to(&self, snapshot: Snapshot) {
        self.value_types.borrow_mut().truncate(snapshot.value_types);
        self.named_values.replace(snapshot.named_values);
    }
}

pub struct Snapshot {
    value_types: usize,
    named_values: Map<Ustr, ValID>,
}

pub struct BlockBuilder<'p, 'a: 'p> {
    parent: &'p IRBuilder<'a>,
    snapshot: ManuallyDrop<Snapshot>,
    operations: Vec<Operation<'a>>,
    location: Option<Location>,
}

impl<'p, 'a: 'p> BlockBuilder<'p, 'a> {
    pub fn new(parent: &'p IRBuilder<'a>) -> Self {
        let snapshot = parent.snapshot();
        Self {
            parent,
            snapshot: ManuallyDrop::new(snapshot),
            operations: Vec::new(),
            location: None,
        }
    }

    pub fn set_location(&mut self, location: Location) {
        self.location = Some(location);
    }

    pub fn add_operation(&mut self, operation: Operation<'a>) {
        self.operations.push(operation);
    }

    pub fn build(mut self) -> Block<'a> {
        Block(
            self.parent
                .arena
                .alloc_slice_fill_iter(self.operations.drain(..)),
        )
    }
}

impl Drop for BlockBuilder<'_, '_> {
    fn drop(&mut self) {
        let snapshot = unsafe { ManuallyDrop::take(&mut self.snapshot) };
        self.parent.recover_to(snapshot);
    }
}
