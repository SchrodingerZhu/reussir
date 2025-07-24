use archery::RcK;
use reussir_core::{
    Location,
    ir::{Block, Operation, OperationKind, OutputValue, ValID},
    path,
    types::{Capability, Primitive, Type, TypeExpr},
};
use rpds::HashTrieMap;
use rustc_hash::{FxHashMapRand, FxRandomState};
use std::{
    cell::{Cell, RefCell},
    collections::hash_map::Entry,
    mem::ManuallyDrop,
};
use ustr::Ustr;

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
    value_types: RefCell<Map<ValID, &'a Type>>,
    named_values: RefCell<Map<Ustr, ValID>>,
    diagnostics: RefCell<Vec<Diagnostic>>,
    primitive_types: RefCell<FxHashMapRand<Primitive, &'a Type>>,
}

impl<'a> IRBuilder<'a> {
    fn snapshot(&self) -> Snapshot<'a> {
        Snapshot {
            value_types: self.value_types.borrow().clone(),
            named_values: self.named_values.borrow().clone(),
        }
    }
    fn recover_to(&self, snapshot: Snapshot<'a>) {
        self.value_types.replace(snapshot.value_types);
        self.named_values.replace(snapshot.named_values);
    }
    fn next_val(&self) -> ValID {
        let val = self.next_val.get();
        self.next_val.set(val + 1);
        val
    }
    pub fn get_primitive_type(&self, primitive: Primitive) -> &'a Type {
        match self.primitive_types.borrow_mut().entry(primitive) {
            Entry::Occupied(entry) => entry.get(),
            Entry::Vacant(entry) => {
                let ty = self.arena.alloc(Type::Atom {
                    capability: Capability::Default,
                    expr: TypeExpr {
                        path: path!(&primitive.to_string()),
                        args: None,
                    },
                });
                entry.insert(ty);
                ty
            }
        }
    }
    pub fn alloc<T: 'a>(&self, value: T) -> &'a T {
        self.arena.alloc(value)
    }
    pub fn alloc_slice_fill_iter<
        T: 'a,
        I: IntoIterator<Item = T, IntoIter = II>,
        II: ExactSizeIterator,
    >(
        &self,
        iter: I,
    ) -> &'a [T] {
        self.arena.alloc_slice_fill_iter(iter)
    }
}
pub struct Snapshot<'a> {
    value_types: Map<ValID, &'a Type>,
    named_values: Map<Ustr, ValID>,
}

pub struct BlockBuilder<'p, 'a: 'p> {
    pub(crate) parent: &'p IRBuilder<'a>,
    snapshot: ManuallyDrop<Snapshot<'a>>,
    operations: RefCell<Vec<Operation<'a>>>,
}

impl<'p, 'a: 'p> std::ops::Deref for BlockBuilder<'p, 'a> {
    type Target = IRBuilder<'a>;

    fn deref(&self) -> &Self::Target {
        self.parent
    }
}

impl<'p, 'a: 'p> BlockBuilder<'p, 'a> {
    pub fn new(parent: &'p IRBuilder<'a>) -> Self {
        let snapshot = parent.snapshot();
        Self {
            parent,
            snapshot: ManuallyDrop::new(snapshot),
            operations: RefCell::new(Vec::new()),
        }
    }

    pub fn build(self) -> Block<'a> {
        Block(
            self.parent
                .arena
                .alloc_slice_fill_iter(self.operations.borrow_mut().drain(..)),
        )
    }

    pub fn new_operation(&self) -> OperationBuilder<'_, 'p, 'a> {
        OperationBuilder {
            parent: self,
            location: None,
            output_value: None,
            output_name: None,
        }
    }
}

pub struct OperationBuilder<'p, 'b: 'p, 'a: 'b> {
    parent: &'p BlockBuilder<'b, 'a>,
    location: Option<Location>,
    output_value: Option<&'a Type>,
    output_name: Option<Ustr>,
}

impl<'p, 'b: 'p, 'a: 'b> OperationBuilder<'p, 'b, 'a> {
    pub fn add_location(&mut self, location: Location) {
        self.location = Some(location);
    }
    pub fn add_output(&mut self, ty: &'a Type, name: Option<Ustr>) {
        self.output_value = Some(ty);
        self.output_name = name;
    }
    pub fn finish(self, kind: OperationKind<'a>) -> Option<ValID> {
        let output = self.output_value.map(|ty| {
            &*self.parent.parent.arena.alloc(OutputValue {
                value: self.parent.next_val(),
                ty,
                name: self.output_name,
            })
        });
        let operation = Operation {
            location: self.location,
            output,
            kind,
        };
        self.parent.operations.borrow_mut().push(operation);
        output.map(|v| v.value)
    }
}

impl Drop for BlockBuilder<'_, '_> {
    fn drop(&mut self) {
        let snapshot = unsafe { ManuallyDrop::take(&mut self.snapshot) };
        self.parent.recover_to(snapshot);
    }
}
