use archery::RcK;
use ariadne::{Report, sources};
use reussir_core::{
    Context, Location,
    func::{FunctionDatabase, FunctionProto},
    ir::{Block, Operation, OperationKind, OutputValue, ValID},
    module::{FunctionInstance, ModuleInstance},
    path,
    types::{Capability, Primitive, Type, TypeDatabase, TypeExpr},
};
use reussir_front::expr::ExprBox;
use rpds::HashTrieMap;
use rustc_hash::{FxHashMapRand, FxRandomState};
use std::{
    cell::{Cell, RefCell},
    collections::hash_map::Entry,
    mem::ManuallyDrop,
};
use ustr::Ustr;

pub type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Ice,
}

pub struct Diagnostic<'a> {
    level: DiagnosticLevel,
    message: &'a str,
    location: Location,
    nested_labels: Option<&'a [(Location, &'a str)]>,
    backtrace: Option<Box<std::backtrace::Backtrace>>,
}

impl Diagnostic<'_> {
    pub fn create_report(&self) -> Report<'static, Location> {
        let (kind, color) = match self.level {
            DiagnosticLevel::Error => (ariadne::ReportKind::Error, ariadne::Color::Red),
            DiagnosticLevel::Warning => (ariadne::ReportKind::Warning, ariadne::Color::Yellow),
            DiagnosticLevel::Note => (ariadne::ReportKind::Advice, ariadne::Color::Blue),
            DiagnosticLevel::Ice => (ariadne::ReportKind::Advice, ariadne::Color::Magenta),
        };
        let mut builder = Report::build(kind, self.location)
            .with_config(
                ariadne::Config::default()
                    .with_index_type(ariadne::IndexType::Char)
                    .with_color(true),
            )
            .with_message(self.message)
            .with_label(
                ariadne::Label::new(self.location)
                    .with_message("this error occurred here")
                    .with_color(color)
                    .with_order(0),
            );
        for (location, message) in self.nested_labels.unwrap_or_default() {
            builder = builder.with_label(
                ariadne::Label::new(*location)
                    .with_message(*message)
                    .with_color(ariadne::Color::Cyan),
            );
        }
        builder.finish()
    }
}

#[derive(Clone)]
pub struct ValueDef<'a> {
    pub location: Option<Location>,
    pub ty: &'a Type,
}

pub struct IRBuilder<'a> {
    ctx: &'a Context,
    next_val: Cell<ValID>,
    value_types: RefCell<Map<ValID, ValueDef<'a>>>,
    named_values: RefCell<Map<Ustr, ValID>>,
    diagnostics: RefCell<Vec<Diagnostic<'a>>>,
    // TODO: this should be cached globally in context or somewhere.
    primitive_types: RefCell<FxHashMapRand<Primitive, &'a Type>>,
}

pub struct DiagnosticBuilder<'b, 'a: 'b> {
    builder: &'b IRBuilder<'a>,
    level: DiagnosticLevel,
    message: &'a str,
    location: Location,
    nested_labels: Vec<(Location, &'a str)>,
    backtrace: Option<Box<std::backtrace::Backtrace>>,
}

impl<'b, 'a: 'b> DiagnosticBuilder<'b, 'a> {
    pub fn add_nested_label(&mut self, location: Location, message: &'a str) {
        self.nested_labels.push((location, message));
    }
}

impl<'b, 'a: 'b> Drop for DiagnosticBuilder<'b, 'a> {
    fn drop(&mut self) {
        let diagnostic = Diagnostic {
            level: self.level,
            message: self.message,
            location: self.location,
            nested_labels: if self.nested_labels.is_empty() {
                None
            } else {
                Some(
                    self.builder
                        .alloc_slice_fill_iter(self.nested_labels.drain(..)),
                )
            },
            backtrace: self.backtrace.take(),
        };
        self.builder.diagnostics.borrow_mut().push(diagnostic);
    }
}

impl<'a> IRBuilder<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            next_val: Cell::new(0),
            value_types: RefCell::new(Map::default()),
            named_values: RefCell::new(Map::default()),
            diagnostics: RefCell::new(Vec::new()),
            primitive_types: RefCell::new(FxHashMapRand::default()),
        }
    }
    pub fn functions(&self) -> &FunctionDatabase {
        self.ctx.functions()
    }

    pub fn types(&self) -> &TypeDatabase {
        self.ctx.types()
    }

    pub fn report_errors(&self, source: &str, file: Ustr) -> bool {
        let _lock_out = std::io::stdout().lock();
        let _lock_err = std::io::stderr().lock();
        for e in self.diagnostics.borrow().iter() {
            let report = e.create_report();
            report.eprint(sources([(file, source)])).unwrap();
        }
        self.has_errors()
    }
    pub fn has_errors(&self) -> bool {
        !self.diagnostics.borrow().is_empty()
    }
    pub fn diagnostic(
        &self,
        level: DiagnosticLevel,
        message: &'a str,
        location: Location,
    ) -> DiagnosticBuilder<'_, 'a> {
        DiagnosticBuilder {
            builder: self,
            level,
            message,
            location,
            nested_labels: Vec::new(),
            backtrace: if matches!(level, DiagnosticLevel::Ice) {
                Some(Box::new(std::backtrace::Backtrace::capture()))
            } else {
                None
            },
        }
    }
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
    fn add_val(&self, ty: &'a Type, location: Option<Location>, name: Option<Ustr>) {
        let val = self.next_val();
        self.value_types
            .borrow_mut()
            .insert_mut(val, ValueDef { location, ty });
        if let Some(name) = name {
            self.bind(name, val);
        }
    }
    fn next_val(&self) -> ValID {
        let val = self.next_val.get();
        self.next_val.set(val + 1);
        val
    }
    pub fn bind(&self, name: Ustr, value: ValID) {
        self.named_values.borrow_mut().insert_mut(name, value);
    }
    pub fn lookup(&self, name: &Ustr) -> Option<(ValID, ValueDef<'a>)> {
        let value = self.named_values.borrow().get(name).copied()?;
        let def = self.value_types.borrow().get(&value).cloned()?;
        Some((value, def))
    }
    pub fn get_primitive_type(&self, primitive: Primitive) -> &'a Type {
        match self.primitive_types.borrow_mut().entry(primitive) {
            Entry::Occupied(entry) => entry.get(),
            Entry::Vacant(entry) => {
                let ty = self.ctx.bump().alloc(Type::Atom {
                    capability: Capability::Default,
                    expr: TypeExpr {
                        path: path!(Into::<&'static str>::into(primitive)),
                        args: None,
                    },
                });
                entry.insert(ty);
                ty
            }
        }
    }
    pub fn alloc<T: 'a>(&self, value: T) -> &'a T {
        self.ctx.bump().alloc(value)
    }
    pub fn alloc_slice_fill_iter<
        T: 'a,
        I: IntoIterator<Item = T, IntoIter = II>,
        II: ExactSizeIterator,
    >(
        &self,
        iter: I,
    ) -> &'a [T] {
        self.ctx.bump().alloc_slice_fill_iter(iter)
    }
    pub fn ctx(&self) -> &'a Context {
        self.ctx
    }
}
pub struct Snapshot<'a> {
    value_types: Map<ValID, ValueDef<'a>>,
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
                .ctx
                .bump()
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
            &*self.parent.parent.ctx.bump().alloc(OutputValue {
                value: self.parent.next_val(),
                ty,
                name: self.output_name,
            })
        });
        if let Some(output) = &output {
            self.parent.parent.value_types.borrow_mut().insert_mut(
                output.value,
                ValueDef {
                    location: self.location,
                    ty: output.ty,
                },
            );
            if let Some(name) = output.name {
                self.parent.parent.bind(name, output.value);
            }
        }
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

pub struct ModuleBuilder<'a> {
    ir_builder: IRBuilder<'a>,
    pub functions: Vec<FunctionInstance<'a>>,
}

impl<'a> ModuleBuilder<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            ir_builder: IRBuilder::new(ctx),
            functions: Vec::new(),
        }
    }

    pub fn report_errors(&self, source: &str, file: Ustr) -> bool {
        self.ir_builder.report_errors(source, file)
    }

    pub fn define_function(&mut self, proto: &FunctionProto, body: &ExprBox) {
        let snapshot = self.ir_builder.snapshot();
        for param in proto.params.iter() {
            let ty = self.ir_builder.alloc(param.ty.clone());
            self.ir_builder
                .add_val(ty, Some(param.location), param.name);
        }
        let block_builder = BlockBuilder::new(&self.ir_builder);
        block_builder.add_expr(body, false, None);
        let blk = block_builder.build();
        let body = self.ir_builder.ctx.bump().alloc(blk);
        let function_instance = FunctionInstance {
            path: proto.path.clone(),
            type_params: None,
            body,
        };
        self.functions.push(function_instance);
        self.ir_builder.recover_to(snapshot);
    }

    pub fn build(self) -> ModuleInstance<'a> {
        let functions = self.ir_builder.alloc_slice_fill_iter(self.functions);
        ModuleInstance {
            ctx: self.ir_builder.ctx,
            functions,
        }
    }
}
