//! The numbering pass: assigns the integer indices the bytecode format uses to
//! reference dialects, operation names, attributes, types, SSA values, and
//! blocks.
//!
//! The scheme reproduces MLIR's `IRNumberingState` for the entities a from
//! scratch writer must agree with the reader on:
//!
//! * **Attributes / types / op names** are referenced by an index equal to
//!   their emission order. The reader rebuilds an indexed table in the same
//!   order, so any internally-consistent ordering is valid; we use first-seen
//!   order. Attributes and types occupy independent index spaces.
//! * **Values** are numbered depth-first. Within a region, block arguments then
//!   operation results are numbered in order; a region's child regions start
//!   numbering just past the parent region's own values, and an isolated-from
//!   above operation restarts at zero. The reader treats each isolated scope's
//!   value table as a stack segmented per region, which is exactly what this
//!   produces.
//! * **Blocks** are numbered by their index within their parent region, which
//!   is what successor references use.
//!
//! Dialects are numbered in first-seen order; their only requirement is that
//! every dialect referenced by an op, attribute, or type appears in the table.

use rustc_hash::FxHashMap;

use crate::context::{Attr, Type};
use crate::ir::{Op, Region};

/// One entry in the operation-name table.
pub struct OpNameEntry<'a> {
    /// Dialect namespace (e.g. `func`).
    pub dialect: &'a str,
    /// Operation name with the dialect prefix stripped (e.g. `func` for
    /// `func.func`, `rc.inc` for `reussir.rc.inc`).
    pub stripped: &'a str,
}

/// The result of numbering an operation tree.
pub struct Numbering<'a> {
    dialects: Vec<&'a str>,
    dialect_index: FxHashMap<&'a str, usize>,

    op_names: Vec<OpNameEntry<'a>>,
    op_name_index: FxHashMap<&'a str, usize>,

    attrs: Vec<Attr<'a>>,
    attr_index: FxHashMap<Attr<'a>, usize>,

    types: Vec<Type<'a>>,
    type_index: FxHashMap<Type<'a>, usize>,

    value_ids: FxHashMap<usize, u64>,
    block_ids: FxHashMap<usize, u64>,
}

impl<'a> Numbering<'a> {
    /// Number the operation tree rooted at `root` (typically the module op).
    pub fn run(root: Op<'a>) -> Numbering<'a> {
        let mut n = Numbering {
            dialects: Vec::new(),
            dialect_index: FxHashMap::default(),
            op_names: Vec::new(),
            op_name_index: FxHashMap::default(),
            attrs: Vec::new(),
            attr_index: FxHashMap::default(),
            types: Vec::new(),
            type_index: FxHashMap::default(),
            value_ids: FxHashMap::default(),
            block_ids: FxHashMap::default(),
        };
        n.collect_op(root);
        // Value numbering starts at the root op's regions; the root (module) is
        // isolated, so its body begins at value id 0.
        for region in root.regions() {
            n.number_region(*region, 0);
        }
        n
    }

    // ----- Lookups used by the writer ---------------------------------------

    /// The numbered dialects, in index order.
    pub fn dialects(&self) -> &[&'a str] {
        &self.dialects
    }

    /// The numbered operation names, in index order.
    pub fn op_names(&self) -> &[OpNameEntry<'a>] {
        &self.op_names
    }

    /// The numbered attributes, in index order.
    pub fn attrs(&self) -> &[Attr<'a>] {
        &self.attrs
    }

    /// The numbered types, in index order.
    pub fn types(&self) -> &[Type<'a>] {
        &self.types
    }

    /// The index of a dialect namespace.
    pub fn dialect_number(&self, ns: &str) -> u64 {
        self.dialect_index[ns] as u64
    }

    /// The index of an operation's name in the op-name table.
    pub fn op_name_number(&self, op: Op<'a>) -> u64 {
        self.op_name_index[op.name()] as u64
    }

    /// The index of an attribute in the attribute table.
    pub fn attr_number(&self, attr: Attr<'a>) -> u64 {
        self.attr_index[&attr] as u64
    }

    /// The index of a type in the type table.
    pub fn type_number(&self, ty: Type<'a>) -> u64 {
        self.type_index[&ty] as u64
    }

    /// The numbering of an SSA value within its isolated scope.
    pub fn value_id(&self, id: usize) -> u64 {
        self.value_ids[&id]
    }

    /// The index of a block within its parent region.
    pub fn block_id(&self, id: usize) -> u64 {
        self.block_ids[&id]
    }

    // ----- Collection --------------------------------------------------------

    fn intern_dialect(&mut self, ns: &'a str) {
        if !self.dialect_index.contains_key(ns) {
            let idx = self.dialects.len();
            self.dialects.push(ns);
            self.dialect_index.insert(ns, idx);
        }
    }

    fn intern_attr(&mut self, attr: Attr<'a>) {
        self.intern_dialect(attr.dialect());
        if !self.attr_index.contains_key(&attr) {
            let idx = self.attrs.len();
            self.attrs.push(attr);
            self.attr_index.insert(attr, idx);
        }
    }

    fn intern_type(&mut self, ty: Type<'a>) {
        // The unit type is never materialized in the bytecode.
        if ty.is_unit() {
            return;
        }
        self.intern_dialect(ty.dialect());
        if !self.type_index.contains_key(&ty) {
            let idx = self.types.len();
            self.types.push(ty);
            self.type_index.insert(ty, idx);
        }
    }

    fn intern_op_name(&mut self, op: Op<'a>) {
        if self.op_name_index.contains_key(op.name()) {
            return;
        }
        let dialect = op.dialect();
        // Strip the dialect prefix and the dot separating it from the op name.
        let stripped = &op.name()[(dialect.len() + 1).min(op.name().len())..];
        let idx = self.op_names.len();
        self.op_names.push(OpNameEntry { dialect, stripped });
        self.op_name_index.insert(op.name(), idx);
    }

    fn collect_op(&mut self, op: Op<'a>) {
        self.intern_dialect(op.dialect());
        self.intern_op_name(op);
        self.intern_attr(op.loc());
        if !op.attrs().is_empty_dict() {
            self.intern_attr(op.attrs());
        }
        for result in op.results() {
            self.intern_type(result.ty());
        }
        for region in op.regions() {
            for block in region.blocks() {
                for (arg, loc) in block.args().iter().zip(block.arg_locs()) {
                    self.intern_type(arg.ty());
                    self.intern_attr(*loc);
                }
                for inner in block.ops() {
                    self.collect_op(*inner);
                }
            }
        }
    }

    // ----- Value and block numbering ----------------------------------------

    fn number_region(&mut self, region: Region<'a>, base: u64) {
        if region.is_empty() {
            return;
        }
        let mut id = base;
        for (block_index, block) in region.blocks().iter().enumerate() {
            self.block_ids.insert(block.id(), block_index as u64);
            for arg in block.args() {
                self.value_ids.insert(arg.id(), id);
                id += 1;
            }
            for op in block.ops() {
                for result in op.results() {
                    self.value_ids.insert(result.id(), id);
                    id += 1;
                }
            }
        }
        // Child regions begin numbering just past this region's own values.
        let child_base = id;
        for block in region.blocks() {
            for op in block.ops() {
                if op.regions().is_empty() {
                    continue;
                }
                let cbase = if op.is_isolated() { 0 } else { child_base };
                for region in op.regions() {
                    self.number_region(*region, cbase);
                }
            }
        }
    }
}

/// Count the values defined directly in a region (block arguments and operation
/// results across all its blocks, not descending into nested regions).
pub fn region_direct_value_count(region: Region<'_>) -> u64 {
    let mut count = 0u64;
    for block in region.blocks() {
        count += block.args().len() as u64;
        for op in block.ops() {
            count += op.results().len() as u64;
        }
    }
    count
}
