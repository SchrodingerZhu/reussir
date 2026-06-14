//! The structural IR: SSA values, operations, blocks, and regions.
//!
//! These nodes form the operation tree that the bytecode writer serializes.
//! Every node is arena-allocated and handed out as a `Copy` handle. Identity is
//! by pointer, which is what the numbering pass keys on when assigning the value
//! and block indices the binary format references.
//!
//! The design mirrors MLIR's own structure (operations own regions, regions own
//! blocks, blocks own operations and block arguments) so that a frontend can
//! build exactly the IR it would have built textually, then emit it as bytecode.

use crate::context::{Attr, Context, Type};

/// An SSA value: either a block argument or an operation result. Carries its
/// type; its bytecode index is assigned later by the numbering pass.
pub struct ValueData<'a> {
    ty: Type<'a>,
}

/// A copyable handle to a [`ValueData`].
#[derive(Clone, Copy)]
pub struct Value<'a>(&'a ValueData<'a>);

impl<'a> Value<'a> {
    /// The type of this value.
    pub fn ty(self) -> Type<'a> {
        self.0.ty
    }

    /// A stable identity key for use in numbering tables.
    pub fn id(self) -> usize {
        self.0 as *const ValueData<'a> as usize
    }
}

impl PartialEq for Value<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}
impl Eq for Value<'_> {}

/// A basic block: typed arguments (each with a source location) and a list of
/// operations terminated by a terminator op.
pub struct BlockData<'a> {
    args: &'a [Value<'a>],
    arg_locs: &'a [Attr<'a>],
    ops: &'a [Op<'a>],
}

/// A copyable handle to a [`BlockData`].
#[derive(Clone, Copy)]
pub struct Block<'a>(&'a BlockData<'a>);

impl<'a> Block<'a> {
    /// The block's arguments.
    pub fn args(self) -> &'a [Value<'a>] {
        self.0.args
    }

    /// The source location of each block argument (parallel to [`Block::args`]).
    pub fn arg_locs(self) -> &'a [Attr<'a>] {
        self.0.arg_locs
    }

    /// The operations in the block.
    pub fn ops(self) -> &'a [Op<'a>] {
        self.0.ops
    }

    /// A stable identity key for use in numbering tables.
    pub fn id(self) -> usize {
        self.0 as *const BlockData<'a> as usize
    }
}

/// A region: an ordered list of blocks. The first block is the entry block.
#[derive(Clone, Copy)]
pub struct Region<'a> {
    blocks: &'a [Block<'a>],
}

impl<'a> Region<'a> {
    /// The blocks of this region.
    pub fn blocks(self) -> &'a [Block<'a>] {
        self.blocks
    }

    /// Whether the region has no blocks.
    pub fn is_empty(self) -> bool {
        self.blocks.is_empty()
    }
}

/// An operation: the unit of the IR tree.
pub struct OpData<'a> {
    name: &'a str,
    dialect: &'a str,
    loc: Attr<'a>,
    attrs: Attr<'a>,
    operands: &'a [Value<'a>],
    results: &'a [Value<'a>],
    successors: &'a [Block<'a>],
    regions: &'a [Region<'a>],
    isolated: bool,
}

/// A copyable handle to an [`OpData`].
#[derive(Clone, Copy)]
pub struct Op<'a>(&'a OpData<'a>);

impl<'a> Op<'a> {
    /// The fully qualified operation name, e.g. `func.func`.
    pub fn name(self) -> &'a str {
        self.0.name
    }

    /// The namespace of the dialect that defines this op (text before the first
    /// dot of the name).
    pub fn dialect(self) -> &'a str {
        self.0.dialect
    }

    /// The operation's source location attribute.
    pub fn loc(self) -> Attr<'a> {
        self.0.loc
    }

    /// The operation's attribute dictionary (possibly empty).
    pub fn attrs(self) -> Attr<'a> {
        self.0.attrs
    }

    /// The operation's operands.
    pub fn operands(self) -> &'a [Value<'a>] {
        self.0.operands
    }

    /// The values defined by this operation.
    pub fn results(self) -> &'a [Value<'a>] {
        self.0.results
    }

    /// The operation's successor blocks (for terminators).
    pub fn successors(self) -> &'a [Block<'a>] {
        self.0.successors
    }

    /// The operation's regions.
    pub fn regions(self) -> &'a [Region<'a>] {
        self.0.regions
    }

    /// Whether this operation's regions are isolated from above (so value
    /// numbering restarts inside them and they are serialized in their own IR
    /// subsection).
    pub fn is_isolated(self) -> bool {
        self.0.isolated
    }

    /// A stable identity key for use in numbering tables.
    pub fn id(self) -> usize {
        self.0 as *const OpData<'a> as usize
    }
}

impl<'a> Context<'a> {
    /// Create a fresh SSA value of the given type.
    pub fn value(&self, ty: Type<'a>) -> Value<'a> {
        Value(self.bump().alloc(ValueData { ty }))
    }

    /// Assemble a block from arguments (each paired with a location) and ops.
    pub fn block(&self, args: &[Value<'a>], arg_locs: &[Attr<'a>], ops: &[Op<'a>]) -> Block<'a> {
        assert_eq!(
            args.len(),
            arg_locs.len(),
            "each block arg needs a location"
        );
        let data = self.bump().alloc(BlockData {
            args: self.bump().alloc_slice_copy(args),
            arg_locs: self.bump().alloc_slice_copy(arg_locs),
            ops: self.bump().alloc_slice_copy(ops),
        });
        Block(data)
    }

    /// Assemble a block whose arguments all share the unknown location.
    pub fn block_unknown_locs(&self, args: &[Value<'a>], ops: &[Op<'a>]) -> Block<'a> {
        let unknown = self.loc_unknown();
        let locs: Vec<Attr<'a>> = vec![unknown; args.len()];
        self.block(args, &locs, ops)
    }

    /// Assemble a region from its blocks.
    pub fn region(&self, blocks: &[Block<'a>]) -> Region<'a> {
        Region {
            blocks: self.bump().alloc_slice_copy(blocks),
        }
    }

    /// Start building an operation named `name` (e.g. `arith.addi`).
    pub fn op<'c>(&'c self, name: &str) -> OpBuilder<'c, 'a> {
        let name_ref = self.bump().alloc_str(name);
        let dialect_end = name_ref.find('.').unwrap_or(name_ref.len());
        OpBuilder {
            ctx: self,
            name: name_ref,
            dialect: &name_ref[..dialect_end],
            loc: self.loc_unknown(),
            attrs: self.empty_dict(),
            operands: Vec::new(),
            result_types: Vec::new(),
            successors: Vec::new(),
            regions: Vec::new(),
            isolated: false,
        }
    }
}

/// A fluent builder for an [`Op`].
///
/// Set operands, result types, attributes, regions and successors, then call
/// [`OpBuilder::build`]. The builder allocates a fresh [`Value`] for each result
/// type and returns the operation together with those result values, so callers
/// can wire them into later operations.
pub struct OpBuilder<'c, 'a> {
    ctx: &'c Context<'a>,
    name: &'a str,
    dialect: &'a str,
    loc: Attr<'a>,
    attrs: Attr<'a>,
    operands: Vec<Value<'a>>,
    result_types: Vec<Type<'a>>,
    successors: Vec<Block<'a>>,
    regions: Vec<Region<'a>>,
    isolated: bool,
}

impl<'c, 'a> OpBuilder<'c, 'a> {
    /// Set the operation's source location.
    pub fn loc(mut self, loc: Attr<'a>) -> Self {
        self.loc = loc;
        self
    }

    /// Set the operation's attribute dictionary.
    pub fn attrs(mut self, attrs: Attr<'a>) -> Self {
        self.attrs = attrs;
        self
    }

    /// Append an operand.
    pub fn operand(mut self, v: Value<'a>) -> Self {
        self.operands.push(v);
        self
    }

    /// Append several operands.
    pub fn operands(mut self, vs: &[Value<'a>]) -> Self {
        self.operands.extend_from_slice(vs);
        self
    }

    /// Append a result of the given type.
    pub fn result(mut self, ty: Type<'a>) -> Self {
        self.result_types.push(ty);
        self
    }

    /// Append several results.
    pub fn results(mut self, tys: &[Type<'a>]) -> Self {
        self.result_types.extend_from_slice(tys);
        self
    }

    /// Append a successor block.
    pub fn successor(mut self, b: Block<'a>) -> Self {
        self.successors.push(b);
        self
    }

    /// Append a region.
    pub fn region(mut self, r: Region<'a>) -> Self {
        self.regions.push(r);
        self
    }

    /// Mark the operation's regions as isolated from above.
    pub fn isolated(mut self, isolated: bool) -> Self {
        self.isolated = isolated;
        self
    }

    /// Finish building. Returns the operation and a freshly-allocated value for
    /// each result type, in order.
    pub fn build(self) -> (Op<'a>, &'a [Value<'a>]) {
        let bump = self.ctx.bump();
        let results: Vec<Value<'a>> = self
            .result_types
            .iter()
            .map(|&ty| self.ctx.value(ty))
            .collect();
        let results_ref = bump.alloc_slice_copy(&results);
        let data = bump.alloc(OpData {
            name: self.name,
            dialect: self.dialect,
            loc: self.loc,
            attrs: self.attrs,
            operands: bump.alloc_slice_copy(&self.operands),
            results: results_ref,
            successors: bump.alloc_slice_copy(&self.successors),
            regions: bump.alloc_slice_copy(&self.regions),
            isolated: self.isolated,
        });
        (Op(data), results_ref)
    }

    /// Finish building an operation that defines no values.
    pub fn build_zero(self) -> Op<'a> {
        debug_assert!(
            self.result_types.is_empty(),
            "use build() for ops with results"
        );
        self.build().0
    }

    /// Finish building an operation with exactly one result, returning the op
    /// and that result value.
    pub fn build_one(self) -> (Op<'a>, Value<'a>) {
        let (op, results) = self.build();
        debug_assert_eq!(results.len(), 1, "build_one expects exactly one result");
        (op, results[0])
    }
}
