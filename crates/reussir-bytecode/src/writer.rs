//! The bytecode writer: serializes an operation tree into an MLIR bytecode
//! container.
//!
//! This crate targets **bytecode version 4**. At versions below 5, MLIR stores
//! each operation's full attribute dictionary (inherent attributes merged with
//! discardable ones) and reconstructs operation *properties* from it on read, so
//! the writer never has to serialize properties itself. Combined with emitting
//! every attribute and type via the textual fallback encoding, this lets the
//! writer stay completely decoupled from MLIR's per-dialect binary codecs while
//! producing output any compatible `mlir-opt` can load.
//!
//! See [`crate::encoder`] for the primitive byte encodings and the file frame,
//! and [`crate::numbering`] for how the referenced indices are assigned. The
//! section, operation, region, and block grammars are documented on the
//! functions that emit them below.
//!
//! # Version gating
//!
//! The format is versioned; features were added at these versions:
//!
//! ```text
//! 1 DialectVersioning            2 LazyLoading (isolated regions as subsections)
//! 3 UseListOrdering              4 ElideUnknownBlockArgLocation
//! 5 NativePropertiesEncoding     6 NativePropertiesODSSegmentSize (current)
//! ```
//!
//! Targeting [`BYTECODE_VERSION`] = 4 keeps properties merged into the attribute
//! dictionary (no `Properties` section), enables block-argument location
//! elision, and — via LazyLoading — wraps each isolated-from-above operation's
//! regions in their own nested IR section.

use crate::encoder::{Emitter, MAGIC, StringTable, op_mask, section};
use crate::ir::{Block, Op, Region};
use crate::numbering::{Numbering, region_direct_value_count};

/// The bytecode format version this writer emits.
pub const BYTECODE_VERSION: u64 = 4;

/// The producer string embedded in the file header.
const PRODUCER: &str = "reussir-bytecode";

/// Serialize the module operation `root` into a complete bytecode file.
///
/// `root` is normally a `builtin.module` operation; it is written as the single
/// top-level operation of the IR section. This realizes the file frame
/// `file = magic version producer section*` (see [`crate::encoder`]) with the
/// sections in the order MLIR's reader expects:
///
/// ```text
/// Dialect, AttrTypeOffset, AttrType, IR, String
/// ```
///
/// The `Resource*` sections are elided (this crate produces no resources) and
/// the `Properties` section does not exist below version 5. Sections that name
/// strings are built before the String section so the table is complete when it
/// is finally written.
pub fn write_module(root: Op<'_>) -> Vec<u8> {
    let numbering = Numbering::run(root);
    let mut strings = StringTable::new();

    // Sections that reference the string table must be built before the string
    // section itself is finalized, so build their bodies first.
    let dialect_body = build_dialect_section(&numbering, &mut strings);
    let (offset_body, attrtype_body) = build_attr_type_sections(&numbering);
    let ir_body = build_ir_section(&numbering, root);

    let mut out = Emitter::new();
    out.bytes(&MAGIC);
    out.var_int(BYTECODE_VERSION);
    out.str_nul(PRODUCER);

    out.section(section::DIALECT, &dialect_body);
    out.section(section::ATTR_TYPE_OFFSET, &offset_body);
    out.section(section::ATTR_TYPE, &attrtype_body);
    out.section(section::IR, &ir_body);
    // No resources are produced, so the resource sections are elided.
    let mut string_body = Emitter::new();
    strings.write(&mut string_body);
    out.section(section::STRING, &string_body);
    // Properties are not a separate section before bytecode version 5.

    out.into_bytes()
}

/// Build the Dialect section (id 1) body.
///
/// ```text
/// dialect_section = varint(num_dialects) dialect_entry{num_dialects}
///                   varint(num_op_names)            // present since version 4
///                   op_name_group*
/// dialect_entry   = varint_flag(name_string_id, has_version)
/// op_name_group   = varint(dialect_number) varint(group_count) op_name{group_count}
/// op_name         = varint(stripped_name_string_id)   // version < 5: no flag
/// ```
///
/// `has_version` is always 0 here (no DialectVersions subsection). Op names omit
/// their dialect prefix (`func` for `func.func`) and are written in maximal runs
/// sharing a dialect; an op references its name by the flat index assigned across
/// all groups in emission order.
fn build_dialect_section(numbering: &Numbering<'_>, strings: &mut StringTable) -> Emitter {
    let mut e = Emitter::new();

    let dialects = numbering.dialects();
    e.var_int(dialects.len() as u64);
    for &name in dialects {
        let name_id = strings.insert(name) as u64;
        // The low bit signals a dialect version subsection, which we never emit.
        e.var_int_with_flag(name_id, false);
    }

    // Version 4 records the total operation-name count before the groupings.
    let op_names = numbering.op_names();
    e.var_int(op_names.len() as u64);

    // Emit op names in contiguous runs sharing the same dialect.
    let mut i = 0;
    while i < op_names.len() {
        let dialect = op_names[i].dialect;
        let mut j = i + 1;
        while j < op_names.len() && op_names[j].dialect == dialect {
            j += 1;
        }
        e.var_int(numbering.dialect_number(dialect));
        e.var_int((j - i) as u64);
        for entry in &op_names[i..j] {
            // Below version 5 the op name carries no "is registered" flag.
            let id = strings.insert(entry.stripped) as u64;
            e.var_int(id);
        }
        i = j;
    }

    e
}

/// Build the AttrTypeOffset section (id 3) and the AttrType data section (id 2).
///
/// ```text
/// attr_type_offset = varint(num_attrs) varint(num_types)
///                    offset_group*        // attribute groups...
///                    offset_group*        // ...then type groups
/// offset_group     = varint(dialect_number) varint(group_count) offset_entry{group_count}
/// offset_entry     = varint_flag(delta, has_custom_encoding)
///
/// attr_type_data   = entry*               // attribute entries then type entries
/// entry            = ascii_text 0x00      // textual fallback: printed form + NUL
/// ```
///
/// Every attribute and type is written using the textual fallback — its MLIR
/// text plus a nul separator — so `has_custom_encoding` is always 0 and no
/// per-dialect binary codec is needed. `delta` is this entry's end offset minus
/// the previous entry's within the shared data buffer; the counter is continuous
/// across attributes then types. Attributes and types occupy independent index
/// spaces, each numbered by emission order; groups are maximal same-dialect runs.
fn build_attr_type_sections(numbering: &Numbering<'_>) -> (Emitter, Emitter) {
    let mut data = Emitter::new();
    let mut offsets = Emitter::new();

    let attrs = numbering.attrs();
    let types = numbering.types();
    offsets.var_int(attrs.len() as u64);
    offsets.var_int(types.len() as u64);

    // A single running offset spans both attributes and types, since both are
    // written into one data buffer with delta-encoded offsets.
    let mut prev_offset = 0u64;

    // Attributes, grouped by dialect.
    let mut i = 0;
    while i < attrs.len() {
        let dialect = attrs[i].dialect();
        let mut j = i + 1;
        while j < attrs.len() && attrs[j].dialect() == dialect {
            j += 1;
        }
        offsets.var_int(numbering.dialect_number(dialect));
        offsets.var_int((j - i) as u64);
        for attr in &attrs[i..j] {
            data.str_nul(attr.text());
            let cur = data.len() as u64;
            offsets.var_int_with_flag(cur - prev_offset, false);
            prev_offset = cur;
        }
        i = j;
    }

    // Types, grouped by dialect, continuing the same offset counter.
    let mut i = 0;
    while i < types.len() {
        let dialect = types[i].dialect();
        let mut j = i + 1;
        while j < types.len() && types[j].dialect() == dialect {
            j += 1;
        }
        offsets.var_int(numbering.dialect_number(dialect));
        offsets.var_int((j - i) as u64);
        for ty in &types[i..j] {
            data.str_nul(ty.text());
            let cur = data.len() as u64;
            offsets.var_int_with_flag(cur - prev_offset, false);
            prev_offset = cur;
        }
        i = j;
    }

    (offsets, data)
}

/// Build the IR section (id 4): the root operation written as the lone op of a
/// synthetic, argument-less top-level block.
///
/// ```text
/// ir_section = varint_flag(1, false) op       // one op, no block arguments
/// ```
fn build_ir_section<'a>(numbering: &Numbering<'a>, root: Op<'a>) -> Emitter {
    let mut e = Emitter::new();
    // One operation, no block arguments.
    e.var_int_with_flag(1, false);
    write_op(&mut e, numbering, root);
    e
}

/// Encode a single operation.
///
/// ```text
/// op      = varint(op_name_number) mask op_body
/// mask    = byte    // OR of op_mask::* bits, back-patched once components are known
/// op_body = varint(location_attr_number)                      // always present
///           ( varint(attr_dict_number)                  )?    // HasAttrs
///           ( varint(num_results) varint(type_number)*  )?    // HasResults
///           ( varint(num_operands) varint(value_id)*    )?    // HasOperands
///           ( varint(num_successors) varint(block_id)*  )?    // HasSuccessors
///           ( varint_flag(num_regions, isolated) regions )?   // HasInlineRegions
/// regions = ir_subsection   if isolated (a nested IR section, version >= 2)
///         | region*         otherwise
/// ```
///
/// At version 4 the attribute dictionary carries the merged inherent +
/// discardable attributes, so `HasProperties` and the properties field are never
/// emitted. Use-list orders are emitted only when a value's uses are reordered
/// relative to definition order; generated IR never reorders, so that field and
/// its mask bit are absent. `value_id` is the operand's index within its
/// enclosing isolated-from-above value scope (see [`crate::numbering`]).
fn write_op<'a>(e: &mut Emitter, numbering: &Numbering<'a>, op: Op<'a>) {
    e.var_int(numbering.op_name_number(op));

    // The encoding mask is only known after scanning the components, so reserve
    // a byte to patch.
    let mask_offset = e.len();
    e.byte(0);
    let mut mask = 0u8;

    // Location is always present.
    e.var_int(numbering.attr_number(op.loc()));

    // Attribute dictionary (merged inherent + discardable at version 4).
    if !op.attrs().is_empty_dict() {
        mask |= op_mask::HAS_ATTRS;
        e.var_int(numbering.attr_number(op.attrs()));
    }

    // Results.
    if !op.results().is_empty() {
        mask |= op_mask::HAS_RESULTS;
        e.var_int(op.results().len() as u64);
        for result in op.results() {
            e.var_int(numbering.type_number(result.ty()));
        }
    }

    // Operands.
    if !op.operands().is_empty() {
        mask |= op_mask::HAS_OPERANDS;
        e.var_int(op.operands().len() as u64);
        for operand in op.operands() {
            e.var_int(numbering.value_id(operand.id()));
        }
    }

    // Successors.
    if !op.successors().is_empty() {
        mask |= op_mask::HAS_SUCCESSORS;
        e.var_int(op.successors().len() as u64);
        for successor in op.successors() {
            e.var_int(numbering.block_id(successor.id()));
        }
    }

    // Use-list orders (version >= 3) are only emitted when a value's uses are
    // reordered relative to definition order. Generated IR never reorders uses,
    // so nothing is emitted and the mask bit stays clear.

    if !op.regions().is_empty() {
        mask |= op_mask::HAS_INLINE_REGIONS;
    }

    e.patch_byte(mask_offset, mask);

    // Regions are emitted after the mask. Isolated-from-above regions are nested
    // in their own IR subsection (enabling lazy loading on read).
    if !op.regions().is_empty() {
        let isolated = op.is_isolated();
        e.var_int_with_flag(op.regions().len() as u64, isolated);
        if isolated {
            let mut region_emitter = Emitter::new();
            for region in op.regions() {
                write_region(&mut region_emitter, numbering, *region);
            }
            e.section(section::IR, &region_emitter);
        } else {
            for region in op.regions() {
                write_region(e, numbering, *region);
            }
        }
    }
}

/// Encode a region.
///
/// ```text
/// region = varint(0)                                  // empty region
///        | varint(num_blocks) varint(num_values) block{num_blocks}
/// ```
///
/// `num_values` counts the values defined *directly* in the region (block
/// arguments and operation results across its blocks), not those in nested
/// regions; the reader uses it to size the region's slice of the value scope.
fn write_region<'a>(e: &mut Emitter, numbering: &Numbering<'a>, region: Region<'a>) {
    if region.is_empty() {
        e.var_int(0);
        return;
    }
    e.var_int(region.blocks().len() as u64);
    e.var_int(region_direct_value_count(region));
    for block in region.blocks() {
        write_block(e, numbering, *block);
    }
}

/// Encode a block.
///
/// ```text
/// block      = varint_flag(num_ops, has_args) block_args? op{num_ops}
/// block_args = varint(num_args) block_arg{num_args} 0x00   // trailing use-list separator
/// block_arg  = varint_flag(type_number, has_loc) varint(loc_attr_number)?
/// ```
///
/// Version 4 elides the location of unknown-location arguments (`has_loc = 0`,
/// no following location). The trailing `0x00` after the arguments is the
/// (empty) block-argument use-list order, present since version 3. Successors of
/// other blocks reference this block by its index within the parent region.
fn write_block<'a>(e: &mut Emitter, numbering: &Numbering<'a>, block: Block<'a>) {
    let has_args = !block.args().is_empty();
    e.var_int_with_flag(block.ops().len() as u64, has_args);

    if has_args {
        e.var_int(block.args().len() as u64);
        for (arg, loc) in block.args().iter().zip(block.arg_locs()) {
            // Version 4 elides the location of arguments with unknown location.
            let known = !loc.is_unknown_loc();
            e.var_int_with_flag(numbering.type_number(arg.ty()), known);
            if known {
                e.var_int(numbering.attr_number(*loc));
            }
        }
        // Block-argument use-list separator (version >= 3): always a single
        // zero byte when no custom use-list order is present.
        e.byte(0);
    }

    for op in block.ops() {
        write_op(e, numbering, *op);
    }
}
