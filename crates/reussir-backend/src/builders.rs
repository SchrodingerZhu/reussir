//! Hand-written op builders for cases melior's generated builders don't cover
//! well enough.
//!
//! melior's ODS/dialect builders are preferred everywhere they fit; this module
//! holds the few operations whose generated builder is missing a needed
//! parameter (so the result/attributes can't be set), keeping the raw
//! [`OperationBuilder`] usage contained in the backend rather than leaking into
//! downstream code generators.

use melior::Context;
use melior::ir::attribute::{
    DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::r#type::IntegerType;
use melior::ir::{Identifier, Location, Operation, Type, Value};

use reussir_backend_sys as sys;

/// Set `operation`'s location after it has been built.
///
/// Used to attach a fused debug-info location (carrying a `DBGLocalVar`) to the
/// op that defines a local variable; `mlir-sys` does not bind
/// `mlirOperationSetLocation`, so this goes through the Reussir C API.
pub fn set_location(operation: OperationRef<'_, '_>, location: Location<'_>) {
    unsafe {
        sys::reussirOperationSetLocation(operation.to_raw(), location.to_raw());
    }
}

/// `arith.truncf %value : <from> to <result_type>`.
///
/// melior's [`arith::truncf`](melior::dialect::arith::truncf) is a plain unary
/// with no result-type parameter, so it cannot express a float *narrowing* (the
/// result width must differ from the operand's). This sets the result type
/// explicitly.
pub fn truncf<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("arith.truncf", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid arith.truncf")
}

/// `reussir.trampoline export "<abi>" @<sym_name> = @<target>`.
///
/// The op is attribute-only and its `direction` is a `TrampolineDirection`
/// `I32EnumAttr` for which neither melior nor the Reussir C API exposes a
/// constructor; the enum is stored as a signless `i32` (`Export` is case `1`).
/// Building it here keeps that representational detail out of the code
/// generator.
pub fn trampoline_export<'c>(
    context: &'c Context,
    abi_name: &str,
    sym_name: &str,
    target: &str,
    location: Location<'c>,
) -> Operation<'c> {
    let export = IntegerAttribute::new(IntegerType::new(context, 32).into(), 1).into();
    OperationBuilder::new("reussir.trampoline", location)
        .add_attributes(&[
            (Identifier::new(context, "direction"), export),
            (
                Identifier::new(context, "target"),
                FlatSymbolRefAttribute::new(context, target).into(),
            ),
            (
                Identifier::new(context, "abi_name"),
                StringAttribute::new(context, abi_name).into(),
            ),
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, sym_name).into(),
            ),
        ])
        .build()
        .expect("valid reussir.trampoline")
}

/// `reussir.record.compound (<fields> : <types>) : <result_type>` — construct a
/// compound record value from its (declaration-ordered) field operands.
///
/// melior has no generated builder for the Reussir dialect, so the op is built
/// raw; the result type (the compound record type) must be supplied explicitly.
pub fn record_compound<'c>(
    fields: &[Value<'c, '_>],
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.record.compound", location)
        .add_operands(fields)
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.record.compound")
}

/// `reussir.rc.create value(<value> : <type>) : <result_type>` — box a value into
/// a fresh reference-counted pointer with an initial count of 1.
///
/// The op carries `AttrSizedOperandSegments` over its `[value, token, region]`
/// operand groups, but melior's generated builder leaves the required
/// `operandSegmentSizes` attribute unset, so it is constructed here with the
/// single value operand present (`[1, 0, 0]`). The allocation token and any
/// region are supplied later by the token-instantiation pass; the rc-create
/// fusion pass then folds an immediately preceding `record.compound` into this
/// op (`reussir.rc.create_compound`).
pub fn rc_create<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.rc.create", location)
        .add_operands(&[value])
        .add_attributes(&[(
            Identifier::new(context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(context, &[1, 0, 0]).into(),
        )])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.rc.create")
}

/// `reussir.record.extract (<record> : <type>) [<index>] : <field_type>` —
/// project a field out of a record value.
///
/// The field selector is an `index`-typed `IndexAttr`; the result type is the
/// projected field's type.
pub fn record_extract<'c>(
    context: &'c Context,
    record: Value<'c, '_>,
    index: usize,
    field_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let index_attr = IntegerAttribute::new(Type::index(context), index as i64).into();
    OperationBuilder::new("reussir.record.extract", location)
        .add_operands(&[record])
        .add_attributes(&[(Identifier::new(context, "index"), index_attr)])
        .add_results(&[field_type])
        .build()
        .expect("valid reussir.record.extract")
}
