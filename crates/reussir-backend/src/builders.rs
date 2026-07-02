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
use melior::ir::operation::{OperationBuilder, OperationRef};
use melior::ir::r#type::IntegerType;
use melior::ir::{Identifier, Location, Operation, Region, Type, Value};

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

/// `reussir.rc.create value(<value>) region(<region>) : <result_type>` — box a
/// value into a fresh region-allocated reference-counted pointer with `flex`
/// capability and an initial count of 1.
///
/// Like [`rc_create`] the op carries `AttrSizedOperandSegments` over its
/// `[value, token, region]` operand groups; here the value and region are
/// present and the token absent (`[1, 0, 1]`). Supplying the region is what
/// gives the result `flex` (region-local, mutable) capability — the region
/// patterns pass later attaches the box vtable and freezes the value to `rigid`
/// where it escapes its region.
pub fn rc_create_in_region<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    region: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.rc.create", location)
        .add_operands(&[value, region])
        .add_attributes(&[(
            Identifier::new(context, "operandSegmentSizes"),
            // operandSegmentSizes over [value, token, region]: value and region
            // present, token absent.
            DenseI32ArrayAttribute::new(context, &[1, 0, 1]).into(),
        )])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.rc.create with region")
}

/// `reussir.record.variant [<tag>] (<payload> : <payload_type>) : <variant_type>`
/// — wrap a case payload into a variant (enum) record value under its tag.
///
/// melior has no generated builder for the Reussir dialect, so the op is built
/// raw; the `tag` selector is an `index`-typed `IndexAttr` and the result is the
/// enum's variant record type. The payload is the `{enum}::{case}` compound
/// produced by a preceding `reussir.record.compound`; the rc-create fusion pass
/// later folds `variant` + `rc.create` into `reussir.rc.create_variant`.
pub fn record_variant<'c>(
    context: &'c Context,
    tag: usize,
    payload: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let tag_attr = IntegerAttribute::new(Type::index(context), tag as i64).into();
    OperationBuilder::new("reussir.record.variant", location)
        .add_operands(&[payload])
        .add_attributes(&[(Identifier::new(context, "tag"), tag_attr)])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.record.variant")
}

/// `reussir.nullable.create (<value> : <type>)? : <result_type>` — wrap a
/// non-null pointer into a nullable pointer, or build a null pointer when no
/// value is given.
///
/// The op's `$ptr` operand is `Optional`, and melior's generated builder is
/// `(context, result_type, location)` — it exposes no parameter for that operand,
/// so it can only build the null pointer. The value-wrapping (non-null) case is
/// built raw here, adding the single pointer operand.
pub fn nullable_create<'c>(
    value: Option<Value<'c, '_>>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.nullable.create", location);
    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }
    builder
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.nullable.create")
}

/// `reussir.region.run (-> <result_type>)? { <body> }` — execute a region scope.
///
/// The body region's single block takes a `!reussir.region` argument (the arena
/// handle) and terminates with `reussir.region.yield`. A `flex` value it yields
/// is frozen to `rigid` as it leaves the scope. melior's generated builder takes
/// only `(context, region, location)` — it exposes no result type for the op's
/// `Optional` result — so a value-yielding `region.run` is built raw here;
/// `result_type` is `None` for a region whose body yields nothing.
pub fn region_run<'c>(
    result_type: Option<Type<'c>>,
    body: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.region.run", location);
    if let Some(result_type) = result_type {
        builder = builder.add_results(&[result_type]);
    }
    builder
        .add_regions([body])
        .build()
        .expect("valid reussir.region.run")
}

/// `reussir.region.yield (<value> : <type>)?` — terminate a `reussir.region.run`
/// body, optionally yielding a value out of the region.
///
/// melior's generated builder takes only `(context, location)` — it exposes no
/// parameter for the op's `Optional` value operand — so the op is built raw here
/// (the value is absent for a unit-typed region).
pub fn region_yield<'c>(value: Option<Value<'c, '_>>, location: Location<'c>) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.region.yield", location);
    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }
    builder.build().expect("valid reussir.region.yield")
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
