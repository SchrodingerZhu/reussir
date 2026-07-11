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
    ArrayAttribute, Attribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute,
    FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
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

/// `reussir.str.global @<sym_name> = "payload"`.
pub fn str_global<'c>(
    context: &'c Context,
    sym_name: &str,
    payload: &str,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.str.global", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, sym_name).into(),
            ),
            (
                Identifier::new(context, "payload"),
                StringAttribute::new(context, payload).into(),
            ),
        ])
        .build()
        .expect("valid reussir.str.global")
}

/// `reussir.str.literal @<sym_name> : <result_type>`.
pub fn str_literal<'c>(
    context: &'c Context,
    sym_name: &str,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.str.literal", location)
        .add_attributes(&[(
            Identifier::new(context, "sym_name"),
            FlatSymbolRefAttribute::new(context, sym_name).into(),
        )])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.str.literal")
}

/// `reussir.str.cast (<global>) : <result_type>`.
pub fn str_cast<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.str.cast", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.str.cast")
}

/// `reussir.str.select (<str>) ["..."] : (...) -> (index, i1)`.
pub fn str_select<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    patterns: &[&str],
    location: Location<'c>,
) -> Operation<'c> {
    let attrs: Vec<Attribute<'c>> = patterns
        .iter()
        .map(|p| StringAttribute::new(context, p).into())
        .collect();
    OperationBuilder::new("reussir.str.select", location)
        .add_operands(&[value])
        .add_attributes(&[(
            Identifier::new(context, "patterns"),
            ArrayAttribute::new(context, &attrs).into(),
        )])
        .add_results(&[Type::index(context), IntegerType::new(context, 1).into()])
        .build()
        .expect("valid reussir.str.select")
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

/// `reussir.nullable.dispatch (%v : nullable<…>) (-> <result_type>)?` — the
/// two-region null dispatch: the first (non-null) region's block takes the
/// unwrapped pointer as its argument, the second (null) region's block takes
/// none; both terminate with `reussir.scf.yield` ([`scf_yield`]).
///
/// Built raw like [`region_run`]: the op's result is `Optional`, which
/// melior's generated builder cannot express (`nullable.check`/`coerce` have
/// required results and use their generated builders directly).
pub fn nullable_dispatch<'c>(
    nullable: Value<'c, '_>,
    result_type: Option<Type<'c>>,
    non_null: Region<'c>,
    null: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.nullable.dispatch", location)
        .add_operands(&[nullable])
        .add_regions_vec(vec![non_null, null]);
    if let Some(result_type) = result_type {
        builder = builder.add_results(&[result_type]);
    }
    builder.build().expect("valid reussir.nullable.dispatch")
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

/// `reussir.closure.create -> <result_type> { body … }` — create a closure from
/// an inlined body region.
///
/// The result is a shared `!reussir.rc<closure<(inputs) -> output>>`; the body
/// region's single block takes the closure input types as arguments (captures
/// followed by parameters) and terminates with `reussir.closure.yield`. This
/// builds the *inlined* form — no allocation token (the token-instantiation pass
/// supplies it) and no outlined vtable (the closure-outlining pass generates the
/// evaluate/drop/clone functions and vtable from the body). The op has a custom
/// assembly format and no generated builder, so it is built raw here.
pub fn closure_create<'c>(
    result_type: Type<'c>,
    body: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.closure.create", location)
        .add_regions([body])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.closure.create")
}

/// `reussir.closure.yield (<value> : <type>)?` — terminate a
/// `reussir.closure.create` body, yielding the closure's result (absent for a
/// closure with no output). Built raw like [`region_yield`] — the generated
/// builder exposes no parameter for the `Optional` value operand.
pub fn closure_yield<'c>(value: Option<Value<'c, '_>>, location: Location<'c>) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.closure.yield", location);
    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }
    builder.build().expect("valid reussir.closure.yield")
}

/// `reussir.closure.apply (<arg> : <type>) to (<closure> : <type>) : <result_type>`
/// — supply one leading argument to a closure, yielding a closure with one fewer
/// input. Built raw here; the result type is the residual closure `rc` type.
pub fn closure_apply<'c>(
    arg: Value<'c, '_>,
    closure: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.closure.apply", location)
        .add_operands(&[arg, closure])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.closure.apply")
}

/// `reussir.closure.uniqify (<closure> : <type>) : <result_type>` — obtain a
/// uniquely-owned copy of a closure.
///
/// It expands (in the SCF-lowering pass) to an `rc.is_unique` guarded `scf.if`:
/// a uniquely-referenced closure is returned as-is, otherwise it is deep-cloned
/// (and the shared reference dropped). It must precede any `reussir.closure.apply`
/// on a closure that may be shared, because `apply` writes the argument into the
/// closure box *in place* — mutating a shared closure would corrupt every other
/// holder. The result type is the same closure `rc` type as the input.
pub fn closure_uniqify<'c>(
    closure: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.closure.uniqify", location)
        .add_operands(&[closure])
        .add_results(&[result_type])
        .build()
        .expect("valid reussir.closure.uniqify")
}

/// `reussir.closure.eval (<closure> : <type>) (: <result_type>)?` — evaluate a
/// fully-applied closure, producing its result (absent for a closure with no
/// output). Built raw here; `result_type` is `None` for a unit-returning closure.
pub fn closure_eval<'c>(
    closure: Value<'c, '_>,
    result_type: Option<Type<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.closure.eval", location);
    builder = builder.add_operands(&[closure]);
    if let Some(result_type) = result_type {
        builder = builder.add_results(&[result_type]);
    }
    builder.build().expect("valid reussir.closure.eval")
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

/// `reussir.record.dispatch (<variant> : <ref>) (-> <result>)? { [t] -> {…} … }`
/// — dispatch on the tag of the variant record behind `variant`, one region per
/// entry of `tags_per_region` (region `i` handles the single tag
/// `tags_per_region[i]`, whose block takes a `!reussir.ref` to that case's
/// payload compound).
///
/// The `tagSets` attribute is an array of `DenseI64ArrayAttr`s (one tag set per
/// region), matching the dialect's expansion to `scf.index_switch`. Each region
/// must terminate with [`scf_yield`]. Built raw because the op's result is
/// `Optional`: melior's generated builder takes the regions and the `tagSets`
/// attribute but exposes no result-type parameter, so it can only build a void
/// dispatch — a value-producing `match` needs the result set here.
pub fn record_dispatch<'c>(
    context: &'c Context,
    variant: Value<'c, '_>,
    tags_per_region: &[i64],
    result_type: Option<Type<'c>>,
    regions: Vec<Region<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let tag_sets: Vec<_> = tags_per_region
        .iter()
        .map(|&t| DenseI64ArrayAttribute::new(context, &[t]).into())
        .collect();
    let tag_sets_attr = ArrayAttribute::new(context, &tag_sets).into();
    let mut builder = OperationBuilder::new("reussir.record.dispatch", location)
        .add_operands(&[variant])
        .add_attributes(&[(Identifier::new(context, "tagSets"), tag_sets_attr)])
        .add_regions_vec(regions);
    if let Some(result_type) = result_type {
        builder = builder.add_results(&[result_type]);
    }
    builder.build().expect("valid reussir.record.dispatch")
}

/// `reussir.scf.yield (<value> : <type>)?` — terminate a `reussir.record.dispatch`
/// or `reussir.nullable.dispatch` arm, optionally yielding the dispatch's result.
/// Built raw like [`region_yield`]: the op's `$value` operand is `Optional`, and
/// melior's generated builder takes only `(context, location)` — it exposes no
/// parameter for it, so it can build only a value-less yield. The value is absent
/// for a void dispatch.
/// `reussir.array.with_unique_view (%array : rc) -> rc { ^bb(%view: memref): … }`
/// — uniqify (clone-if-shared) an rc array and run `body` with a mutable view
/// of its payload. With an empty terminating `reussir.scf.yield`, the op's
/// *implicit* result is the uniquified array itself, which is the only form
/// the code generator emits.
///
/// Built raw because the op has a custom assembly format and an optional
/// result, which the generated builder does not surface.
pub fn array_with_unique_view<'c>(
    array: Value<'c, '_>,
    result_type: Type<'c>,
    body: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("reussir.array.with_unique_view", location)
        .add_operands(&[array])
        .add_results(&[result_type])
        .add_regions([body])
        .build()
        .expect("valid reussir.array.with_unique_view")
}

pub fn scf_yield<'c>(value: Option<Value<'c, '_>>, location: Location<'c>) -> Operation<'c> {
    let mut builder = OperationBuilder::new("reussir.scf.yield", location);
    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }
    builder.build().expect("valid reussir.scf.yield")
}

/// `ub.poison : <ty>` — a poison value of arbitrary type, used to terminate a
/// provably-dead branch (a non-exhaustive gap or a redundant arm the frontend
/// already diagnosed) whose result is never observed.
///
/// Built raw because melior binds no `ub` dialect at all (it is neither a
/// `melior::dialect` module nor one of its `ods` dialects), so there is no
/// generated builder; the dialect itself is registered in the context as a
/// Reussir lowering dependency. The `value` attribute is the `#ub.poison`
/// poison attribute, parsed since melior cannot construct it directly.
pub fn poison<'c>(context: &'c Context, ty: Type<'c>, location: Location<'c>) -> Operation<'c> {
    let value = melior::ir::attribute::Attribute::parse(context, "#ub.poison")
        .expect("valid #ub.poison attribute");
    OperationBuilder::new("ub.poison", location)
        .add_attributes(&[(Identifier::new(context, "value"), value)])
        .add_results(&[ty])
        .build()
        .expect("valid ub.poison")
}
