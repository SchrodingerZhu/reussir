//! Hand-written op builders for cases melior's generated builders don't cover
//! well enough.
//!
//! melior's ODS/dialect builders are preferred everywhere they fit; this module
//! holds the few operations whose generated builder is missing a needed
//! parameter (so the result/attributes can't be set), keeping the raw
//! [`OperationBuilder`] usage contained in the backend rather than leaking into
//! downstream code generators.

use melior::Context;
use melior::ir::attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute};
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
use melior::ir::{Identifier, Location, Operation, Type, Value};

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
