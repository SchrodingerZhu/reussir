//! Constructors for the Reussir dialect's debug-info attributes.
//!
//! These attributes describe a value's debug type or a source variable; codegen
//! attaches them as `FusedLoc` metadata and the debug-info conversion pass
//! translates them to LLVM DI. Like the dialect's types, their custom storage
//! cannot be built through the generic MLIR APIs, so these wrap the Reussir C API
//! constructors.

use melior::Context;
use melior::ir::attribute::{AttributeLike, StringAttribute};
use melior::ir::r#type::TypeLike;
use melior::ir::{Attribute, Type};

use reussir_backend_sys as sys;

fn raw_attrs(attrs: &[Attribute]) -> Vec<sys::mlir_sys::MlirAttribute> {
    attrs.iter().map(AttributeLike::to_raw).collect()
}

/// A `#reussir.dbg_inttype` debug type for an integer `inner` named `name`.
pub fn int_type<'c>(
    context: &'c Context,
    inner: Type<'c>,
    is_signed: bool,
    name: StringAttribute<'c>,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGIntTypeAttrGet(
            context.to_raw(),
            inner.to_raw(),
            is_signed,
            name.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_fptype` debug type for a floating-point `inner` named `name`.
pub fn fp_type<'c>(
    context: &'c Context,
    inner: Type<'c>,
    name: StringAttribute<'c>,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGFPTypeAttrGet(
            context.to_raw(),
            inner.to_raw(),
            name.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_record_member`: a field `name` with debug type `type_attr`.
pub fn record_member<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    type_attr: Attribute<'c>,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGRecordMemberAttrGet(
            context.to_raw(),
            name.to_raw(),
            type_attr.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_recordtype` over `members` (each a [`record_member`]).
/// `underlying` is the lowered MLIR record used for size/offset computation.
pub fn record_type<'c>(
    context: &'c Context,
    members: &[Attribute<'c>],
    is_variant: bool,
    underlying: Type<'c>,
    name: StringAttribute<'c>,
) -> Attribute<'c> {
    let raw = raw_attrs(members);
    unsafe {
        Attribute::from_raw(sys::reussirDBGRecordTypeAttrGet(
            context.to_raw(),
            raw.len() as isize,
            raw.as_ptr(),
            is_variant,
            underlying.to_raw(),
            name.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_subprogram` for a function named `raw_name` with parameter
/// debug types `type_params`.
pub fn subprogram<'c>(
    context: &'c Context,
    raw_name: StringAttribute<'c>,
    type_params: &[Attribute<'c>],
) -> Attribute<'c> {
    let raw = raw_attrs(type_params);
    unsafe {
        Attribute::from_raw(sys::reussirDBGSubprogramAttrGet(
            context.to_raw(),
            raw_name.to_raw(),
            raw.len() as isize,
            raw.as_ptr(),
        ))
    }
}

/// A `#reussir.dbg_boxedtype` for a reference-counted value, wrapping the
/// payload's debug type `inner`.
pub fn boxed_type<'c>(context: &'c Context, inner: Attribute<'c>) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGBoxedTypeAttrGet(
            context.to_raw(),
            inner.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_localvar` of debug type `dbg_type` and source `name`.
pub fn local_var<'c>(
    context: &'c Context,
    dbg_type: Attribute<'c>,
    name: StringAttribute<'c>,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGLocalVarAttrGet(
            context.to_raw(),
            dbg_type.to_raw(),
            name.to_raw(),
        ))
    }
}

/// A `#reussir.dbg_func_arg` of debug type `dbg_type`, source `name`, and 1-based
/// `index`.
pub fn func_arg<'c>(
    context: &'c Context,
    dbg_type: Attribute<'c>,
    name: StringAttribute<'c>,
    index: u32,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(sys::reussirDBGFuncArgAttrGet(
            context.to_raw(),
            dbg_type.to_raw(),
            name.to_raw(),
            index,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::r#type::IntegerType;

    #[test]
    fn debug_attrs_construct_and_print() {
        let context = crate::context();
        let i64 = IntegerType::new(&context, 64).into();
        let name = |s| StringAttribute::new(&context, s);

        let int = int_type(&context, i64, true, name("i64"));
        assert!(int.to_string().contains("dbg_inttype"), "{int}");

        let member = record_member(&context, name("x"), int);
        let record = record_type(&context, &[member], false, i64, name("Point"));
        assert!(record.to_string().contains("dbg_recordtype"), "{record}");

        let local = local_var(&context, int, name("v"));
        assert!(local.to_string().contains("dbg_localvar"), "{local}");

        let arg = func_arg(&context, int, name("a"), 1);
        assert!(arg.to_string().contains("dbg_func_arg"), "{arg}");

        let sp = subprogram(&context, name("add"), &[int, int]);
        assert!(sp.to_string().contains("dbg_subprogram"), "{sp}");
    }
}
