//! Constructors for the Reussir dialect's types.
//!
//! The dialect types use custom printers/parsers, so they cannot be built with
//! the generic MLIR type APIs. These functions wrap the Reussir C API type
//! constructors, mirroring melior's own typed type builders (such as
//! [`melior::dialect::llvm::r#type`]).

use std::ptr::null;

use melior::Context;
use melior::ir::Type;
use melior::ir::attribute::{AttributeLike, FlatSymbolRefAttribute, StringAttribute};
use melior::ir::r#type::TypeLike;

use reussir_backend_sys as sys;

pub use sys::{ReussirAtomicKind, ReussirCapability, ReussirLifeScope, ReussirRecordKind};

fn null_type() -> sys::mlir_sys::MlirType {
    sys::mlir_sys::MlirType { ptr: null() }
}

fn null_attribute() -> sys::mlir_sys::MlirAttribute {
    sys::mlir_sys::MlirAttribute { ptr: null() }
}

/// Creates a `!reussir.raw_ptr<element>` type.
pub fn raw_ptr(element: Type) -> Type {
    unsafe { Type::from_raw(sys::reussirRawPtrTypeGet(element.to_raw())) }
}

/// Creates a `!reussir.token<align: …, size: …>` type.
pub fn token(context: &Context, align: usize, size: usize) -> Type<'_> {
    unsafe { Type::from_raw(sys::reussirTokenTypeGet(context.to_raw(), align, size)) }
}

/// Creates a `!reussir.region` type.
pub fn region(context: &Context) -> Type<'_> {
    unsafe { Type::from_raw(sys::reussirRegionTypeGet(context.to_raw())) }
}

/// Creates a `!reussir.rc<element, capability, atomicKind>` type.
pub fn rc(element: Type, capability: ReussirCapability, atomic_kind: ReussirAtomicKind) -> Type {
    unsafe {
        Type::from_raw(sys::reussirRcTypeGet(
            element.to_raw(),
            capability,
            atomic_kind,
        ))
    }
}

/// Creates a `!reussir.nullable<pointer>` type. `pointer` must be a non-null
/// pointer type (e.g. a `!reussir.token` or `!reussir.ref`).
pub fn nullable(pointer: Type) -> Type {
    unsafe { Type::from_raw(sys::reussirNullableTypeGet(pointer.to_raw())) }
}

/// Creates a `!reussir.ref<element, capability, atomicKind>` type.
pub fn r#ref(element: Type, capability: ReussirCapability, atomic_kind: ReussirAtomicKind) -> Type {
    unsafe {
        Type::from_raw(sys::reussirRefTypeGet(
            element.to_raw(),
            capability,
            atomic_kind,
        ))
    }
}

/// Creates a `!reussir.hole<element>` type.
pub fn hole(element: Type) -> Type {
    unsafe { Type::from_raw(sys::reussirHoleTypeGet(element.to_raw())) }
}

/// Creates a `!reussir.rc_box<element>` type, `regional` selecting the regional
/// box layout.
pub fn rc_box(element: Type, regional: bool) -> Type {
    unsafe { Type::from_raw(sys::reussirRcBoxTypeGet(element.to_raw(), regional)) }
}

/// Creates a `!reussir.closure<(inputs...) -> output>` type. Pass `None` for
/// `output` for a closure with no result.
pub fn closure<'c>(
    context: &'c Context,
    inputs: &[Type<'c>],
    output: Option<Type<'c>>,
) -> Type<'c> {
    let raw_inputs: Vec<_> = inputs.iter().map(TypeLike::to_raw).collect();
    let raw_output = output.map(|t| t.to_raw()).unwrap_or_else(null_type);
    unsafe {
        Type::from_raw(sys::reussirClosureTypeGet(
            context.to_raw(),
            raw_inputs.len() as isize,
            raw_inputs.as_ptr(),
            raw_output,
        ))
    }
}

/// Creates a `!reussir.closure_box<payloads...>` type.
pub fn closure_box<'c>(context: &'c Context, payloads: &[Type<'c>]) -> Type<'c> {
    let raw_payloads: Vec<_> = payloads.iter().map(TypeLike::to_raw).collect();
    unsafe {
        Type::from_raw(sys::reussirClosureBoxTypeGet(
            context.to_raw(),
            raw_payloads.len() as isize,
            raw_payloads.as_ptr(),
        ))
    }
}

/// Creates a `!reussir.array<shape x element>` type.
pub fn array<'c>(shape: &[i64], element: Type<'c>) -> Type<'c> {
    unsafe {
        Type::from_raw(sys::reussirArrayTypeGet(
            shape.len() as isize,
            shape.as_ptr(),
            element.to_raw(),
        ))
    }
}

/// Creates a `!reussir.view<isMutable, array>` type. `array` must be a
/// `!reussir.array` type.
pub fn view(is_mutable: bool, array: Type) -> Type {
    unsafe { Type::from_raw(sys::reussirViewTypeGet(is_mutable, array.to_raw())) }
}

/// Creates a `!reussir.ffi_object<ffiName, cleanupHook>` type.
pub fn ffi_object<'c>(
    context: &'c Context,
    ffi_name: StringAttribute<'c>,
    cleanup_hook: FlatSymbolRefAttribute<'c>,
) -> Type<'c> {
    unsafe {
        Type::from_raw(sys::reussirFFIObjectTypeGet(
            context.to_raw(),
            ffi_name.to_raw(),
            cleanup_hook.to_raw(),
        ))
    }
}

/// Creates a `!reussir.str<lifeScope>` type.
pub fn str(context: &Context, life_scope: ReussirLifeScope) -> Type<'_> {
    unsafe { Type::from_raw(sys::reussirStrTypeGet(context.to_raw(), life_scope)) }
}

/// Creates a complete `!reussir.record` type. Pass `None` for `name` to build an
/// anonymous record, or a [`StringAttribute`] to build an identified one.
/// `members` and `member_is_field` must have the same length.
pub fn record_complete<'c>(
    context: &'c Context,
    members: &[Type<'c>],
    member_is_field: &[bool],
    name: Option<StringAttribute<'c>>,
    kind: ReussirRecordKind,
    default_capability: ReussirCapability,
) -> Type<'c> {
    assert_eq!(
        members.len(),
        member_is_field.len(),
        "members and member_is_field must have the same length"
    );
    let raw_members: Vec<_> = members.iter().map(TypeLike::to_raw).collect();
    let raw_name = name.map(|n| n.to_raw()).unwrap_or_else(null_attribute);
    unsafe {
        Type::from_raw(sys::reussirRecordTypeGetComplete(
            context.to_raw(),
            raw_members.len() as isize,
            raw_members.as_ptr(),
            member_is_field.as_ptr(),
            raw_name,
            kind,
            default_capability,
        ))
    }
}

/// Creates an identified but incomplete `!reussir.record` type. Complete it
/// later with [`record_complete_in_place`].
pub fn record_incomplete<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    kind: ReussirRecordKind,
) -> Type<'c> {
    unsafe {
        Type::from_raw(sys::reussirRecordTypeGetIncomplete(
            context.to_raw(),
            name.to_raw(),
            kind,
        ))
    }
}

/// Whether an identified record type has had its body filled in. A type from
/// [`record_incomplete`] reports `false` until [`record_complete_in_place`]
/// runs, so a caller can look a record up by name and skip rebuilding its
/// members when it is already complete.
pub fn record_is_complete(record: Type) -> bool {
    unsafe { sys::reussirRecordTypeIsComplete(record.to_raw()) }
}

/// Completes a previously created incomplete record type in place. `members` and
/// `member_is_field` must have the same length.
pub fn record_complete_in_place(
    record: Type,
    members: &[Type],
    member_is_field: &[bool],
    default_capability: ReussirCapability,
) {
    assert_eq!(
        members.len(),
        member_is_field.len(),
        "members and member_is_field must have the same length"
    );
    let raw_members: Vec<_> = members.iter().map(TypeLike::to_raw).collect();
    unsafe {
        sys::reussirRecordTypeComplete(
            record.to_raw(),
            raw_members.len() as isize,
            raw_members.as_ptr(),
            member_is_field.as_ptr(),
            default_capability,
        );
    }
}
