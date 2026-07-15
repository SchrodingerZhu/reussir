//===-- Types.h - Reussir dialect type C API -------------------*- C -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// C API constructors for the Reussir dialect types. The dialect types use
// custom printers/parsers, so the generic MLIR C API cannot build them; these
// entry points wrap each type's C++ `get` builder so the Rust bindings
// (mlir-sys / melior) can construct Reussir types directly rather than by
// round-tripping through textual MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_TYPES_H
#define REUSSIR_C_TYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Enums (mirroring the dialect's I32EnumAttr definitions)
//===----------------------------------------------------------------------===//

// Reference/record capability. Mirrors `reussir::Capability`.
typedef enum ReussirCapability {
  ReussirCapabilityUnspecified = 0,
  ReussirCapabilityValue = 1,
  ReussirCapabilityShared = 2,
  ReussirCapabilityFlex = 3,
  ReussirCapabilityRigid = 4,
  ReussirCapabilityField = 5,
  ReussirCapabilityRegional = 6,
} ReussirCapability;

// Reference-count atomicity. Mirrors `reussir::AtomicKind`.
typedef enum ReussirAtomicKind {
  ReussirAtomicKindNormal = 0,
  ReussirAtomicKindAtomic = 1,
} ReussirAtomicKind;

// Cell payload storage strategy. Mirrors `reussir::CellKind`.
typedef enum ReussirCellKind {
  ReussirCellKindPlain = 0,
  ReussirCellKindExclusive = 1,
  ReussirCellKindAtomic = 2,
} ReussirCellKind;

// Record flavour. Mirrors `reussir::RecordKind`.
typedef enum ReussirRecordKind {
  ReussirRecordKindCompound = 0,
  ReussirRecordKindVariant = 1,
} ReussirRecordKind;

// String lifetime scope. Mirrors `reussir::LifeScope`.
typedef enum ReussirLifeScope {
  ReussirLifeScopeGlobal = 0,
  ReussirLifeScopeLocal = 1,
} ReussirLifeScope;

//===----------------------------------------------------------------------===//
// Type constructors
//===----------------------------------------------------------------------===//

// `!reussir.raw_ptr<eleTy>`. The context is taken from `elementType`.
MlirType reussirRawPtrTypeGet(MlirType elementType);

// `!reussir.token<align: N, size: N>`.
MlirType reussirTokenTypeGet(MlirContext context, size_t align, size_t size);

// `!reussir.region`.
MlirType reussirRegionTypeGet(MlirContext context);

// `!reussir.rc<eleTy [, capability [, atomicKind]]>`. The context is taken from
// `elementType`.
MlirType reussirRcTypeGet(MlirType elementType, ReussirCapability capability,
                          ReussirAtomicKind atomicKind);

// `!reussir.nullable<ptrTy>`. The context is taken from `pointerType`.
MlirType reussirNullableTypeGet(MlirType pointerType);

// Compatibility constructor for `!reussir.cell<elementType [exclusive]>`.
// The context is taken from `elementType`.
MlirType reussirCellTypeGet(MlirType elementType, bool exclusive);

// `!reussir.cell<elementType [exclusive|atomic]>`. The kind is a single
// storage strategy; exclusive and atomic therefore cannot be combined.
MlirType reussirCellTypeGetWithKind(MlirType elementType, ReussirCellKind kind);

// `!reussir.ref<eleTy [, capability [, atomicKind]]>`. The context is taken
// from `elementType`.
MlirType reussirRefTypeGet(MlirType elementType, ReussirCapability capability,
                           ReussirAtomicKind atomicKind);

// `!reussir.hole<eleTy>`. The context is taken from `elementType`.
MlirType reussirHoleTypeGet(MlirType elementType);

// `!reussir.rc_box<eleTy [, regional]>`. The context is taken from
// `elementType`.
MlirType reussirRcBoxTypeGet(MlirType elementType, bool regional);

// `!reussir.closure<(inputTypes...) [-> outputType]>`. Pass a null `outputType`
// for a closure with no result.
MlirType reussirClosureTypeGet(MlirContext context, intptr_t nInputs,
                               MlirType const *inputTypes, MlirType outputType);

// `!reussir.closure_box<payloadTypes...>`.
MlirType reussirClosureBoxTypeGet(MlirContext context, intptr_t nPayloads,
                                  MlirType const *payloadTypes);

// `!reussir.array<shape x eleTy>`. The context is taken from `elementType`.
MlirType reussirArrayTypeGet(intptr_t nDims, int64_t const *shape,
                             MlirType elementType);

// `!reussir.view<isMutable, arrayTy>`. The context is taken from `arrayType`.
MlirType reussirViewTypeGet(bool isMutable, MlirType arrayType);

// `!reussir.ffi_object<ffiName, cleanupHook>`. `ffiName` must be a `StringAttr`
// and `cleanupHook` a `FlatSymbolRefAttr`.
MlirType reussirFFIObjectTypeGet(MlirContext context, MlirAttribute ffiName,
                                 MlirAttribute cleanupHook);

// `!reussir.str<lifeScope>`.
MlirType reussirStrTypeGet(MlirContext context, ReussirLifeScope lifeScope);

//===----------------------------------------------------------------------===//
// Record types
//===----------------------------------------------------------------------===//

// A complete record type. `name` may be a null `MlirAttribute` for an anonymous
// record, or a `StringAttr` for an identified one. `memberIsField` is an array
// of `nMembers` booleans. `fixed` pins uniform max-arm box sizing on a variant
// (`#[repr(fixed)]`); pass false for compounds and for variants that use the
// default per-constructor sizing.
MlirType
reussirRecordTypeGetComplete(MlirContext context, intptr_t nMembers,
                             MlirType const *members, bool const *memberIsField,
                             MlirAttribute name, ReussirRecordKind kind,
                             ReussirCapability defaultCapability, bool fixed);

// An identified, incomplete record type. `name` must be a `StringAttr`.
// Complete it later with `reussirRecordTypeComplete`.
MlirType reussirRecordTypeGetIncomplete(MlirContext context, MlirAttribute name,
                                        ReussirRecordKind kind);

// Completes a previously created incomplete record type in place. `fixed` pins
// uniform max-arm box sizing on a variant (`#[repr(fixed)]`); pass false for
// compounds and for variants that use the default per-constructor sizing.
void reussirRecordTypeComplete(MlirType record, intptr_t nMembers,
                               MlirType const *members,
                               bool const *memberIsField,
                               ReussirCapability defaultCapability, bool fixed);

// Whether an identified record type has had its body filled in. A handle
// obtained from `reussirRecordTypeGetIncomplete` reports `false` until
// `reussirRecordTypeComplete` runs; this lets a caller look a record up by name
// and skip rebuilding its members when it is already complete.
bool reussirRecordTypeIsComplete(MlirType record);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_TYPES_H
