//===-- Types.cpp - Reussir dialect type C API ----------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Types.h"

// Complete mlir::DialectVersion before MLIR C API headers reach
// BytecodeWriter.h; MSVC's STL rejects destroying unique_ptr<T> when T is only
// forward declared.
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "Reussir/IR/ReussirTypes.h"

#include "llvm/ADT/SmallVector.h"

using namespace reussir;

namespace {
// Collects an MlirType array into a vector of unwrapped MLIR types.
llvm::SmallVector<mlir::Type> unwrapTypes(intptr_t n, MlirType const *types) {
  llvm::SmallVector<mlir::Type> result;
  result.reserve(n);
  for (intptr_t i = 0; i < n; ++i)
    result.push_back(unwrap(types[i]));
  return result;
}

Capability toCapability(ReussirCapability capability) {
  return static_cast<Capability>(capability);
}

AtomicKind toAtomicKind(ReussirAtomicKind atomicKind) {
  return static_cast<AtomicKind>(atomicKind);
}

RecordKind toRecordKind(ReussirRecordKind kind) {
  return static_cast<RecordKind>(kind);
}
} // namespace

MlirType reussirRawPtrTypeGet(MlirType elementType) {
  return wrap(RawPtrType::get(unwrap(elementType)));
}

MlirType reussirTokenTypeGet(MlirContext context, size_t align, size_t size) {
  return wrap(TokenType::get(unwrap(context), align, size));
}

MlirType reussirRegionTypeGet(MlirContext context) {
  return wrap(RegionType::get(unwrap(context)));
}

MlirType reussirRcTypeGet(MlirType elementType, ReussirCapability capability,
                          ReussirAtomicKind atomicKind) {
  mlir::Type ele = unwrap(elementType);
  return wrap(RcType::get(ele.getContext(), ele, toCapability(capability),
                          toAtomicKind(atomicKind)));
}

MlirType reussirNullableTypeGet(MlirType pointerType) {
  mlir::Type ptr = unwrap(pointerType);
  return wrap(NullableType::get(ptr.getContext(), ptr));
}

MlirType reussirCellTypeGet(MlirType elementType) {
  mlir::Type ele = unwrap(elementType);
  return wrap(CellType::get(ele.getContext(), ele));
}

MlirType reussirRefTypeGet(MlirType elementType, ReussirCapability capability,
                           ReussirAtomicKind atomicKind) {
  mlir::Type ele = unwrap(elementType);
  return wrap(RefType::get(ele.getContext(), ele, toCapability(capability),
                           toAtomicKind(atomicKind)));
}

MlirType reussirHoleTypeGet(MlirType elementType) {
  mlir::Type ele = unwrap(elementType);
  return wrap(HoleType::get(ele.getContext(), ele));
}

MlirType reussirRcBoxTypeGet(MlirType elementType, bool regional) {
  mlir::Type ele = unwrap(elementType);
  return wrap(RcBoxType::get(ele.getContext(), ele, regional));
}

MlirType reussirClosureTypeGet(MlirContext context, intptr_t nInputs,
                               MlirType const *inputTypes,
                               MlirType outputType) {
  llvm::SmallVector<mlir::Type> inputs = unwrapTypes(nInputs, inputTypes);
  mlir::Type output = outputType.ptr ? unwrap(outputType) : mlir::Type();
  return wrap(ClosureType::get(unwrap(context), inputs, output));
}

MlirType reussirClosureBoxTypeGet(MlirContext context, intptr_t nPayloads,
                                  MlirType const *payloadTypes) {
  llvm::SmallVector<mlir::Type> payloads = unwrapTypes(nPayloads, payloadTypes);
  return wrap(ClosureBoxType::get(unwrap(context), payloads));
}

MlirType reussirArrayTypeGet(intptr_t nDims, int64_t const *shape,
                             MlirType elementType) {
  mlir::Type ele = unwrap(elementType);
  return wrap(ArrayType::get(ele.getContext(),
                             llvm::ArrayRef<int64_t>(shape, nDims), ele));
}

MlirType reussirViewTypeGet(bool isMutable, MlirType arrayType) {
  auto array = llvm::cast<ArrayType>(unwrap(arrayType));
  return wrap(ViewType::get(array.getContext(), isMutable, array));
}

MlirType reussirFFIObjectTypeGet(MlirContext context, MlirAttribute ffiName,
                                 MlirAttribute cleanupHook) {
  return wrap(FFIObjectType::get(
      unwrap(context), llvm::cast<mlir::StringAttr>(unwrap(ffiName)),
      llvm::cast<mlir::FlatSymbolRefAttr>(unwrap(cleanupHook))));
}

MlirType reussirStrTypeGet(MlirContext context, ReussirLifeScope lifeScope) {
  return wrap(StrType::get(unwrap(context), static_cast<LifeScope>(lifeScope)));
}

MlirType
reussirRecordTypeGetComplete(MlirContext context, intptr_t nMembers,
                             MlirType const *members, bool const *memberIsField,
                             MlirAttribute name, ReussirRecordKind kind,
                             ReussirCapability defaultCapability, bool fixed) {
  llvm::SmallVector<mlir::Type> memberTypes = unwrapTypes(nMembers, members);
  llvm::ArrayRef<bool> fields(memberIsField, nMembers);
  if (name.ptr)
    return wrap(RecordType::get(unwrap(context), memberTypes, fields,
                                llvm::cast<mlir::StringAttr>(unwrap(name)),
                                toRecordKind(kind),
                                toCapability(defaultCapability), fixed));
  return wrap(RecordType::get(unwrap(context), memberTypes, fields,
                              toRecordKind(kind),
                              toCapability(defaultCapability), fixed));
}

MlirType reussirRecordTypeGetIncomplete(MlirContext context, MlirAttribute name,
                                        ReussirRecordKind kind) {
  return wrap(RecordType::get(unwrap(context),
                              llvm::cast<mlir::StringAttr>(unwrap(name)),
                              toRecordKind(kind)));
}

void reussirRecordTypeComplete(MlirType record, intptr_t nMembers,
                               MlirType const *members,
                               bool const *memberIsField,
                               ReussirCapability defaultCapability,
                               bool fixed) {
  llvm::SmallVector<mlir::Type> memberTypes = unwrapTypes(nMembers, members);
  llvm::ArrayRef<bool> fields(memberIsField, nMembers);
  llvm::cast<RecordType>(unwrap(record))
      .complete(memberTypes, fields, toCapability(defaultCapability), fixed);
}

bool reussirRecordTypeIsComplete(MlirType record) {
  return llvm::cast<RecordType>(unwrap(record)).getComplete();
}
