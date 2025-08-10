//===-- ReussirOps.cpp - Reussir operations implementation ------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the operations used in the Reussir dialect.
//
//===----------------------------------------------------------------------===//
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.cpp.inc"

namespace reussir {
///===----------------------------------------------------------------------===//
// ReussirTokenReinterpretOp
//===----------------------------------------------------------------------===//
// ReinterpretOp verification
mlir::LogicalResult ReussirTokenReinterpretOp::verify() {
  TokenType tokenType = getToken().getType();
  RefType resultType = getReinterpreted().getType();
  mlir::Type elementType = resultType.getElementType();
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(elementType);
  auto size = dataLayout.getTypeSize(elementType);
  if (!size.isFixed())
    return emitOpError("reinterpreted type must have a fixed size");
  if (tokenType.getAlign() != alignment)
    return emitOpError(
               "token alignment must match reinterpreted type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", element alignment: " << alignment;
  if (tokenType.getSize() != size)
    return emitOpError("token size must match reinterpreted type size, ")
           << "token size:  " << tokenType.getSize()
           << ", element size: " << size.getFixedValue();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir RC Operation
//===----------------------------------------------------------------------===//
// RcIncOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcIncOp::verify() {
  RcType RcType = getRcPtr().getType();
  if (RcType.getCapability() == reussir::Capability::flex)
    return emitOpError("cannot increase reference count of a flex RC type");

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// RcDecOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcDecOp::verify() {
  RcType RcType = getRcPtr().getType();
  NullableType nullableType = getNullableToken().getType();
  TokenType tokenType = llvm::dyn_cast<TokenType>(nullableType.getPtrTy());
  if (!tokenType)
    return emitOpError("nullable token must be of TokenType");
  mlir::Type eleTy = RcType.getElementType();
  if (RcType.getCapability() == reussir::Capability::flex)
    return emitOpError("cannot decrease reference count of a flex RC type");
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(eleTy);
  auto size = dataLayout.getTypeSize(eleTy);
  if (!size.isFixed())
    return emitOpError("managed type must have a fixed size");

  if (tokenType.getAlign() != alignment)
    return emitOpError("token alignment must match managed type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", element alignment: " << alignment;

  if (tokenType.getSize() != size)
    return emitOpError("token size must match managed type size, ")
           << "token size: " << tokenType.getSize()
           << ", element size: " << size.getFixedValue();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcCreateOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcCreateOp::verify() {
  TokenType tokenType = getToken().getType();
  RcType RcType = getRcPtr().getType();
  mlir::Type valueType = getValue().getType();
  if (valueType != RcType.getElementType())
    return emitOpError("value type must match RC element type, ")
           << "value type: " << valueType
           << ", RC element type: " << RcType.getElementType();
  Capability expectedCap = getRegion() == nullptr ? reussir::Capability::shared
                                                  : reussir::Capability::flex;
  if (RcType.getCapability() != expectedCap)
    return emitOpError("RC type capability must be ")
           << stringifyCapability(expectedCap) << ", but got "
           << stringifyCapability(RcType.getCapability());
  auto rcBoxType =
      RcBoxType::get(getContext(), valueType, getRegion() != nullptr);
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  auto size = dataLayout.getTypeSize(rcBoxType);
  if (!size.isFixed())
    return emitOpError("RC type must have a fixed size");
  if (tokenType.getAlign() != alignment)
    return emitOpError("token alignment must match RC type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", RC type alignment: " << alignment;
  if (tokenType.getSize() != size)
    return emitOpError("token size must match RC type size, ")
           << "token size: " << tokenType.getSize()
           << ", RC type size: " << size.getFixedValue();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Borrow Operation
//===----------------------------------------------------------------------===//
// BorrowOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcBorrowOp::verify() {
  RcType rcType = getRcPtr().getType();
  RefType refType = getBorrowed().getType();
  if (refType.getCapability() != rcType.getCapability())
    return emitOpError(
               "borrowed type capability must match RC type capability, ")
           << "borrowed type capability: "
           << stringifyCapability(refType.getCapability())
           << ", RC type capability: "
           << stringifyCapability(rcType.getCapability());

  if (refType.getElementType() != rcType.getElementType())
    return emitOpError(
               "borrowed type element type must match RC element type, ")
           << "borrowed type element type: " << refType.getElementType()
           << ", RC element type: " << rcType.getElementType();

  if (refType.getAtomicKind() != rcType.getAtomicKind())
    return emitOpError(
               "borrowed type atomic kind must match RC type atomic kind, ")
           << "borrowed type atomic kind: "
           << stringifyAtomicKind(refType.getAtomicKind())
           << ", RC type atomic kind: "
           << stringifyAtomicKind(rcType.getAtomicKind());

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// Reussir Record Operations
//===----------------------------------------------------------------------===//
// RecordCompoundOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordCompoundOp::verify() {
  auto compoundType = getCompound().getType();
  if (!compoundType.getComplete())
    return emitOpError("cannot assemble incomplete compound record");
  if (!compoundType.isCompound())
    return emitOpError("compound type must be a compound record");
  if (compoundType.getMembers().size() != getFields().size())
    return emitOpError("number of fields must match number of members");
  bool hasFieldCapability = false;
  for (auto [field, member, memberCapability] :
       llvm::zip(getFields(), compoundType.getMembers(),
                 compoundType.getMemberCapabilities())) {
    // Since this is assemble phase, assume flex ref capability.
    mlir::Type projectedType =
        reussir::getProjectedType(member, memberCapability, Capability::flex);
    if (projectedType != field.getType())
      return emitOpError("field type must match projected member type, ")
             << "field type: " << field.getType()
             << ", projected member type: " << projectedType;
    hasFieldCapability |= (memberCapability == Capability::field);
  }
  if (hasFieldCapability)
    return emitOpError("TODO: check this is nested in a region operation");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Record Variant Op
//===----------------------------------------------------------------------===//
// RecordVariantOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordVariantOp::verify() {
  auto variantType = getVariant().getType();
  if (!variantType.getComplete())
    return emitOpError("cannot assemble incomplete variant record");
  if (!variantType.isVariant())
    return emitOpError("variant type must be a variant record");
  size_t tag = getTag().getZExtValue();
  if (tag >= variantType.getMembers().size())
    return emitOpError("tag out of bounds");
  mlir::Type targetVariantType = variantType.getMembers()[tag];
  Capability targetVariantCapability = variantType.getMemberCapabilities()[tag];
  mlir::Type projectedType = reussir::getProjectedType(
      targetVariantType, targetVariantCapability, Capability::flex);
  if (projectedType != getValue().getType())
    return emitOpError("value type must match projected type, ")
           << "value type: " << getValue().getType()
           << ", projected type: " << projectedType;
  if (targetVariantCapability == Capability::flex)
    return emitOpError("TODO: check this is nested in a region operation");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Reference Operations
//===----------------------------------------------------------------------===//
// ReferenceProjectOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirReferenceProjectOp::verify() {
  RefType refType = getRef().getType();
  RefType projectedType = getProjected().getType();

  // Check that the reference element type is a record type
  mlir::Type elementType = refType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
  if (!recordType)
    return emitOpError("reference element type must be a record type, got: ")
           << elementType;

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot project into incomplete record");

  // Check that the index is within bounds
  size_t index = getIndex().getZExtValue();
  if (index >= recordType.getMembers().size())
    return emitOpError("index out of bounds: ")
           << index << " >= " << recordType.getMembers().size();

  // Get the member type and capability at the specified index
  mlir::Type memberType = recordType.getMembers()[index];
  Capability memberCapability = recordType.getMemberCapabilities()[index];

  // Calculate the expected projected type based on the member type, member
  // capability, and reference capability
  mlir::Type expectedProjectedType = reussir::getProjectedType(
      memberType, memberCapability, refType.getCapability());

  // Check that the projected type matches the expected type
  if (expectedProjectedType != projectedType.getElementType())
    return emitOpError("projected type mismatch: expected ")
           << expectedProjectedType << ", got "
           << projectedType.getElementType();

  // Check that the projected reference has the same capability as the original
  // reference. Or if the refType is flex, then the projected type can be field
  // if target is field.
  bool isOfSameCapability =
      projectedType.getCapability() == refType.getCapability();
  bool projectFieldOutOfFlex =
      projectedType.getCapability() == Capability::field &&
      refType.getCapability() == Capability::flex &&
      memberCapability == Capability::field;
  if (!isOfSameCapability && !projectFieldOutOfFlex)
    return emitOpError(
               "projected reference capability must match original "
               "reference capability or be a field projection: original "
               "capability: ")
           << stringifyCapability(refType.getCapability()) << ", got "
           << stringifyCapability(projectedType.getCapability());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefSpilledOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefSpilledOp::verify() {
  mlir::Type valueType = getValue().getType();
  RefType refType = getSpilled().getType();
  if (valueType != refType.getElementType())
    return emitOpError("value type must match spilled element type, ")
           << "value type: " << valueType
           << ", spilled element type: " << refType.getElementType();
  if (refType.getCapability() != reussir::Capability::unspecified)
    return emitOpError("spilled type capability must be unspecified, ")
           << "spilled type capability: "
           << stringifyCapability(refType.getCapability());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefLoadOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefLoadOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type valueType = getValue().getType();
  if (valueType != refType.getElementType())
    return emitOpError("value type must match reference element type, ")
           << "value type: " << valueType
           << ", reference element type: " << refType.getElementType();
  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// Reussir Dialect Operations Registration
//===-----------------------------------------------------------------------===//
void ReussirDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Reussir/IR/ReussirOps.cpp.inc"
      >();
}
} // namespace reussir
