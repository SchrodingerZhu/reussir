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

///===----------------------------------------------------------------------===//
// Reussir Dialect Operations Registration
//===----------------------------------------------------------------------===//
void ReussirDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Reussir/IR/ReussirOps.cpp.inc"
      >();
}
} // namespace reussir
