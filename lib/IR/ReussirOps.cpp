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
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
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
// RecordTagOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordTagOp::verify() {
  RefType variantType = getVariant().getType();

  // Check that the input is a reference to a record type
  mlir::Type elementType = variantType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
  if (!recordType)
    return emitOpError("input must be a reference to a record type, got: ")
           << elementType;

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot get tag of incomplete record");

  // Check that the record is a variant record
  if (!recordType.isVariant())
    return emitOpError("can only get tag of variant records");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Reference Operations
//===----------------------------------------------------------------------===//
// ReferenceProjectOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefProjectOp::verify() {
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

//===----------------------------------------------------------------------===//
// RefStoreOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefStoreOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type valueType = getValue().getType();

  // Check that the target reference has field capability
  if (refType.getCapability() != reussir::Capability::field)
    return emitOpError("target reference must have field capability, got: ")
           << stringifyCapability(refType.getCapability());

  // Check that the value type matches the reference element type
  if (valueType != refType.getElementType())
    return emitOpError("value type must match reference element type, ")
           << "value type: " << valueType
           << ", reference element type: " << refType.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Region Operations
//===----------------------------------------------------------------------===//
// RegionRunOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRegionRunOp::verify() {
  // - Check that the region has exactly one argument of !reussir.region type
  // - Check that the region can optionally yield a value
  if (getRegion().getNumArguments() != 1)
    return emitOpError("region must have exactly one argument");
  mlir::Type argType = getRegion().getArgumentTypes()[0];
  if (argType != reussir::RegionType::get(getContext()))
    return emitOpError("region argument must be of !reussir.region type");
  if (getResults().size() > 1)
    return emitOpError("region must have at most one result");
  if (mlir::Value result = getResult()) {
    RcType rcType = llvm::dyn_cast<RcType>(result.getType());
    if (!rcType)
      return emitOpError("region result must be of RC type (for now)");
    if (rcType.getCapability() != reussir::Capability::rigid &&
        rcType.getCapability() != reussir::Capability::shared)
      return emitOpError("region result must be of rigid or shared RC type");
  }
  // Check that the region is not nested in the same function
  if (this->getOperation()->getParentOfType<ReussirRegionRunOp>() != nullptr)
    return emitOpError("region cannot be nested in the same function");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRegionYieldOp::verify() {
  // Check that it is consistent on whether yielding a value or not
  // If yielding a value, check that the value is of RC type, and it the
  // capability of the Rc type is flex, convert the return type to rigid
  // counterpart. Then verify that the value is of the same type as the region
  // argument.
  auto parentOp = this->getParentOp();
  if (getValue() != nullptr) {
    if (parentOp->getNumResults() != 1)
      return emitOpError("region must have exactly one result");
    RcType rcType = getValue().getType();
    if (rcType.getCapability() == reussir::Capability::flex)
      rcType = RcType::get(getContext(), rcType.getElementType(),
                           reussir::Capability::rigid);
    if (rcType != parentOp->getResult(0).getType())
      return emitOpError("value type must match region result type, ")
             << "value type: " << rcType
             << ", region result type: " << parentOp->getResult(0).getType();
  } else if (parentOp->getNumResults() != 0)
    return emitOpError("region must have no result");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionRunOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirRegionRunOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.emplace_back(&getRegion());
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.emplace_back(getResults());
}

//===----------------------------------------------------------------------===//
// Reussir Record Dispatch Op
//===----------------------------------------------------------------------===//
// RecordDispatchOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordDispatchOp::verify() {
  // TODO: Implement verification
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RecordDispatchOp custom assembly format
//===----------------------------------------------------------------------===//
mlir::ParseResult ReussirRecordDispatchOp::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand variantRefOperand;
  llvm::SMLoc variantRefOperandsLoc;
  RefType variantRefType;
  mlir::Type valueType;
  llvm::SmallVector<std::unique_ptr<mlir::Region>> regions;
  llvm::SmallVector<mlir::Attribute> tagSets;
  if (parser.parseLParen())
    return mlir::failure();

  variantRefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(variantRefOperand))
    return mlir::failure();
  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseCustomTypeWithFallback(variantRefType))
    return mlir::failure();

  if (parser.parseRParen())
    return mlir::failure();
  if (llvm::succeeded(parser.parseOptionalArrow()))
    if (llvm::failed(parser.parseType(valueType)))
      return mlir::failure();

  if (parser.parseLBrace())
    return mlir::failure();

  llvm::SmallVector<int64_t> tags;
  auto parseTag = [&]() -> mlir::ParseResult {
    llvm::APInt tag;
    if (parser.parseInteger(tag))
      return mlir::failure();
    if (tag.isNegative())
      return parser.emitError(parser.getCurrentLocation(),
                              "tag must be positive");
    tags.push_back(tag.getZExtValue());
    return mlir::success();
  };
  while (llvm::succeeded(parser.parseOptionalLSquare())) {
    if (llvm::failed(parser.parseCommaSeparatedList(parseTag)))
      return llvm::failure();
    if (llvm::failed(parser.parseRSquare()))
      return llvm::failure();
    if (llvm::failed(parser.parseArrow()))
      return mlir::failure();
    if (llvm::failed(parser.parseRegion(
            *regions.emplace_back(std::make_unique<mlir::Region>()))))
      return mlir::failure();
    tagSets.push_back(mlir::DenseI64ArrayAttr::get(parser.getContext(), tags));
    tags.clear();
  }
  if (parser.parseRBrace())
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  if (valueType)
    result.addTypes(valueType);
  result.addRegions(regions);
  result.addAttribute("tagSets",
                      mlir::ArrayAttr::get(parser.getContext(), tagSets));
  if (llvm::failed(parser.resolveOperands({variantRefOperand}, variantRefType,
                                          variantRefOperandsLoc,
                                          result.operands)))
    return mlir::failure();
  return mlir::success();
}

void ReussirRecordDispatchOp::print(mlir::OpAsmPrinter &p) {
  // Print the variant reference operand and type
  p << "(";
  p.printOperand(getVariant());
  p << ' ' << ":";
  p.printType(getVariant().getType());
  p << ")";

  // Print optional result type
  if (getValue()) {
    p << ' ' << "->" << ' ';
    p.printType(getValue().getType());
  }

  // Print the dispatch body
  p << "{";
  p.increaseIndent();
  for (auto [region, tagSetAttr] : llvm::zip(getRegions(), getTagSets())) {
    p.printNewline();
    auto tagSet = llvm::cast<mlir::DenseI64ArrayAttr>(tagSetAttr);
    p << '[';
    llvm::interleaveComma(tagSet.asArrayRef(), p,
                          [&](int64_t tag) { p << tag; });
    p << ']' << ' ' << "->" << ' ';
    p.printRegion(region);
  }
  p.decreaseIndent();
  p.printNewline();
  p << "}";
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"tagSets"});
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
