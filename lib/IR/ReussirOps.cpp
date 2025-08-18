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
#include <array>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
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
// Reussir Record Coerce Op
//===----------------------------------------------------------------------===//
// RecordCoerceOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordCoerceOp::verify() {
  RefType variantRefType = getVariant().getType();
  RefType coercedRefType = getCoerced().getType();

  // Check that the input reference is a reference to a record type
  mlir::Type variantElementType = variantRefType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(variantElementType);
  if (!recordType)
    return emitOpError("input must be a reference to a record type, got: ")
           << variantElementType;

  // Check that the record is a variant record
  if (!recordType.isVariant())
    return emitOpError("input must be a reference to a variant record");

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot coerce incomplete variant record");

  // Get the tag and validate it's within bounds
  size_t tag = getTag().getZExtValue();
  if (tag >= recordType.getMembers().size())
    return emitOpError("tag out of bounds: ")
           << tag << " >= " << recordType.getMembers().size();

  // Get the target variant element type at the specified tag position
  mlir::Type targetVariantElementType = recordType.getMembers()[tag];

  // Check that the output reference element type matches the target variant
  // element type
  mlir::Type coercedElementType = coercedRefType.getElementType();
  if (coercedElementType != targetVariantElementType)
    return emitOpError("output reference element type must match target "
                       "variant element type, ")
           << "expected: " << targetVariantElementType
           << ", got: " << coercedElementType;

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
// NullableDispatchOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirNullableDispatchOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the parent operation, branch into one of the regions.
  if (point.isParent()) {
    regions.emplace_back(&getNonNullRegion());
    regions.emplace_back(&getNullRegion());
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.emplace_back(getResults());
}

//===----------------------------------------------------------------------===//
// RecordDispatchOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirRecordDispatchOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the parent operation, branch into one of the regions.
  if (point.isParent()) {
    for (mlir::Region &region : getRegions())
      regions.emplace_back(&region);
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
  // Get the variant reference type and extract the record type
  RefType variantRefType = getVariant().getType();
  RecordType recordType =
      llvm::dyn_cast<RecordType>(variantRefType.getElementType());
  if (!recordType)
    return emitOpError("variant operand must be a reference to a record type");

  if (!recordType.isVariant())
    return emitOpError(
        "variant operand must be a reference to a variant record type");

  if (!recordType.getComplete())
    return emitOpError("variant record type must be complete");

  // Get the number of variant members
  auto members = recordType.getMembers();
  size_t numMembers = members.size();

  // Check that tagSets array size matches the number of regions
  auto tagSets = getTagSets();
  auto regions = getRegions();

  if (tagSets.size() != regions.size())
    return emitOpError("number of tag sets must match number of regions, ")
           << "tag sets: " << tagSets.size() << ", regions: " << regions.size();

  // Track which tags are covered
  llvm::SmallSet<int64_t, 8> coveredTags;

  // Verify each tag set and corresponding region
  for (size_t i = 0; i < tagSets.size(); ++i) {
    auto tagSetAttr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(tagSets[i]);
    if (!tagSetAttr)
      return emitOpError("tag set ") << i << " must be a DenseI64ArrayAttr";

    auto tagSet = tagSetAttr.asArrayRef();
    if (tagSet.empty())
      return emitOpError("tag set ") << i << " must have at least one value";

    // Check that all tags in this set are valid and not already covered
    for (int64_t tag : tagSet) {
      if (tag < 0 || static_cast<size_t>(tag) >= numMembers)
        return emitOpError("tag ")
               << tag << " in tag set " << i << " is out of range [0, "
               << numMembers << ")";
      if (coveredTags.contains(tag))
        return emitOpError("tag ")
               << tag << " in tag set " << i
               << " is already covered by a previous tag set";
      coveredTags.insert(tag);
    }

    // Verify region argument types based on tag set size
    mlir::Region &region = regions[i];
    if (region.empty())
      return emitOpError("region ") << i << " cannot be empty";

    mlir::Block &block = region.front();
    if (tagSet.size() == 1) {
      // Single tag: region should accept a reference to the target variant
      // element type
      if (block.getNumArguments() != 1)
        return emitOpError("region ")
               << i << " must have exactly one argument for single tag";

      int64_t tag = tagSet[0];
      mlir::Type expectedType = members[tag];
      mlir::Type actualType = block.getArgument(0).getType();

      // The argument should be a reference to the member type
      RefType actualRefType = llvm::dyn_cast<RefType>(actualType);
      if (!actualRefType)
        return emitOpError("region ")
               << i << " argument must be a reference type";

      if (actualRefType.getElementType() != expectedType)
        return emitOpError("region ")
               << i << " argument type must match variant member type, "
               << "argument type: " << actualRefType.getElementType()
               << ", expected type: " << expectedType;
    } else if (block.getNumArguments() != 0)
      return emitOpError("region ")
             << i << " must have no arguments for multiple tags";
  }

  // Check that all possible tags are covered
  for (size_t i = 0; i < numMembers; ++i)
    if (!coveredTags.contains(i))
      return emitOpError("tag ") << i << " is not covered by any tag set";

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

//===----------------------------------------------------------------------===//
// Reussir Nullable Coerce Op
//===----------------------------------------------------------------------===//
// NullableCoerceOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirNullableCoerceOp::verify() {
  NullableType nullableType = getNullable().getType();
  mlir::Type coercedType = getNonnull().getType();

  // Check that the coerced type is the same as the PtrTy of the nullable input
  mlir::Type expectedType = nullableType.getPtrTy();
  if (coercedType != expectedType)
    return emitOpError("coerced type must match nullable pointer type, ")
           << "expected: " << expectedType << ", got: " << coercedType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Nullable Dispatch Op
//===----------------------------------------------------------------------===//
// NullableDispatchOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirNullableDispatchOp::verify() {
  // Get the nullable type and extract the inner pointer type
  NullableType nullableType = getNullable().getType();
  mlir::Type innerType = nullableType.getPtrTy();

  // Verify nonNullRegion has exactly one argument matching the inner type
  mlir::Region &nonNullRegion = getNonNullRegion();
  if (nonNullRegion.empty())
    return emitOpError("nonnull region cannot be empty");

  mlir::Block &nonNullBlock = nonNullRegion.front();
  if (nonNullBlock.getNumArguments() != 1)
    return emitOpError("nonnull region must have exactly one argument");

  mlir::Type nonNullArgType = nonNullBlock.getArgument(0).getType();
  if (nonNullArgType != innerType)
    return emitOpError(
               "nonnull region argument type must match nullable inner type, ")
           << "argument type: " << nonNullArgType
           << ", expected type: " << innerType;

  // Verify nullRegion has no arguments
  mlir::Region &nullRegion = getNullRegion();
  if (nullRegion.empty())
    return emitOpError("null region cannot be empty");

  mlir::Block &nullBlock = nullRegion.front();
  if (nullBlock.getNumArguments() != 0)
    return emitOpError("null region must have no arguments");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Scf Yield Op
//===----------------------------------------------------------------------===//
// ScfYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirScfYieldOp::verify() {
  mlir::Type yieldedType = getValue() ? getValue().getType() : mlir::Type{};
  mlir::Type expectedType = mlir::Type{};
  if (auto nullableParent =
          getOperation()->getParentOfType<ReussirNullableDispatchOp>())
    expectedType = nullableParent.getValue()
                       ? nullableParent.getValue().getType()
                       : mlir::Type{};
  else if (auto recordParent =
               getOperation()->getParentOfType<ReussirRecordDispatchOp>())
    expectedType = recordParent.getValue() ? recordParent.getValue().getType()
                                           : mlir::Type{};
  else
    llvm_unreachable("unexpected parent operation");

  if (expectedType && !yieldedType)
    return emitOpError(
        "parent operation expected a value, but nothing is yielded");
  if (yieldedType && !expectedType)
    return emitOpError(
        "parent operation did not expect a value, but one is yielded");

  if (yieldedType && expectedType && yieldedType != expectedType)
    return emitOpError("yielded type must match parent operation result type, ")
           << "yielded type: " << yieldedType
           << ", expected type: " << expectedType;

  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// Reussir Closure Create Op
//===-----------------------------------------------------------------------===//
// ClosureCreateOp custom assembly format
//===-----------------------------------------------------------------------===//
mlir::ParseResult ReussirClosureCreateOp::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  llvm::SMLoc operationLoc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand tokenOperand;
  mlir::FlatSymbolRefAttr vtableAttr [[maybe_unused]];
  std::unique_ptr<mlir::Region> bodyRegion = std::make_unique<mlir::Region>();
  TokenType tokenType;
  ClosureType closureType;
  enum class Keyword {
    vtable,
    body,
    token,
    unknown,
  };
  constexpr size_t NUM_KEYWORDS = static_cast<size_t>(Keyword::unknown);
  std::array<bool, NUM_KEYWORDS> appeared{false};
  // Parse return type
  if (parser.parseArrow())
    return mlir::failure();
  if (parser.parseCustomTypeWithFallback(closureType))
    return mlir::failure();
  if (parser.parseLBrace())
    return mlir::failure();
  // Parse order insensitive fields (vtable, body, token)
  for (;;) {
    llvm::SMLoc keywordLoc = parser.getCurrentLocation();
    llvm::StringRef keyword;
    if (llvm::failed(parser.parseOptionalKeyword(&keyword)))
      break;
    auto dispatch = llvm::StringSwitch<Keyword>(keyword)
                        .Case("vtable", Keyword::vtable)
                        .Case("body", Keyword::body)
                        .Case("token", Keyword::token)
                        .Default(Keyword::unknown);
    if (appeared[static_cast<size_t>(dispatch)])
      return parser.emitError(keywordLoc,
                              "keyword " + keyword + " appeared twice");
    appeared[static_cast<size_t>(dispatch)] = true;
    switch (dispatch) {
    case Keyword::vtable: {
      if (parser.parseLParen())
        return mlir::failure();
      if (parser.parseCustomAttributeWithFallback(vtableAttr))
        return mlir::failure();
      if (parser.parseRParen())
        return mlir::failure();
      break;
    }
    case Keyword::body: {
      if (parser.parseRegion(*bodyRegion))
        return mlir::failure();
      break;
    }
    case Keyword::token: {
      if (parser.parseLParen())
        return mlir::failure();
      if (parser.parseOperand(tokenOperand))
        return mlir::failure();
      if (parser.parseColon())
        return mlir::failure();
      if (parser.parseCustomTypeWithFallback(tokenType))
        return mlir::failure();
      if (parser.parseRParen())
        return mlir::failure();
      break;
    }
    case Keyword::unknown:
      return parser.emitError(keywordLoc, "unknown keyword: " + keyword);
    }
  }

  if (parser.parseRBrace())
    return mlir::failure();

  if (!appeared[static_cast<size_t>(Keyword::token)])
    return parser.emitError(operationLoc, "token is required");

  if (vtableAttr)
    result.addAttribute("vtable", vtableAttr);
  result.addRegion(std::move(bodyRegion));
  result.addTypes(closureType);
  if (llvm::failed(parser.resolveOperands({tokenOperand}, tokenType,
                                          operationLoc, result.operands)))
    return mlir::failure();

  return mlir::success();
}

void ReussirClosureCreateOp::print(mlir::OpAsmPrinter &p) {
  // Print return type
  p << " -> ";
  p.printStrippedAttrOrType(getClosure().getType());
  p << " {";
  p.increaseIndent();

  // Print order insensitive fields: token, vtable, body
  // Token is required
  p.printNewline();
  p << " token (";
  p.printOperand(getToken());
  p << " : ";
  p.printStrippedAttrOrType(getToken().getType());
  p << ")";

  // Print vtable if present
  if (getVtableAttr()) {
    p.printNewline();
    p << " vtable (" << getVtableAttr() << ")";
  }

  // Print body region if present
  if (!getBody().empty()) {
    p.printNewline();
    p << " body ";
    p.printRegion(getBody());
  }

  p.decreaseIndent();
  p.printNewline();
  p << "}";
}

//===-----------------------------------------------------------------------===//
// ClosureCreateOp verification
//===-----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCreateOp::verify() {
  bool outlinedFlag = isOutlined();
  bool inlinedFlag = isInlined();
  ClosureType closureType = getClosure().getType();
  if (!outlinedFlag && !inlinedFlag)
    return emitOpError("closure must be outlined or inlined");
  ClosureBoxType closureBoxType = getClosureBoxType();
  auto dataLayout = mlir::DataLayout::closest(this->getOperation());
  auto closureBoxSize = dataLayout.getTypeSize(closureBoxType);
  auto closureBoxAlignment = dataLayout.getTypeABIAlignment(closureBoxType);
  TokenType tokenType = getToken().getType();
  if (closureBoxSize != tokenType.getSize())
    return emitOpError("closure box size must match token size")
           << ", closure box size: " << closureBoxSize.getFixedValue()
           << ", token size: " << tokenType.getSize();
  if (closureBoxAlignment != tokenType.getAlign())
    return emitOpError("closure box alignment must match token alignment")
           << ", closure box alignment: " << closureBoxAlignment
           << ", token alignment: " << tokenType.getAlign();
  // Check that region arguments match the closure input types
  if (inlinedFlag) {
    auto types = getBody().getArgumentTypes();
    if (types.size() != closureType.getInputTypes().size())
      return emitOpError("inlined closure body must have the same number of "
                         "arguments as the closure input types");
    if (llvm::any_of(llvm::zip(types, closureType.getInputTypes()),
                     [](auto &&argAndType) {
                       auto [argTy, type] = argAndType;
                       return argTy != type;
                     }))
      return emitOpError(
          "inlined closure body arguments must match the closure input types");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ClosureCreateOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCreateOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  // NYI for body
  if (getVtableAttr()) {
    // TODO: Verify that the vtable symbol exists and is valid
    // This would typically check that the referenced vtable operation exists
    // and has the correct signature matching the closure type
  }
  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// ClosureCreateOp helper methods
//===-----------------------------------------------------------------------===//
bool ReussirClosureCreateOp::isOutlined() {
  return getBody().empty() && getVtableAttr();
}

bool ReussirClosureCreateOp::isInlined() {
  return !getBody().empty() && !getVtableAttr();
}

ClosureBoxType ReussirClosureCreateOp::getClosureBoxType() {
  ClosureType closureType = getClosure().getType();
  return ClosureBoxType::get(getContext(), closureType.getInputTypes());
}

mlir::FlatSymbolRefAttr ReussirClosureCreateOp::getTrivialForwardingTarget() {
  // Only inlined closures can be trivially forwarding
  if (!isInlined())
    return nullptr;

  // Get the body region
  auto &body = getBody();
  if (body.empty())
    return nullptr;

  // Get the block in the body region
  auto &block = body.front();

  // Check if the block has exactly one operation (the terminator)
  if (block.getOperations().size() != 2) // 1 operation + 1 terminator
    return nullptr;

  // Get the first operation (skip the terminator)
  auto &firstOp = block.getOperations().front();

  // Check if it's a function call operation
  auto callOp = llvm::dyn_cast<mlir::func::CallOp>(firstOp);
  if (!callOp)
    return nullptr;

  // Get the closure type to check argument types
  ClosureType closureType = getClosure().getType();
  auto closureInputTypes = closureType.getInputTypes();
  auto closureOutputType = closureType.getOutputType();

  // Check if the call operation has the same number of arguments as the closure
  auto callArgs = callOp.getOperands();
  if (callArgs.size() != closureInputTypes.size())
    return nullptr;

  // Check if all argument types match
  for (size_t i = 0; i < callArgs.size(); ++i)
    if (callArgs[i].getType() != closureInputTypes[i])
      return nullptr;

  // Check if the call operation has the same return type as the closure
  auto callResults = callOp.getResults();
  if (closureOutputType) {
    // Closure has a return type
    if (callResults.size() != 1)
      return nullptr;
    if (callResults[0].getType() != closureOutputType)
      return nullptr;
  } else if (!callResults.empty())
    return nullptr;

  // Check if the yield operation yields the result of the call
  auto yieldOp = llvm::dyn_cast<ReussirClosureYieldOp>(block.getTerminator());
  if (!yieldOp)
    return nullptr;

  if (closureOutputType) {
    // Closure has a return type, so yield should yield the call result
    if (!yieldOp.getValue() || yieldOp.getValue() != callResults[0])
      return nullptr;
  } else {
    // Closure has no return type, so yield should not yield anything
    if (yieldOp.getValue())
      return nullptr;
  }

  // All checks passed, return the function name
  return callOp.getCalleeAttr();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Vtable Op
//===----------------------------------------------------------------------===//
// ClosureVtableOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureVtableOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  // NYI for body
  if (getFuncAttr()) {
    // TODO: Verify that the func symbol exists and is valid
    // This would typically check that the referenced function operation exists
    // and has the correct signature matching the closure type
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Yield Op
//===----------------------------------------------------------------------===//
// ClosureYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureYieldOp::verify() {
  // Get the parent closure operation
  auto parentOp = getOperation()->getParentOfType<ReussirClosureCreateOp>();
  if (!parentOp)
    return emitOpError(
        "closure yield must be inside a closure create operation");

  // Get the closure type to determine if it has a return value
  ClosureType closureType = parentOp.getClosure().getType();
  mlir::Type expectedReturnType = closureType.getOutputType();

  // Check consistency between yield value and closure return type
  if (expectedReturnType) {
    // Closure has a return type, so yield must provide a value
    if (!getValue())
      return emitOpError("closure has return type ")
             << expectedReturnType << " but yield provides no value";

    // Check that the yielded value type matches the closure return type
    mlir::Type yieldedType = getValue().getType();
    if (yieldedType != expectedReturnType)
      return emitOpError("yielded type must match closure return type, ")
             << "yielded type: " << yieldedType
             << ", expected type: " << expectedReturnType;
  } else {
    // Closure has no return type, so yield must not provide a value
    if (getValue())
      return emitOpError(
                 "closure has no return type but yield provides value of type ")
             << getValue().getType();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Apply Op
//===----------------------------------------------------------------------===//
// ClosureApplyOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureApplyOp::verify() {
  ClosureType closureType = getClosure().getType();
  mlir::Type argType = getArg().getType();

  // Get the input types of the closure
  auto inputTypes = closureType.getInputTypes();

  // Check that the closure has at least one input type
  if (inputTypes.empty())
    return emitOpError("cannot apply to closure with no input types");

  // Check that the argument type matches the first input type
  mlir::Type expectedArgType = inputTypes[0];
  if (argType != expectedArgType)
    return emitOpError("argument type must match first closure input type, ")
           << "argument type: " << argType
           << ", expected type: " << expectedArgType;

  // Verify the result type
  ClosureType resultType = getApplied().getType();

  // The result closure should have one less input type
  auto expectedInputTypes = inputTypes.drop_front(1);
  auto resultInputTypes = resultType.getInputTypes();

  if (resultInputTypes.size() != expectedInputTypes.size())
    return emitOpError("result closure must have one less input type, ")
           << "expected " << expectedInputTypes.size() << " input types, "
           << "but got " << resultInputTypes.size();

  // Check that the remaining input types match
  for (size_t i = 0; i < expectedInputTypes.size(); ++i) {
    if (resultInputTypes[i] != expectedInputTypes[i])
      return emitOpError("result closure input types must match remaining "
                         "input types, ")
             << "mismatch at index " << i << ": expected "
             << expectedInputTypes[i] << ", but got " << resultInputTypes[i];
  }

  // Check that the output types match
  mlir::Type closureOutputType = closureType.getOutputType();
  mlir::Type resultOutputType = resultType.getOutputType();

  if (closureOutputType != resultOutputType)
    return emitOpError("result closure output type must match original closure "
                       "output type, ")
           << "original output type: " << closureOutputType
           << ", result output type: " << resultOutputType;

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
