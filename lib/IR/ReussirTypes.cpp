//===-- ReussirTypes.cpp - Reussir types implementation ---------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the types used in the Reussir dialect.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <bit>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <optional>
#include <tuple>

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Reussir/IR/ReussirOpsTypes.cpp.inc"

// Macro to generate standard DataLayoutInterface implementations for
// pointer-like types
#define REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(TypeName)                   \
  llvm::TypeSize TypeName::getTypeSizeInBits(                                  \
      const mlir::DataLayout &dataLayout,                                      \
      [[maybe_unused]] mlir::DataLayoutEntryListRef params) const {            \
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());               \
    return dataLayout.getTypeSizeInBits(ptrTy);                                \
  }                                                                            \
                                                                               \
  uint64_t TypeName::getABIAlignment(                                          \
      const mlir::DataLayout &dataLayout,                                      \
      [[maybe_unused]] mlir::DataLayoutEntryListRef params) const {            \
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());               \
    return dataLayout.getTypeABIAlignment(ptrTy);                              \
  }                                                                            \
                                                                               \
  uint64_t TypeName::getPreferredAlignment(                                    \
      const mlir::DataLayout &dataLayout,                                      \
      [[maybe_unused]] mlir::DataLayoutEntryListRef params) const {            \
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());               \
    return dataLayout.getTypePreferredAlignment(ptrTy);                        \
  }

namespace reussir {
//===----------------------------------------------------------------------===//
// Common Parser/Printer Helpers
//===----------------------------------------------------------------------===//
template <typename T, Capability DefaultCap = reussir::Capability::unspecified>
mlir::Type parseTypeWithCapabilityAndAtomicKind(mlir::AsmParser &parser) {
  using namespace mlir;
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Location encLoc = parser.getEncodedSourceLoc(loc);
  if (parser.parseLess().failed())
    return {};
  Type eleTy;
  if (parser.parseType(eleTy).failed())
    return {};
  std::optional<reussir::Capability> capability;
  std::optional<reussir::AtomicKind> atomicKind;
  llvm::StringRef keyword;
  while (parser.parseOptionalKeyword(&keyword).succeeded()) {
    if (std::optional<reussir::Capability> cap = symbolizeCapability(keyword)) {
      if (capability) {
        parser.emitError(parser.getCurrentLocation(),
                         "Capability is already specified");
        return {};
      }
      capability = cap;
    } else if (std::optional<reussir::AtomicKind> kind =
                   symbolizeAtomicKind(keyword)) {
      if (atomicKind) {
        parser.emitError(parser.getCurrentLocation(),
                         "AtomicKind is already specified");
        return {};
      }
      atomicKind = kind;
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "Unknown attribute in RcType: " + keyword);
      return {};
    }
  }
  Capability capValue = capability ? *capability : DefaultCap;
  AtomicKind atomicValue =
      atomicKind ? *atomicKind : reussir::AtomicKind::normal;
  return T::getChecked(encLoc, parser.getContext(), eleTy, capValue,
                       atomicValue);
}

template <typename T, Capability DefaultCap = reussir::Capability::unspecified>
void printTypeWithCapabilityAndAtomicKind(mlir::AsmPrinter &printer,
                                          const T &type) {
  printer << "<";
  printer.printType(type.getElementType());
  if (type.getCapability() != DefaultCap)
    printer << ' ' << type.getCapability();

  if (type.getAtomicKind() != reussir::AtomicKind::normal)
    printer << ' ' << type.getAtomicKind();
  printer << ">";
}
//===----------------------------------------------------------------------===//
// isNonNullPointerType
//===----------------------------------------------------------------------===//
bool isNonNullPointerType(mlir::Type type) {
  if (!type)
    return false;
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<TokenType, RcType, RecordType, RawPtrType, RefType>(
          [](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}
//===----------------------------------------------------------------------===//
// deriveCompoundSizeAndAlignment
//===----------------------------------------------------------------------===//
std::optional<std::tuple<llvm::TypeSize, llvm::Align, mlir::Type>>
deriveCompoundSizeAndAlignment(mlir::MLIRContext *context,
                               llvm::ArrayRef<mlir::Type> members,
                               llvm::ArrayRef<Capability> memberCapabilities,
                               const mlir::DataLayout &dataLayout) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  llvm::TypeSize resultSize = llvm::TypeSize::getFixed(0);
  llvm::Align resultAlignment{1};
  mlir::Type memberWithLargestAlignment;
  if (memberCapabilities.size() != members.size())
    llvm::report_fatal_error(
        "Number of member capabilities must match number of members");
  for (auto [rawMember, rawCapability] :
       llvm::zip(members, memberCapabilities)) {
    if (!rawMember)
      continue;
    Capability capability = rawCapability;
    if (auto recordTy = llvm::dyn_cast<RecordType>(rawMember)) {
      if (capability == Capability::unspecified)
        capability = recordTy.getDefaultCapability();
      if (capability == Capability::unspecified)
        capability = Capability::shared;
    }
    mlir::Type member;
    if (capability == Capability::flex || capability == Capability::rigid ||
        capability == Capability::shared || capability == Capability::field)
      member = ptrTy;
    else
      member = rawMember;
    llvm::TypeSize memberSize = dataLayout.getTypeSize(member);
    if (!memberSize.isFixed())
      return std::nullopt;

    uint64_t memberAlignment = dataLayout.getTypeABIAlignment(member);
    llvm::Align memberAlign(memberAlignment);
    uint64_t alignedSize = llvm::alignTo(resultSize, memberAlign);
    if (memberAlign > resultAlignment) {
      resultAlignment = memberAlign;
      memberWithLargestAlignment = member;
    }
    resultSize =
        llvm::TypeSize::getFixed(alignedSize + memberSize.getFixedValue());
  }
  resultSize = llvm::alignTo(resultSize, resultAlignment.value());
  return std::make_optional(
      std::make_tuple(resultSize, resultAlignment, memberWithLargestAlignment));
}

//===----------------------------------------------------------------------===//
// RecordType
//===----------------------------------------------------------------------===//
llvm::LogicalResult
RecordType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<mlir::Type> members,
                   llvm::ArrayRef<reussir::Capability> memberCapabilities,
                   mlir::StringAttr name, bool complete,
                   reussir::RecordKind kind,
                   reussir::Capability defaultCapability) {
  if (memberCapabilities.size() != members.size()) {
    emitError() << "Number of member capabilities must match number of members";
    return mlir::failure();
  }
  for (size_t i = 0; i < members.size(); ++i) {
    const auto &member = members[i];
    const auto &capability = memberCapabilities[i];

    if (!member) {
      emitError() << "Members must not be null";
      return mlir::failure();
    }

    if (llvm::isa<RcType>(member) || llvm::isa<RefType>(member)) {
      emitError()
          << "Members must not be Rc or Ref types, use capability instead";
      return mlir::failure();
    }

    if (capability == reussir::Capability::flex) {
      emitError() << "Flex capability is not allowed in record members";
      return mlir::failure();
    }
  }
  if (complete && defaultCapability != reussir::Capability::shared &&
      defaultCapability != reussir::Capability::value &&
      defaultCapability != reussir::Capability::unspecified) {
    emitError()
        << "Default capability must be either Shared, Value, or Default";
    return mlir::failure();
  }
  return mlir::success();
}
//===----------------------------------------------------------------------===//
// RecordType Parse/Print
//===----------------------------------------------------------------------===//
mlir::Type RecordType::parse(mlir::AsmParser &parser) {
  using namespace mlir;
  llvm::FailureOr<AsmParser::CyclicParseReset> cyclicParseGuard;
  const llvm::SMLoc loc = parser.getCurrentLocation();
  const Location encLoc = parser.getEncodedSourceLoc(loc);
  llvm::SmallVector<mlir::Type> members;
  llvm::SmallVector<reussir::Capability> memberCapabilities;
  Capability defaultCapability;
  RecordKind kind;
  StringAttr name;
  bool incomplete = true;
  mlir::MLIRContext *context = parser.getContext();

  // Parse '<' to start the type.
  if (parser.parseLess().failed())
    return {};

  // Now parse the kind of the record.
  FailureOr<RecordKind> kindOrError = FieldParser<RecordKind>::parse(parser);
  if (failed(kindOrError))
    return {};
  kind = *kindOrError;

  // Try parse name. It can be empty.
  parser.parseOptionalAttribute(name);

  // Check if the record type ends with name. If so, this is a self-referential
  // case. In this case, the cyclic parsing process must be already started. If
  // not, we fail the parsing.
  if (name && parser.parseOptionalGreater().succeeded()) {
    RecordType type = getChecked(encLoc, context, name, kind);
    if (succeeded(parser.tryStartCyclicParse(type))) {
      parser.emitError(loc, "invalid self-reference within record");
      return {};
    }
    return type;
  }

  // This is a named record definition: ensure name has not been parsed yet.
  // `tryStartCyclicParse` will fail if there is already a parsing in progress.
  if (name) {
    RecordType type = getChecked(encLoc, context, name, kind);
    cyclicParseGuard = parser.tryStartCyclicParse(type);
    if (failed(cyclicParseGuard)) {
      parser.emitError(loc, "record already defined");
      return {};
    }
  }

  auto parseOptionalCapability = [kind](mlir::AsmParser &parser,
                                        bool forField =
                                            true) -> FailureOr<Capability> {
    if (parser.parseOptionalLSquare().succeeded()) {
      FailureOr<std::optional<Capability>> capOrError =
          FieldParser<std::optional<Capability>>::parse(parser);
      if (failed(capOrError))
        return mlir::failure();
      if (failed(parser.parseRSquare()))
        return mlir::failure();
      if (capOrError->has_value())
        return capOrError->value();
    }
    return (kind == RecordKind::variant && forField)
               ? reussir::Capability::value
               : reussir::Capability::unspecified;
  };
  // Start parsing member fields.
  if (parser.parseOptionalKeyword("incomplete").failed()) {
    incomplete = false;
    // First, check if default capability is specified.
    FailureOr<Capability> defaultCapOrError =
        parseOptionalCapability(parser, false);
    if (failed(defaultCapOrError))
      return {};

    defaultCapability = defaultCapOrError.value();

    // Now parse the members and their capabilities.
    const auto delimiter = AsmParser::Delimiter::Braces;
    const auto parseElementFn = [&parser, &members, &memberCapabilities,
                                 &parseOptionalCapability]() -> ParseResult {
      FailureOr<reussir::Capability> capOrError =
          parseOptionalCapability(parser, true);
      if (failed(capOrError))
        return mlir::failure();
      else {
        if (*capOrError == reussir::Capability::flex ||
            *capOrError == reussir::Capability::rigid) {
          parser.emitError(
              parser.getCurrentLocation(),
              "flex or rigid capabilities are not allowed in record members");
          return mlir::failure();
        }
      }
      memberCapabilities.push_back(capOrError.value());
      return parser.parseType(members.emplace_back());
    };
    if (parser.parseCommaSeparatedList(delimiter, parseElementFn).failed())
      return {};
  }
  // end the member parsing.
  if (parser.parseGreater().failed())
    return {};

  // Start creating the record type.
  RecordType result;
  ArrayRef<Type> membersRef{members};
  ArrayRef<reussir::Capability> memberCapabilitiesRef{memberCapabilities};

  if (name && incomplete) {
    // Named incomplete record.
    result = getChecked(encLoc, context, name, kind);
  } else if (!name && !incomplete) {
    // Anonymous complete record.
    result = getChecked(encLoc, context, membersRef, memberCapabilitiesRef,
                        kind, defaultCapability);
  } else if (!incomplete) {
    // Named complete record.
    result = getChecked(encLoc, context, membersRef, memberCapabilitiesRef,
                        name, kind, defaultCapability);
    // If the record has a self-reference, its type already exists in a
    // incomplete state. In this case, we must complete it.
    if (result && !result.getComplete())
      result.complete(membersRef, memberCapabilitiesRef, defaultCapability);
  } else { // anonymous & incomplete
    parser.emitError(loc, "anonymous records must be complete");
    return {};
  }

  return result;
}

void RecordType::print(::mlir::AsmPrinter &printer) const {
  llvm::FailureOr<mlir::AsmPrinter::CyclicPrintReset> cyclicPrintGuard;
  // Start printing the record type.
  printer << '<';
  // Print the kind of the record.
  printer << getKind() << ' ';

  if (getName())
    printer << getName();

  // Current type has already been printed: print as self reference.
  cyclicPrintGuard = printer.tryStartCyclicPrint(*this);
  if (failed(cyclicPrintGuard)) {
    printer << '>';
    return;
  }

  printer << ' ';

  if (!getComplete())
    printer << "incomplete";
  else {
    if (getDefaultCapability() != reussir::Capability::unspecified)
      printer << '[' << getDefaultCapability() << "] ";
    printer << '{';
    if (!getMembers().empty()) {
      llvm::interleaveComma(llvm::zip(getMembers(), getMemberCapabilities()),
                            printer, [&](auto memberAndCap) {
                              auto [member, cap] = memberAndCap;
                              auto defaultCapability =
                                  reussir::Capability::unspecified;
                              if (getKind() == RecordKind::variant)
                                defaultCapability = reussir::Capability::value;
                              if (cap != defaultCapability)
                                printer << '[' << cap << "] ";
                              printer << member;
                            });
    }
    printer << '}';
  }
  // End the record type.
  printer << '>';
}

//===----------------------------------------------------------------------===//
// RecordType Getters
//===----------------------------------------------------------------------===//
llvm::ArrayRef<mlir::Type> RecordType::getMembers() const {
  return getImpl()->members;
}
llvm::ArrayRef<reussir::Capability> RecordType::getMemberCapabilities() const {
  return getImpl()->memberCapabilities;
}
mlir::StringAttr RecordType::getName() const { return getImpl()->name; }
bool RecordType::getComplete() const { return getImpl()->complete; }
reussir::RecordKind RecordType::getKind() const { return getImpl()->kind; }
reussir::Capability RecordType::getDefaultCapability() const {
  return getImpl()->defaultCapability;
}
//===----------------------------------------------------------------------===//
// RecordType Mutations
//===----------------------------------------------------------------------===//
void RecordType::complete(
    llvm::ArrayRef<mlir::Type> members,
    llvm::ArrayRef<reussir::Capability> memberCapabilities,
    reussir::Capability defaultCapability) {
  if (mutate(members, memberCapabilities, defaultCapability).failed())
    llvm_unreachable("failed to complete record");
}

//===----------------------------------------------------------------------===//
// RecordType GetElementRegionSizeAndAlignment
//===----------------------------------------------------------------------===//
RecordType::LayoutInfo RecordType::getElementRegionLayoutInfo(
    const mlir::DataLayout &dataLayout) const {
  if (isCompound()) {
    auto derived = deriveCompoundSizeAndAlignment(
        getContext(), getMembers(), getMemberCapabilities(), dataLayout);
    if (!derived)
      llvm_unreachable("RecordType must have a fixed size");
    auto [sizeInBytes, alignment, memberWithLargestAlignment] = *derived;
    return {sizeInBytes, alignment, memberWithLargestAlignment};
  }
  llvm::TypeSize largestSize = llvm::TypeSize::getZero();
  llvm::Align largestAlignment = llvm::Align(1);
  mlir::Type memberWithLargestAlignment;
  for (auto [rawMember, cap] :
       llvm::zip(getMembers(), getMemberCapabilities())) {
    if (!rawMember)
      continue;
    mlir::Type member = getProjectedType(rawMember, cap, Capability::value);
    llvm::TypeSize memberSize = dataLayout.getTypeSize(member);
    if (!memberSize.isFixed())
      llvm_unreachable("RecordType must have a fixed size");
    largestSize = std::max(largestSize, memberSize);
    llvm::Align memberAlignment{dataLayout.getTypeABIAlignment(member)};
    if (memberAlignment > largestAlignment) {
      largestAlignment = memberAlignment;
      memberWithLargestAlignment = member;
    }
  }
  llvm::TypeSize size = llvm::alignTo(largestSize, largestAlignment.value());
  return {size, largestAlignment, memberWithLargestAlignment};
}

//===----------------------------------------------------------------------===//
// RecordType DataLayoutInterface
//===----------------------------------------------------------------------===//
llvm::TypeSize
RecordType::getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                              ::mlir::DataLayoutEntryListRef params) const {
  auto [size, align, _] = getElementRegionLayoutInfo(dataLayout);
  if (isCompound())
    return size * 8; // Convert to bits

  mlir::IndexType tagType =
      mlir::IndexType::get(getContext()); // Use IndexType for tag
  llvm::TypeSize tagSize = dataLayout.getTypeSize(tagType);
  llvm::Align tagAlign{dataLayout.getTypeABIAlignment(tagType)};
  llvm::Align finalAlign =
      std::max(align, tagAlign); // Use the larger of the two alignments
  llvm::TypeSize headerSize =
      llvm::alignTo(tagSize, align.value()); // Align the tag size
  llvm::TypeSize totalSize =
      llvm::alignTo(size + headerSize, finalAlign.value());
  return totalSize * 8; // Convert to bits
}

uint64_t
RecordType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                            ::mlir::DataLayoutEntryListRef params) const {
  auto [size, align, _] = getElementRegionLayoutInfo(dataLayout);
  if (isCompound())
    return align.value();

  mlir::IndexType tagType =
      mlir::IndexType::get(getContext()); // Use IndexType for tag
  uint64_t tagAlignment = dataLayout.getTypeABIAlignment(tagType);
  uint64_t finalAlignment = std::max(align.value(), tagAlignment);
  return finalAlignment;
}

uint64_t
RecordType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getABIAlignment(dataLayout, params);
}

//===----------------------------------------------------------------------===//
// Reussir Dialect
//===----------------------------------------------------------------------===//

void reussir::ReussirDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Reussir/IR/ReussirOpsTypes.cpp.inc"
      >();
}
mlir::Type ReussirDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;
  mlir::Type genType;

  // Try to parse as a tablegen'd type.
  mlir::OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Type is not tablegen'd: try to parse as a raw C++ type.
  return llvm::StringSwitch<llvm::function_ref<mlir::Type()>>(mnemonic)
      .Case("record", [&] { return RecordType::parse(parser); })
      .Default([&] {
        parser.emitError(typeLoc) << "unknown reussir type: " << mnemonic;
        return mlir::Type{};
      })();
}
void ReussirDialect::printType(mlir::Type type,
                               mlir::DialectAsmPrinter &printer) const {
  // Try to print as a tablegen'd type.
  if (generatedTypePrinter(type, printer).succeeded())
    return;

  // Type is not tablegen'd: try printing as a raw C++ type.
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<RecordType>([&](RecordType type) {
        printer << type.getMnemonic();
        type.print(printer);
      })
      .Default([](mlir::Type) {
        llvm::report_fatal_error("printer is missing a handler for this type");
      });
}

//===----------------------------------------------------------------------===//
// Token Type
//===----------------------------------------------------------------------===//
// TokenType validation
//===----------------------------------------------------------------------===//
mlir::LogicalResult
TokenType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                  size_t align, size_t size) {
  if (align == 0) {
    emitError() << "Token alignment must be non-zero";
    return mlir::failure();
  }
  if (!std::has_single_bit(align)) {
    emitError() << "Token alignment must be a power of two";
    return mlir::failure();
  }

  if (size % align != 0) {
    emitError() << "Token size must be a multiple of alignment";
    return mlir::failure();
  }
  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TokenType DataLayoutInterface
//===----------------------------------------------------------------------===//
REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(TokenType)

///===---------------------------------------------------------------------===//
// Reussir Region Type
//===----------------------------------------------------------------------===//
// RegionType DataLayoutInterface
//===----------------------------------------------------------------------===//
REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(RegionType)

///===----------------------------------------------------------------------===//
// Reussir RC Type
//===----------------------------------------------------------------------===//
// RcType validation
//===----------------------------------------------------------------------===//
mlir::LogicalResult
RcType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
               mlir::Type eleTy, reussir::Capability capability,
               reussir::AtomicKind atomicKind) {
  if (capability != reussir::Capability::shared &&
      capability != reussir::Capability::flex &&
      capability != reussir::Capability::rigid) {
    emitError() << "Capability must be shared, flex or rigid for RcType";
    return mlir::failure();
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// RcType getInnerBoxType
//===----------------------------------------------------------------------===//
RcBoxType RcType::getInnerBoxType() const {
  bool isFlexOrRigid = getCapability() == ::reussir::Capability::flex ||
                       getCapability() == ::reussir::Capability::rigid;
  return RcBoxType::get(getContext(), getEleTy(), isFlexOrRigid);
}
//===----------------------------------------------------------------------===//
// RcType DataLayoutInterface
//===----------------------------------------------------------------------===//
REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(RcType)
//===----------------------------------------------------------------------===//
// RcType pasrse/print
//===----------------------------------------------------------------------===//
mlir::Type RcType::parse(mlir::AsmParser &parser) {
  return parseTypeWithCapabilityAndAtomicKind<RcType, Capability::shared>(
      parser);
}

void RcType::print(mlir::AsmPrinter &printer) const {
  printTypeWithCapabilityAndAtomicKind<RcType, Capability::shared>(printer,
                                                                   *this);
}

///===----------------------------------------------------------------------===//
// Reussir Nullable Type
//===----------------------------------------------------------------------===//
// ReussirNullableType DataLayoutInterface
//===----------------------------------------------------------------------===//
REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(NullableType);

//===----------------------------------------------------------------------===//
// Reussir Reference Type
//===----------------------------------------------------------------------===//
// RefType Parse/Print
//===----------------------------------------------------------------------===//
mlir::Type RefType::parse(mlir::AsmParser &parser) {
  return parseTypeWithCapabilityAndAtomicKind<RefType>(parser);
}
void RefType::print(mlir::AsmPrinter &printer) const {
  printTypeWithCapabilityAndAtomicKind(printer, *this);
}

//===----------------------------------------------------------------------===//
// RefType DataLayoutInterface
//===----------------------------------------------------------------------===//
REUSSIR_POINTER_LIKE_DATA_LAYOUT_INTERFACE(RefType)

//===----------------------------------------------------------------------===//
// RefType Validation
//===----------------------------------------------------------------------===//
mlir::LogicalResult
RefType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                mlir::Type eleTy, reussir::Capability capability,
                reussir::AtomicKind atomicKind) {
  if (capability == reussir::Capability::value) {
    emitError() << "Capability must not be Value for RefType";
    return mlir::failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Rc Box Type
//===----------------------------------------------------------------------===//
// RcBoxType Parse/Print
//===----------------------------------------------------------------------===//
mlir::Type RcBoxType::parse(mlir::AsmParser &parser) {
  using namespace mlir;
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location encLoc = parser.getEncodedSourceLoc(loc);
  Type eleTy;
  bool regional;

  if (parser.parseLess().failed())
    return {};

  regional = parser.parseOptionalKeyword("regional").succeeded();

  if (parser.parseType(eleTy).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  return RcBoxType::getChecked(encLoc, parser.getContext(), eleTy, regional);
}

void RcBoxType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  if (this->isRegional())
    printer << "regional ";
  printer.printType(getEleTy());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// RcBoxType DataLayoutInterface
//===----------------------------------------------------------------------===//
llvm::TypeSize
RcBoxType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                             mlir::DataLayoutEntryListRef params) const {
  mlir::IndexType type = mlir::IndexType::get(getContext());
  auto derived = isRegional()
                     ? deriveCompoundSizeAndAlignment(
                           getContext(), {type, type, type, getEleTy()},
                           {Capability::value, Capability::value,
                            Capability::value, Capability::value},
                           dataLayout)
                     : deriveCompoundSizeAndAlignment(
                           getContext(), {type, getEleTy()},
                           {Capability::value, Capability::value}, dataLayout);
  if (!derived)
    llvm_unreachable("RcBoxType must have a fixed size");
  auto [sizeInBytes, _x, _y] = *derived;
  return sizeInBytes * 8; // Convert to bits
}

uint64_t RcBoxType::getABIAlignment(const mlir::DataLayout &dataLayout,
                                    mlir::DataLayoutEntryListRef params) const {
  mlir::IndexType type = mlir::IndexType::get(getContext());
  auto derived = isRegional()
                     ? deriveCompoundSizeAndAlignment(
                           getContext(), {type, type, type, getEleTy()},
                           {Capability::value, Capability::value,
                            Capability::value, Capability::value},
                           dataLayout)
                     : deriveCompoundSizeAndAlignment(
                           getContext(), {type, getEleTy()},
                           {Capability::value, Capability::value}, dataLayout);
  if (!derived)
    llvm_unreachable("RcBoxType must have a fixed alignment");
  auto [_x, alignment, _y] = *derived;
  return alignment.value();
}

uint64_t
RcBoxType::getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                 mlir::DataLayoutEntryListRef params) const {
  return getABIAlignment(dataLayout, params);
}
//===----------------------------------------------------------------------===//
// getProjectedType
//===----------------------------------------------------------------------===//
mlir::Type getProjectedType(mlir::Type type, Capability fieldCap,
                            Capability refCap) {
  fieldCap = llvm::TypeSwitch<mlir::Type, Capability>(type)
                 .Case<RecordType>([&](RecordType type) -> Capability {
                   if (fieldCap == Capability::unspecified)
                     fieldCap = type.getDefaultCapability();
                   if (fieldCap == Capability::unspecified)
                     fieldCap = Capability::shared;
                   return fieldCap;
                 })
                 .Default([fieldCap](mlir::Type) {
                   return fieldCap == Capability::unspecified
                              ? Capability::value
                              : fieldCap;
                 });
  if (fieldCap == Capability::field) {
    // Target capability is rigid unless the reference capability is flex
    RcType rcTy = RcType::get(type.getContext(), type,
                              refCap == Capability::flex ? Capability::flex
                                                         : Capability::rigid);
    NullableType nullableTy = NullableType::get(type.getContext(), rcTy);
    return nullableTy;
  }
  if (fieldCap == Capability::shared)
    return RcType::get(type.getContext(), type, Capability::shared);
  if (fieldCap == Capability::rigid)
    return RcType::get(type.getContext(), type, Capability::rigid);
  if (fieldCap == Capability::value)
    return type;
  llvm_unreachable("invalid field capability");
}

} // namespace reussir
