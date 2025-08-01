//===-- ReussirTypes.cppm - Reussir types implementation --------*- c++ -*-===//
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
module;

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Reussir/IR/ReussirOpsTypes.cpp.inc"

namespace reussir {
//===----------------------------------------------------------------------===//
// RecordType
//===----------------------------------------------------------------------===//
llvm::LogicalResult
RecordType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<mlir::Type> members,
                   llvm::ArrayRef<reussir::CapabilityAttr> memberCapabilities,
                   mlir::StringAttr name, bool complete,
                   reussir::RecordKindAttr kind,
                   reussir::CapabilityAttr defaultCapability) {
  if (memberCapabilities.size() != members.size()) {
    emitError().attachNote()
        << "Number of member capabilities must match number of members";
    return mlir::failure();
  }
  if (complete) {
    if (members.empty()) {
      emitError().attachNote()
          << "Record type must have at least one member when complete";
      return mlir::failure();
    }
    if (!defaultCapability) {
      emitError().attachNote()
          << "Default capability must be specified for complete record types";
      return mlir::failure();
    }
  }
  if (complete && defaultCapability.getValue() != reussir::Capability::Shared &&
      defaultCapability.getValue() != reussir::Capability::Value &&
      defaultCapability.getValue() != reussir::Capability::Default) {
    emitError().attachNote()
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
  llvm::SmallVector<reussir::CapabilityAttr> memberCapabilities;
  CapabilityAttr defaultCapability;
  RecordKindAttr kind;
  StringAttr name;
  bool incomplete = true;
  mlir::MLIRContext *context = parser.getContext();

  // Parse '<' to start the type.
  if (parser.parseLess().failed())
    return {};

  // Now parse the kind of the record.
  if (parser.parseAttribute(kind).failed()) {
    parser.emitError(loc, "expected record kind");
    return {};
  }

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

  // Start parsing member fields.
  if (parser.parseOptionalKeyword("incomplete").failed()) {
    incomplete = false;
    // First, check if default capability is specified.
    if (!parser.parseOptionalAttribute(defaultCapability).has_value())
      defaultCapability = reussir::CapabilityAttr::get(
          parser.getContext(), reussir::Capability::Default);
    // Now parse the members and their capabilities.
    const auto delimiter = AsmParser::Delimiter::Braces;
    const auto parseElementFn = [&parser, &members, &memberCapabilities]() {
      if (!parser.parseOptionalAttribute(memberCapabilities.emplace_back())
               .has_value())
        memberCapabilities.back() = reussir::CapabilityAttr::get(
            parser.getContext(), reussir::Capability::Default);
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
  ArrayRef<reussir::CapabilityAttr> memberCapabilitiesRef{memberCapabilities};

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
    if (!result.getComplete())
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
    if (getDefaultCapability().getValue() != reussir::Capability::Default)
      printer << getDefaultCapability() << ' ';
    printer << '{';
    llvm::interleaveComma(llvm::zip(getMembers(), getMemberCapabilities()),
                          printer, [&](auto memberAndCap) {
                            auto [member, cap] = memberAndCap;
                            if (cap.getValue() != reussir::Capability::Default)
                              printer << cap << ' ';
                            printer.printType(member);
                          });
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
llvm::ArrayRef<reussir::CapabilityAttr>
RecordType::getMemberCapabilities() const {
  return getImpl()->memberCapabilities;
}
mlir::StringAttr RecordType::getName() const { return getImpl()->name; }
bool RecordType::getComplete() const { return getImpl()->complete; }
reussir::RecordKindAttr RecordType::getKind() const { return getImpl()->kind; }
reussir::CapabilityAttr RecordType::getDefaultCapability() const {
  return getImpl()->defaultCapability;
}
//===----------------------------------------------------------------------===//
// RecordType Mutations
//===----------------------------------------------------------------------===//
void RecordType::complete(
    llvm::ArrayRef<mlir::Type> members,
    llvm::ArrayRef<reussir::CapabilityAttr> memberCapabilities,
    reussir::CapabilityAttr defaultCapability) {
  if (mutate(members, memberCapabilities, defaultCapability).failed())
    llvm_unreachable("failed to complete record");
}
//===----------------------------------------------------------------------===//
// RecordType DataLayoutInterface
//===----------------------------------------------------------------------===//
llvm::TypeSize
RecordType::getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                              ::mlir::DataLayoutEntryListRef params) const {
  llvm_unreachable("Not implemented yet");
}
uint64_t
RecordType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                            ::mlir::DataLayoutEntryListRef params) const {
  llvm_unreachable("Not implemented yet");
}

uint64_t
RecordType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  llvm_unreachable("Not implemented yet");
}

} // namespace reussir

export module Reussir.IR.Types;
