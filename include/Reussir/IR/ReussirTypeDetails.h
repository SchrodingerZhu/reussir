//===-- ReussirTypeDetails.h - Reussir type details impl --------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the types used in the Reussir dialect (internal
// details).
//
//===----------------------------------------------------------------------===//
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"

namespace reussir {
namespace detail {

//===----------------------------------------------------------------------===//
// RecordTypeStorage
//===----------------------------------------------------------------------===//
//
// We manually define the storage class for RecordType to handle self-references
// in the members and memberCapabilities arrays. Named structures can be
// initialized as incomplete such that they can be referred by their own.
//
//===----------------------------------------------------------------------===//

struct RecordTypeStorage : public mlir::TypeStorage {
  llvm::ArrayRef<mlir::Type> members;
  llvm::ArrayRef<reussir::CapabilityAttr> memberCapabilities;
  mlir::StringAttr name;
  bool complete;
  reussir::RecordKindAttr kind;
  reussir::CapabilityAttr defaultCapability;

  using KeyTy = RecordTypeStorage;

  RecordTypeStorage(llvm::ArrayRef<mlir::Type> members,
                    llvm::ArrayRef<reussir::CapabilityAttr> memberCapabilities,
                    mlir::StringAttr name, bool complete,
                    reussir::RecordKindAttr kind,
                    reussir::CapabilityAttr defaultCapability)
      : members(members), memberCapabilities(memberCapabilities), name(name),
        complete(complete), kind(kind), defaultCapability(defaultCapability) {}

  RecordTypeStorage(const KeyTy &key) = default;

  KeyTy getAsKey() const { return *this; }

  bool operator==(const KeyTy &other) const {
    if (name)
      return name == other.name && kind == other.kind;
    return members == other.members &&
           memberCapabilities == other.memberCapabilities &&
           kind == other.kind && defaultCapability == other.defaultCapability &&
           complete == other.complete;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name, key.kind);
    return llvm::hash_combine(key.members, key.memberCapabilities, key.kind,
                              key.defaultCapability, key.complete);
  }

  static RecordTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<RecordTypeStorage>()) RecordTypeStorage(key);
  }

  /// Mutates the members and attributes an identified record.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous records are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete record does not
  /// change the record.
  llvm::LogicalResult
  mutate(mlir::TypeStorageAllocator &allocator,
         llvm::ArrayRef<mlir::Type> members,
         llvm::ArrayRef<reussir::CapabilityAttr> memberCapabilities,
         reussir::CapabilityAttr defaultCapability) {

    // Anonymous records cannot mutate.
    if (!name)
      return llvm::failure();

    // Mutation of complete records are allowed if they change nothing.
    if (complete)
      return llvm::success(members == this->members &&
                           memberCapabilities == this->memberCapabilities &&
                           defaultCapability == this->defaultCapability);

    // Mutate incomplete records.
    this->members = allocator.copyInto(members);
    this->memberCapabilities = allocator.copyInto(memberCapabilities);
    this->defaultCapability = defaultCapability;
    this->complete = true;
    return llvm::success();
  }
};

} // namespace detail
} // namespace reussir
